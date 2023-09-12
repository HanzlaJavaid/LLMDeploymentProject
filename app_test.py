from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
#from ggml_model_test import generate, AI_INIT
#from gptq_model_run import generate, AI_INIT, count_tokens,model_name_or_path
from production_model_interface import generate, AI_INIT, count_tokens,model_name_or_path
from pydantic import BaseModel
import json
from typing import List
import datetime
import random
from connect_dynamodb import DynamoDBManager
from dynamodb_json import json_util
import pandas as pd
from finetune_interface import finetune
import os
import sys

chain = AI_INIT(prompt_template="""
SYSTEM: You are a helpful assistant that answers questions of user. Be respectful, dont try to answer things you dont know. Be friendly with the user.

{history}

{input}
ASSISTANT:
""")

def generate_random_id(length):
    number_string = ''.join([str(random.randint(0, 9)) for _ in range(length)])
    return number_string

def conversation_history_format_new(history):
    formatted_messages = "\n".join([f"{msg.role}: {msg.content}" for msg in history])
    print(formatted_messages)
    return formatted_messages

class MessageObject(BaseModel):
    role: str
    content: str

class DataObject(BaseModel):
    model: str
    messages: List[MessageObject]
    temperature: float

class DataFineTuneObject(BaseModel):
	user_message: str
	ai_message: str

class TrainParamsObject(BaseModel):
        current_ds: str


app = FastAPI()
database = DynamoDBManager()
@app.post('/generate')
async def reply(data: DataObject):
    try:
        messages = data.messages
    
        if len(messages)>1:
            history_messages = messages[:-1]
            history_conversation = conversation_history_format_new(history_messages)
        else:
            history_conversation = " "

        new_message = data.messages[-1]

        new_message_formatted = new_message.role + " " + new_message.content

        ai_response = generate(chain=chain, user_input=new_message_formatted,conversation_history = history_conversation)
        
        tokens_generated = count_tokens(ai_response)
        prompt_tokens = count_tokens(history_conversation + new_message_formatted)
        all_tokens = tokens_generated + prompt_tokens

        unix_timestamp = str(datetime.datetime.now())
        
        completion_id = generate_random_id(6)

        response = {

                "id":"chatcmpl-"+completion_id,
                "object":"chat.completion",
                "created": unix_timestamp,
                "model": model_name_or_path,
                "usage" : {"prompt_tokens": prompt_tokens, "completion_tokens": tokens_generated, "total_tokens": all_tokens}
                ,"choices": [
                        {
                            "message":{
                                    "role":"assistant",
                                    "content": ai_response
                                },
                            "finish_reason":"stop",
                            "index":0
                        }
                    ]
            }

        return JSONResponse(content=response)
    except Exception as e:
        print(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


@app.post('/store_response')
async def store_response(data: DataFineTuneObject):
	try:
		print(data)
		database.create_document({'user_message': data.user_message, 'ai_message':data.ai_message})

		response = {
			"message": "Feedback_stored",
			"feedback": f"user_message: {data.user_message} ai_message:{data.ai_message}"			
}
		return JSONResponse(content=response)
	except Exception as e:
		print(f"An error occurred: {e}")
		raise HTTPException(status_code=500, detail="Internal Server Error")


@app.post('/train')
async def train(data: TrainParamsObject):
       try:
               response = database.fetch_document(data.current_ds)
               data_object = pd.DataFrame(json_util.loads(response))
               data_object["text"] = "SYSTEM: You are a helpful assistant that answers questions of user. Be respectful, dont try to answer things you dont know. Be friendly with the user. \n User: " + data_object["user_message"] + "\n" + "ASSISTANT: " + data_object["ai_message"]
               print(data_object.head())
               finetune(train_df = data_object)
               print("Finetune Complete")
               os.execv(sys.executable, ['python'] + sys.argv)

       except Exception as e:
               print(f"An error occurred: {e}")
               raise HTTPException(status_code=500, detail="Internal Server Error")	

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)