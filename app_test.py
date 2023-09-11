from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
#from ggml_model_test import generate, AI_INIT
from gptq_model_run import generate, AI_INIT, count_tokens,model_name_or_path
from pydantic import BaseModel
import json
from typing import List
import datetime
import random

chain = AI_INIT(prompt_template="""
SYSTEM: You are a helpful assistant that answers questions of user. Be respectful, dont try to answer things you dont know. Be friendly with the user.

{history}

{input}
ASSISTANT:
""")

def generate_random_id(length):
    number_string = ''.join([str(random.randint(0, 9)) for _ in range(length)])
    return number_string

def conversation_history_format(history):
    formatted_messages = "\n".join([f"User: {msg.user_message}\nAssistant: {msg.assistant_reply}" for msg in history])
    return formatted_messages

def conversation_history_format_new(history):
    formatted_messages = "\n".join([f"{msg.role}: {msg.content}" for msg in history])
    print(formatted_messages)
    return formatted_messages

class Message(BaseModel):
    user_message: str
    assistant_reply: str

class PostData(BaseModel):
    previous_messages: List[Message]
    user_message: str
    get_context: bool

class MessageObject(BaseModel):
    role: str
    content: str

class DataObject(BaseModel):
    model: str
    messages: List[MessageObject]
    temperature: float


app = FastAPI()

@app.post('/generate')
async def reply(data: PostData):
    try:
        history_raw = data.previous_messages
        history = conversation_history_format(history_raw)
        print(history)
        user_message = str(data.user_message)
        print(user_message)
        ai_response = generate(chain=chain, user_input=user_message,conversation_history=history)
        if (data.get_context):
            response = {
                "conversation_history": jsonable_encoder(data.previous_messages),
                "user_message":user_message,
                "assistant_reply":ai_response
            }
        else:
            response = {
                "user_message":user_message,
                "assistant_reply":ai_response
                }

        return JSONResponse(content=response)
    except Exception as e:
        print(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


@app.post('/generate_test')
async def reply_test(data: DataObject):
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


