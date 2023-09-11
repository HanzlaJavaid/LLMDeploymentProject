from langchain.llms import CTransformers
from langchain.llms import LlamaCpp
from langchain import PromptTemplate, LLMChain, ConversationChain
from langchain.callbacks.manager import CallbackManager
from langchain.memory import ConversationBufferMemory,ConversationTokenBufferMemory,ConversationBufferWindowMemory
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from huggingface_hub import hf_hub_download
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.schema.messages import HumanMessage
from langchain.schema.messages import AIMessage
from langchain import HuggingFacePipeline
from huggingface_hub import hf_hub_download

model_name_or_path = "TheBloke/Wizard-Vicuna-13B-Uncensored-SuperHOT-8K-GGML"

filename = "wizard-vicuna-13b-uncensored-superhot-8k.ggmlv3.q2_K.bin"

def get_modelpath():
    model_path = hf_hub_download(repo_id=model_name_or_path, filename=filename)
    return model_path

def createPrompt(template):
    prompt = PromptTemplate(template=template,input_variables=["history","input"])
    return prompt

def createLLM(model_path):
    llm = CTransformers(
        model=model_path
        ,model_type='llama'
        ,max_new_tokens = 1028
        ,top_p = 0.9
        ,temprature = 0.9
        ,gpu_layers = 128
    )
    return llm

def createChain(llm,prompt):
    chain = LLMChain(llm=llm,verbose=True,prompt=prompt)
    return chain

def generate(conversation_history,user_input,chain):
    response = chain.predict(history=conversation_history,input=user_input)
    return response

def AI_INIT(prompt_template):
    model_path=get_modelpath()
    print(f"Model Synchronized \n Model Name: {model_name_or_path}")
    prompt = createPrompt(template=prompt_template)
    print(f"Prompt Template Synchronized \n {prompt_template}")
    llm = createLLM(model_path)
    print("LLM Created")
    chain = createChain(llm,prompt)
    print("Chain Created")
    return chain

