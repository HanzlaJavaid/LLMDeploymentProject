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
from transformers import AutoTokenizer, pipeline, logging
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
import argparse

model_name_or_path = "TheBloke/Wizard-Vicuna-7B-Uncensored-SuperHOT-8K-GPTQ"
model_basename = "wizard-vicuna-7b-uncensored-superhot-8k-GPTQ-4bit-128g.no-act.order"

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

def createPrompt(template):
    prompt = PromptTemplate(template=template,input_variables=["history","input"])
    return prompt

def createLLM():
    
    use_triton = False

    model = AutoGPTQForCausalLM.from_quantized(model_name_or_path,
        use_safetensors=True,
        trust_remote_code=True,
        device_map='auto',
        use_triton=use_triton,
        quantize_config=None)
    
    pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=1024,
    temperature=0.9,
    top_p=0.95,
    repetition_penalty=1.15
    )
    model.seqlen = 2000
    
    llm = HuggingFacePipeline(pipeline = pipe)

    return llm

def createChain(llm,prompt):
    chain = LLMChain(llm=llm,verbose=True,prompt=prompt)
    return chain

def generate(conversation_history,user_input,chain):
    response = chain.predict(history=conversation_history,input=user_input)
    return response

def count_tokens(text):
    tokens = tokenizer.tokenize(text)
    return len(tokens)

def AI_INIT(prompt_template):
    print(f"Model Synchronized \n Model Name: {model_name_or_path}")
    prompt = createPrompt(template=prompt_template)
    print(f"Prompt Template Synchronized \n {prompt_template}")
    llm = createLLM()
    print("LLM Created")
    chain = createChain(llm,prompt)
    print("Chain Created")
    return chain

