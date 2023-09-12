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
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline
)
import torch
from peft import LoraConfig, AutoPeftModelForCausalLM, PeftModel
from trl import SFTTrainer
from auth import hf_write_token

from huggingface_hub import login

login(hf_write_token)

model_name_or_path = "hcevik/customml-test"
base_model_name = "hcevik/customml-test"
real_model_name = "hcevik/customml-test"

tokenizer = AutoTokenizer.from_pretrained(real_model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"  # Fix for fp16

def createPrompt(template):
    prompt = PromptTemplate(template=template,input_variables=["history","input"])
    return prompt

def createLLM():
    
    use_triton = False

    # Quantization Config
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=False
    )

    # Model
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=quant_config,
        device_map={"": 0},
    )
    
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

