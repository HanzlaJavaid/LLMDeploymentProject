device_map={"": 0}
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline
)
from peft import LoraConfig, AutoPeftModelForCausalLM, PeftModel
from trl import SFTTrainer
import pandas as pd
from datasets import Dataset,DatasetDict
from auth import hf_read_token, hf_write_token

from huggingface_hub import login

refined_model = "Wizard-Vicuna-7B-Uncensored-HF_REFINED"
remote_repo = "hcevik/customml-test"

base_model_name = "hcevik/customml-test"
real_model_name = "hcevik/customml-test"

def finetune(train_df,finetune_epochs = 10,batch_size=4):

    login(token=hf_write_token)

    # Dataset
    train_dataset_dict = DatasetDict({
        "train": Dataset.from_pandas(train_df),
    })
    training_data=train_dataset_dict['train']
    
    print("Finetune data initialized")

    
    # Tokenizer
    llama_tokenizer = AutoTokenizer.from_pretrained(real_model_name, trust_remote_code=True)
    llama_tokenizer.pad_token = llama_tokenizer.eos_token
    llama_tokenizer.padding_side = "right"  # Fix for fp16

    # Quantization Config
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=False
    )

    # Model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=quant_config,
        device_map={"": 0},
    )
    base_model.config.use_cache = False
    base_model.config.pretraining_tp = 1

    print("Model initialized")


    lora_alpha = 16
    lora_dropout = 0.1
    lora_r = 8

    # LoRA Config
    peft_parameters = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=8,
        bias="none",
        task_type="CAUSAL_LM"
    )
    save_steps = finetune_epochs + 1
    # Training Params
    train_params = TrainingArguments(
        output_dir="./results",
        num_train_epochs=1,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=batch_size,
        optim="paged_adamw_32bit",
        save_steps=save_steps,
        logging_steps=1,
        learning_rate=2e-4,
        weight_decay=0.001,
        fp16=False,
        bf16=False,
        max_grad_norm=0.3,
        max_steps=finetune_epochs,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="constant",
        report_to="tensorboard"
    )


    # Trainer
    fine_tuning = SFTTrainer(
        model=base_model,
        train_dataset=training_data,
        peft_config=peft_parameters,
        dataset_text_field="text",
        tokenizer=llama_tokenizer,
        args=train_params
    )

    print("Trainer Initialized")


    # Training
    fine_tuning.train()

    # Save Model
    fine_tuning.model.save_pretrained(refined_model)

    print("Training completed")


    # Reload model in FP16 and merge it with LoRA weights
    base_model_FP16 = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        low_cpu_mem_usage=True,
        return_dict=True,
        torch_dtype=torch.float16,
        device_map=device_map,
    )

    model_to_save = PeftModel.from_pretrained(base_model_FP16, refined_model)
    model_to_save = model_to_save.merge_and_unload()

    print("Merging models")

    # Reload tokenizer to save it
    tokenizer = AutoTokenizer.from_pretrained(real_model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model_to_save.push_to_hub(remote_repo,token = hf_write_token)
    tokenizer.push_to_hub(remote_repo,token = hf_write_token)

    print("Finetuned model saved in HF.")

    return True