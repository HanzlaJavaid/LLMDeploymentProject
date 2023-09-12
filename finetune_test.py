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

from auth import hf_read_token, hf_write_token

from huggingface_hub import login
login(token=hf_write_token)

refined_model = "Wizard-Vicuna-7B-Uncensored-HF_REFINED"
remote_repo = "hcevik/customml-test"

# Dataset
data_name = "mlabonne/guanaco-llama2-1k"
training_data = load_dataset(data_name, split="train")

# Model and tokenizer names
base_model_name = "TheBloke/Wizard-Vicuna-7B-Uncensored-HF"
real_model_name = "TheBloke/Wizard-Vicuna-7B-Uncensored-HF"

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

# Training Params
train_params = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    optim="paged_adamw_32bit",
    save_steps=10,
    logging_steps=1,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=False,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=10,
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

# Training
fine_tuning.train()

# Save Model
fine_tuning.model.save_pretrained(refined_model)

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

# Reload tokenizer to save it
tokenizer = AutoTokenizer.from_pretrained(real_model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model_to_save.push_to_hub(remote_repo,token = hf_write_token)
tokenizer.push_to_hub(remote_repo,token = hf_write_token)
