# taken from here:
# https://www.datacamp.com/tutorial/fine-tuning-llama-2

# Model from Hugging Face hub
base_model = "NousResearch/Llama-2-7b-chat-hf"

# 1. getting started
print("--> getting started")

import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig
from trl import SFTTrainer

# New instruction dataset
guanaco_dataset = "mlabonne/guanaco-llama2-1k"

# Fine-tuned model
new_model = "llama-2-7b-chat-guanaco"

# 3. load dataset
print("--> load dataset")
dataset = load_dataset(guanaco_dataset, split="train")

# 4. 4-bit quantization
print("--> 4-bit quantization")
compute_dtype = getattr(torch, "float16")

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=False,
)

# 5. loading llama 2 model
print("--> loading llama 2 model")
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=quant_config,
    device_map={"": 0}
)
model.config.use_cache = False
model.config.pretraining_tp = 1

# 6. loading tokenizer
print("--> loading tokenizer")
tokenizer = AutoTokenizer.from_pretrained(base_model)  # was missing in tutorial
tokenizer.padding_side = "right"

# 7. peft parameters
print("--> peft params")
peft_params = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
)

# 8. training parameters
print("--> training params")
training_params = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    optim="paged_adamw_32bit",
    save_steps=25,
    logging_steps=25,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=False,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant",
    #report_to="tensorboard"   # don't have tensorboard installed
)

# 9. model finetuning
print("--> model finetuning")
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_params,
    dataset_text_field="text",
    max_seq_length=None,
    tokenizer=tokenizer,
    args=training_params,
    packing=False,
)
trainer.train()
trainer.model.save_pretrained(new_model)
trainer.tokenizer.save_pretrained(new_model)

# 10. evaluation
print("--> evaluation")
# no tensorboard installed
#from tensorboard import notebook
#log_dir = "results/runs"
#notebook.start("--logdir {} --port 4000".format(log_dir))
#logging.set_verbosity(logging.CRITICAL)

prompt = "Who is Leonardo Da Vinci?"
pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=200)
result = pipe(f"<s>[INST] {prompt} [/INST]")
print(result[0]['generated_text'])

prompt = "What is Datacamp Career track?"
result = pipe(f"<s>[INST] {prompt} [/INST]")
print(result[0]['generated_text'])

