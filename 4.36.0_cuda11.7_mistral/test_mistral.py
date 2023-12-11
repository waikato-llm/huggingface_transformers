print("--> importing modules")
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,HfArgumentParser,TrainingArguments,pipeline, logging, TextStreamer
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
import os,torch, platform, warnings
from datasets import load_dataset
from trl import SFTTrainer
from huggingface_hub import notebook_login

base_model, dataset_name, new_model = "mistralai/Mistral-7B-v0.1" , "gathnex/Gath_baize", "gathnex/Gath_mistral_7b"

print("--> load dataset")
dataset = load_dataset(dataset_name, split="train")
dataset["chat_sample"][0]

print("--> init model")
bnb_config = BitsAndBytesConfig(
    load_in_4bit= True,
    bnb_4bit_quant_type= "nf4",
    bnb_4bit_compute_dtype= torch.bfloat16,
    bnb_4bit_use_double_quant= False,
)
model = AutoModelForCausalLM.from_pretrained(
   base_model,
    quantization_config=bnb_config,
    device_map={"": 0}
)
model.config.use_cache = False # silence the warnings. Please re-enable for inference!
model.config.pretraining_tp = 1
model.gradient_checkpointing_enable()
# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
tokenizer.padding_side = 'right'
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_eos_token = True
tokenizer.add_bos_token, tokenizer.add_eos_token

model = prepare_model_for_kbit_training(model)
peft_config = LoraConfig(
        r=16,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj","gate_proj"]
    )
model = get_peft_model(model, peft_config)

print("--> train")
#Hyperparamter
training_arguments = TrainingArguments(
    output_dir= "./results",
    num_train_epochs= 2,
    per_device_train_batch_size= 8,
    gradient_accumulation_steps= 2,
    optim = "paged_adamw_8bit",
    save_steps= 1000,
    logging_steps= 30,
    learning_rate= 2e-4,
    weight_decay= 0.001,
    fp16= False,
    bf16= False,
    max_grad_norm= 0.3,
    max_steps= -1,
    warmup_ratio= 0.3,
    group_by_length= True,
    lr_scheduler_type= "constant",
    report_to="tensorboard"
)
# Setting sft parameters
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    max_seq_length= None,
    dataset_text_field="chat_sample",
    tokenizer=tokenizer,
    args=training_arguments,
    packing= False,
)
trainer.train()

print("--> save model")
# Save the fine-tuned model
trainer.model.save_pretrained(new_model)
model.config.use_cache = True
model.eval()

print("--> stream")
def stream(user_prompt):
    runtimeFlag = "cuda:0"
    system_prompt = 'The conversation between Human and AI assisatance named Gathnex\n'
    B_INST, E_INST = "[INST]", "[/INST]"
    prompt = f"{system_prompt}{B_INST}{user_prompt.strip()}\n{E_INST}"
    inputs = tokenizer([prompt], return_tensors="pt").to(runtimeFlag)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    _ = model.generate(**inputs, streamer=streamer, max_new_tokens=200)

stream("what is newtons 2rd law and its formula")

print("--> clear/use again")
# Clear the memory footprint
del model, trainer
torch.cuda.empty_cache()

# Reload the base model
base_model_reload = AutoModelForCausalLM.from_pretrained(
    base_model, low_cpu_mem_usage=True,
    return_dict=True,torch_dtype=torch.bfloat16,
    device_map= {"": 0})
model = PeftModel.from_pretrained(base_model_reload, new_model)
model = model.merge_and_unload()

# Reload tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

stream("what is newtons 2rd law and its formula")

