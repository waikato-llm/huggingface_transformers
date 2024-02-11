# based on:
# https://www.datacamp.com/tutorial/fine-tuning-llama-2
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)
from peft import PeftModel

# Fine-tuned model
print("--> loading llama 2 model")
base_model = "NousResearch/Llama-2-7b-chat-hf"
compute_dtype = getattr(torch, "float16")
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=False,
)
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=quant_config,
    device_map={"": 0}
)
new_model = PeftModel.from_pretrained(model, "./llama-2-7b-chat-guanaco")

# loading tokenizer
print("--> loading tokenizer")
tokenizer = AutoTokenizer.from_pretrained(base_model)  # was missing in tutorial
tokenizer.padding_side = "right"

# predict
pipe = pipeline(task="text-generation", model=new_model, tokenizer=tokenizer, max_length=200)
while True:
    prompt = input("Please enter the text to complete by Llama2: ")
    result = pipe(f"<s>[INST] {prompt} [/INST]")
    print(result[0]['generated_text'])

