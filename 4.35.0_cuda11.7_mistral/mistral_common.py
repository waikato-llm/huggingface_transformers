# based on:
# https://gathnex.medium.com/mistral-7b-fine-tuning-a-step-by-step-guide-52122cdbeca8

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)
from peft import PeftModel


def get_4bit_quant_config():
    """
    Returns the 4bit quantization config.

    :return: the configuration
    :rtype: BitsAndBytesConfig
    """
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=False,
    )
    return quant_config


def load_base_model(base_model):
    """
    Loads and returns the base model.

    :param base_model: the name/dir of the base model
    :type base_model: str
    :return: the model
    """
    print("--> loading mistral base model: %s" % base_model)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=get_4bit_quant_config(),
        device_map={"": 0}
    )
    return model


def load_base_tokenizer(base_model):
    """
    Loads and returns the tokenizer of the base model.

    :param base_model: the name/dir of the base model
    :type base_model: str
    :return: the tokenizer
    """
    print("--> loading tokenizer: %s" % base_model)
    tokenizer = AutoTokenizer.from_pretrained(base_model)  # was missing in tutorial
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_eos_token = True
    return tokenizer


def load_finetuned_model(base_model, model_dir):
    """
    Loads the model/tokenizer and returns it.

    :param base_model: the name/dir of the base model
    :type base_model: str
    :param model_dir: the directory with the PEFT model
    :type model_dir: str
    :return: the tuple of the model and tokenizer
    :rtype: tuple
    """
    model = load_base_model(base_model)

    print("--> loading peft model: %s" % model_dir)
    peft_model = PeftModel.from_pretrained(model, model_dir)

    # loading tokenizer
    tokenizer = load_base_tokenizer(base_model)

    return peft_model, tokenizer


def build_pipeline(model, tokenizer, max_length=200):
    """
    Builds a pipeline from the model and tokenizer.

    :param model: the model to use
    :param tokenizer: the tokenizer to use
    :param max_length: the maximum length of the response
    :type max_length: int
    :return: the pipeline
    """
    return pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=max_length)


def predict(prompt, pipe, raw=False):
    """
    Presents the prompt to the model and returns the generated answer.

    :param prompt: the prompt to send
    :type prompt: str
    :param pipe: the prediction pipeline to use
    :param raw: whether to return the raw response
    :type raw: bool
    :return: the generated answer
    """
    result = pipe(f"<s>[INST] {prompt} [/INST]")
    result = result[0]['generated_text']
    if not raw:
        if result is not None:
            if "[/INST]" in result:
                result = result.replace("[/INST]", "\t")
                result = result[result.index("\t"):].strip()
    return result
