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


def get_4bit_quant_config():
    """
    Returns the 4bit quantization config.

    :return: the configuration
    :rtype: BitsAndBytesConfig
    """
    compute_dtype = getattr(torch, "float16")
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        load_in_8bit=False,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=False,
    )
    return quant_config


def get_8bit_quant_config():
    """
    Returns the 8bit quantization config.

    :return: the configuration
    :rtype: BitsAndBytesConfig
    """
    compute_dtype = getattr(torch, "float16")
    quant_config = BitsAndBytesConfig(
        load_in_4bit=False,
        load_in_8bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=False,
    )
    return quant_config


def load_base_model(base_model, quantization="4bit"):
    """
    Loads and returns the base model.

    :param base_model: the name/dir of the base model
    :type base_model: str
    :param quantization: the quantization to use (4bit/8bit)
    :type quantization: str
    :return: the model
    """
    print("--> loading llama 2 base model: %s" % base_model)
    if quantization == "4bit":
        config = get_4bit_quant_config()
    elif quantization == "8bit":
        config = get_8bit_quant_config()
    else:
        raise Exception("Unsupported quantization: %s" % quantization)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=config,
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
    tokenizer.padding_side = "right"
    return tokenizer


def load_finetuned_model(base_model, model_dir, quantization="4bit"):
    """
    Loads the model/tokenizer and returns it.

    :param base_model: the name/dir of the base model
    :type base_model: str
    :param model_dir: the directory with the PEFT model
    :type model_dir: str
    :param quantization: the quantization to use (4bit/8bit)
    :type quantization: str
    :return: the tuple of the model and tokenizer
    :rtype: tuple
    """
    model = load_base_model(base_model, quantization=quantization)

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
