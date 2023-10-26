from transformers import AutoTokenizer, RemBertConfig, RemBertForCausalLM

MODEL = "google/rembert"


def load_model(repo_or_dir):
    """
    Loads the model and tokenizer.

    :param repo_or_dir: the huggingface repo or local dir to load the model from
    :type repo_or_dir: str
    :return: the tuple of tokenizer and model
    :rtype: tuple
    """
    print("Loading tokenizer from : %s" % repo_or_dir)
    tokenizer = AutoTokenizer.from_pretrained(repo_or_dir)
    config = RemBertConfig.from_pretrained(MODEL)
    config.is_decoder = True
    print("Loading model from : %s" % repo_or_dir)
    model = RemBertForCausalLM.from_pretrained(repo_or_dir, config=config)
    return tokenizer, model


def predict(tokenizer, model, prompt, max_new_tokens=100, do_sample=True, top_k=50, top_p=0.95):
    """
    Completes the prompt.

    :param tokenizer: the tokenizer to use
    :param model: the model to use
    :param prompt: the prompt to complete
    :type prompt: str
    :param max_new_tokens: the maximum number of new tokens to generate
    :type max_new_tokens: int
    :param do_sample: whether to perform sampling
    :type do_sample: bool
    :param top_k: top k results to return
    :type top_k: int
    :param top_p: the minimum probability for the top k results
    :type top_p: float
    :return: the list of completed prompts that meet the probability
    :rtype: list
    """
    inputs = tokenizer(prompt, return_tensors="pt").input_ids
    outputs = model.generate(inputs, max_new_tokens=max_new_tokens, do_sample=do_sample, top_k=top_k, top_p=top_p)
    text = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return text
