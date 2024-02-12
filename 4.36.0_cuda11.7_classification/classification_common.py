# based on:
# https://huggingface.co/docs/transformers/v4.36.1/en/tasks/sequence_classification

from transformers import pipeline


def load_finetuned_model(model_dir):
    """
    Loads the model and returns the pipeline.

    :param model_dir: the directory with the model
    :type model_dir: str
    :return: the pipeline
    """
    return pipeline("sentiment-analysis", model=model_dir)


def predict(text, pipe):
    """
    Passes the text through the pipeline and returns the result.

    :param text: the prompt to send
    :type text: str
    :param pipe: the prediction pipeline to use
    :return: the generated answer (label and score)
    :rtype: dict
    """
    return pipe(text)
