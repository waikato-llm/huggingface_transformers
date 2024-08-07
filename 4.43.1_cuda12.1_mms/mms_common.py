# based on:
# https://huggingface.co/docs/transformers/main/en/model_doc/mms#mms

import scipy
import torch
from datasets import load_dataset, Audio, Dataset
from transformers import set_seed
from transformers import VitsTokenizer, VitsModel
from transformers import Wav2Vec2ForCTC, AutoProcessor
from transformers import Wav2Vec2ForSequenceClassification, AutoFeatureExtractor


def load_asr(model_id, target_lang):
    """
    Loads the ASR processor and model.

    :param model_id: the model ID, e.g., facebook/mms-1b-all
    :type model_id: str
    :param target_lang: the language to transcribe, e.g., eng or fra
    :type target_lang: str
    :return: the tuple of processor and model
    :rtype: tuple
    """
    print("Loading ASR: %s/%s" % (model_id, target_lang))
    processor = AutoProcessor.from_pretrained(model_id, target_lang=target_lang)
    model = Wav2Vec2ForCTC.from_pretrained(model_id, target_lang=target_lang, ignore_mismatched_sizes=True)
    return processor, model


def infer_asr(processor, model, sample):
    """
    Transcribes the audio sample using the supplied processor and model.

    :param processor: the processor to use
    :param model: the model to use
    :param sample: the audio sample to transcribe
    :return: the transcribed text
    :rtype: str
    """
    inputs = processor(sample, sampling_rate=16_000, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs).logits

    ids = torch.argmax(outputs, dim=-1)[0]
    transcription = processor.decode(ids)
    return transcription


def load_tts(model_id):
    """
    Loads the TTS tokenizer and model.

    :param model_id: the ID of the model to use, e.g., facebook/mms-tts-eng
    :type model_id: str
    :return: the tuple of tokenizer and model
    :rtype: tuple
    """
    print("Loading TTS: %s" % model_id)
    tokenizer = VitsTokenizer.from_pretrained(model_id)
    model = VitsModel.from_pretrained(model_id)
    return tokenizer, model


def infer_tts(tokenizer, model, text):
    """
    Generates audio from the text.

    :param tokenizer: the tokenizer to apply to the text
    :param model: the model to use for speech synthesis
    :param text: the text to turn into audio
    :type text: str
    :return: the generated audio
    """
    inputs = tokenizer(text=text, return_tensors="pt")

    set_seed(555)  # make deterministic

    with torch.no_grad():
        outputs = model(**inputs)

    waveform = outputs.waveform[0].numpy()
    return waveform


def load_lid(model_id):
    """
    Loads the language ID (LID) processor and model.

    :param model_id: the model ID to use, e.g., facebook/mms-lid-126
    :type model_id: str
    :return: the tuple of processor and model
    :rtype: tuple
    """
    processor = AutoFeatureExtractor.from_pretrained(model_id)
    model = Wav2Vec2ForSequenceClassification.from_pretrained(model_id)
    return processor, model


def infer_lid(processor, model, sample):
    """
    Detects the language of the sounds sample.

    :param processor: the LID processor to use
    :param model: the LID model to use
    :param sample: the sample to process
    :return: the detected language
    :rtype: str
    """
    inputs = processor(sample, sampling_rate=16000, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs).logits

    lang_id = torch.argmax(outputs, dim=-1)[0].item()
    detected_lang = model.config.id2label[lang_id]
    return detected_lang


def load_audio(audio_file, sampling_rate=16000):
    """
    Loads the audio file.

    :param audio_file: the file to load
    :type audio_file: str
    :param sampling_rate: the sample rate to use
    :type sampling_rate: int
    :return: the loaded sample
    """
    audio_dataset = Dataset.from_dict({
        "audio": [audio_file]
    }).cast_column("audio", Audio(sampling_rate=sampling_rate))
    result = audio_dataset[0]["audio"]["array"]
    return result


def save_audio(sample, audio_file, sampling_rate=16000):
    """
    Saves the sample to the specified audio file.

    :param sample: the audio sample to save
    :param audio_file: the file to save to
    :type audio_file: str
    :param sampling_rate: the sampling rate to use
    :type sampling_rate: int
    """
    scipy.io.wavfile.write(audio_file, rate=sampling_rate, data=sample)
