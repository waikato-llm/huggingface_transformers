# based on:
# https://huggingface.co/docs/transformers/v4.36.1/en/tasks/sequence_classification
import argparse
import json
import traceback
import evaluate
import numpy as np

from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer


def finetune(train_data, label_map, output_dir, test_data=None, test_size=0.2, base_model="distilbert-base-uncased",
             num_train_epochs=2.0, save_steps=100, logging_steps=100, learning_rate=2e-5, weight_decay=0.01,
             text_field="text", batch_size=16, max_checkpoints=5):
    """
    Fine-tunes a text classification model.

    :param train_data: the training data in jsonlines format
    :type train_data: str
    :param label_map: the mapping between label string (key) and label int index (value)
    :type label_map: dict
    :param output_dir: the directory to store the finetuned model in (and checkpoints)
    :type output_dir: str
    :param test_data: the (optional) test data in jsonlines format
    :type test_data: str
    :param test_size: the size of the data to split off the training data if no explicit test data provided (0-1)
    :type test_size: float
    :param base_model: the name/dir of the base model to use
    :type base_model: str
    :param num_train_epochs: the number of training epochs to perform
    :type num_train_epochs: float
    :param save_steps: number of update steps before saving
    :type save_steps: int
    :param logging_steps: number of update steps before logging
    :type logging_steps: int
    :param learning_rate: the learning rate to use
    :type learning_rate: float
    :param weight_decay: the weight decay to use
    :type weight_decay: float
    :param text_field: the field in the jsonlines data with the text to learn from
    :type text_field: str
    :param batch_size: the size of the batch for training/evaluation
    :type batch_size: int
    :param max_checkpoints: the maximum number of checkpoints to keep
    :type max_checkpoints: int
    """
    accuracy = evaluate.load("accuracy")

    def preprocess_function(examples):
        return tokenizer(examples[text_field], truncation=True)

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return accuracy.compute(predictions=predictions, references=labels)

    # create reverse lookup for labels
    reverse_label_map = dict()
    for k, v in label_map.items():
        reverse_label_map[v] = k

    print("--> load dataset")
    if test_data is not None:
        dataset = load_dataset("json", data_files={"train": train_data, "test": test_data})
    else:
        dataset = load_dataset("json", data_files={"train": train_data})
        dataset = dataset["train"].train_test_split(test_size=test_size)

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    # preprocess
    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # load model
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model, num_labels=len(label_map), id2label=reverse_label_map, label2id=label_map
    )

    # training parameters
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_train_epochs,
        weight_decay=weight_decay,
        save_steps=save_steps,
        logging_steps=logging_steps,
        save_total_limit=max_checkpoints,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        push_to_hub=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # model finetuning
    print("--> training")
    trainer.train()
    trainer.model.save_pretrained(output_dir)
    trainer.tokenizer.save_pretrained(output_dir)


def main(args=None):
    """
    The main method for parsing command-line arguments and starting the training.

    :param args: the commandline arguments, uses sys.argv if not supplied
    :type args: list
    """

    parser = argparse.ArgumentParser(
        description="Text classification - Fine-tune a model",
        prog="classification_finetune",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--train_data', metavar="PATH", type=str, required=True, help='The training data to use (jsonlines format).')
    parser.add_argument('--test_data', metavar="PATH", type=str, required=False, default=None, help='The test data to use (jsonlines format).')
    parser.add_argument('--test_size', metavar="SIZE", type=float, required=False, default=0.2, help='The size of the data to split off the training set for the test set (0-1) if no explicit test set provided.')
    parser.add_argument('--text_field', type=str, metavar="FIELD", default="text", required=False, help='The name of the field containing the training text.')
    parser.add_argument('--label_map', type=str, metavar="PATH", required=True, help='The JSON file containing the mapping between label string (key) and label index (value).')
    parser.add_argument('--base_model', metavar="NAME_OR_PATH", type=str, default="distilbert-base-uncased", required=False, help='The name/dir base model to use')
    parser.add_argument('--num_train_epochs', type=float, metavar="FLOAT", default=2.0, required=False, help='The number of training epochs to perform.')
    parser.add_argument('--save_steps', type=int, metavar="INT", default=100, required=False, help='Number of updates steps before two checkpoint saves.')
    parser.add_argument('--logging_steps', type=int, metavar="INT", default=100, required=False, help='Number of update steps between two logs.')
    parser.add_argument('--learning_rate', type=float, metavar="FLOAT", default=2e-5, required=False, help='The learning rate to use.')
    parser.add_argument('--weight_decay', type=float, metavar="FLOAT", default=0.01, required=False, help='The weight decay to apply.')
    parser.add_argument('--batch_size', type=int, metavar="SIZE", default=16, required=False, help='The batch size to use for training and evaluation.')
    parser.add_argument('--max_checkpoints', type=int, metavar="INT", default=5, help='The maximum number of checkpoints to keep.')
    parser.add_argument('--output_dir', required=False, metavar="DIR", type=str, default="./output", help='The directory to store the checkpoints and model in.')
    parsed = parser.parse_args(args=args)

    with open(parsed.label_map, "r") as fp:
        label_map = json.load(fp)

    finetune(parsed.train_data, label_map, parsed.output_dir, base_model=parsed.base_model,
             text_field=parsed.text_field, test_data=parsed.test_data, test_size=parsed.test_size,
             num_train_epochs=parsed.num_train_epochs, save_steps=parsed.save_steps,
             logging_steps=parsed.logging_steps, learning_rate=parsed.learning_rate,
             weight_decay=parsed.weight_decay, batch_size=parsed.batch_size,
             max_checkpoints=parsed.max_checkpoints)


def sys_main() -> int:
    """
    Runs the main function using the system cli arguments, and
    returns a system error code.
    :return:    0 for success, 1 for failure.
    """
    try:
        main()
        return 0
    except KeyboardInterrupt:
        return 0
    except Exception:
        print(traceback.format_exc())
        return 1


if __name__ == '__main__':
    main()
