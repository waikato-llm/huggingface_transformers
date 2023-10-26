# Based on example here:
# https://huggingface.co/docs/transformers/tasks/language_modeling
# https://huggingface.co/docs/datasets/v1.2.1/loading_datasets.html#from-local-files

import argparse
import math
import os
import traceback

from transformers import AutoTokenizer, RemBertConfig, DataCollatorForLanguageModeling, RemBertForCausalLM, TrainingArguments, Trainer
from datasets import load_dataset
from rembert_common import MODEL


def determine_files(files, file_lists):
    """
    Determines the actual files, supplied as explicit files or from text files listing them.

    :param files: the explicit list of files, ignored if None
    :type files: list
    :param file_lists: the list of text files containing the actual files to use, ignored if None
    :type file_lists: list
    :return: the actual list of files
    :rtype: list
    """
    result = []

    if files is not None:
        for f in files:
            if os.path.exists(f):
                result.append(f)

    if file_lists is not None:
        for l in file_lists:
            with open(l, "r") as fp:
                files = [x.strip() for x in fp.readlines()]
            for f in files:
                if os.path.exists(f):
                    result.append(f)

    return result


def finetune(train_files, output_dir, resume_from=None, test_size=0.2,
             block_size=128, num_train_epochs=3.0, learning_rate=2e-5,
             weight_decay=0.01, max_checkpoints=5):
    """
    Fine-tunes a RemBERT model.

    :param train_files: the text files to train with
    :type train_files: list
    :param output_dir: the directory to store the fine-tuned model and tokenizer in
    :type output_dir: str
    :param resume_from: the checkpoint to resume from, can be None
    :type resume_from: str
    :param test_size: the size of the test set (0-1), when not supplying explicit test files
    :type test_size: float
    :param block_size: the block size to use for training
    :type block_size: int
    :param num_train_epochs: the number of epochs to train the model for
    :type num_train_epochs: float
    :param learning_rate: the learning rate to use
    :type learning_rate: float
    :param weight_decay: the weight decay to apply
    :type weight_decay: float
    :param max_checkpoints: the maximum number of checkpoints to keep
    :type max_checkpoints: int
    """
    print("--> load dataset")
    data = load_dataset('text', data_files=train_files, split="train")
    data = data.train_test_split(test_size=test_size)

    print("--> preprocess")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    config = RemBertConfig.from_pretrained(MODEL)
    config.is_decoder = True

    def preprocess_function(examples):
        return tokenizer([" ".join(x) for x in examples["text"]])

    tokenized_data = data.map(
        preprocess_function,
        batched=True,
        num_proc=4,
        remove_columns=data["train"].column_names,
    )

    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of block_size.
        result = {
            k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    lm_dataset = tokenized_data.map(group_texts, batched=True, num_proc=4)
    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    print("--> train")
    model = RemBertForCausalLM.from_pretrained(MODEL, config=config)
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        push_to_hub=False,
        save_total_limit=max_checkpoints,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_dataset["train"],
        eval_dataset=lm_dataset["test"],
        data_collator=data_collator,
    )

    trainer.train(resume_from_checkpoint=resume_from)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Evaluate
    eval_results = trainer.evaluate()
    print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")


def main(args=None):
    """
    The main method for parsing command-line arguments and starting the training.

    :param args: the commandline arguments, uses sys.argv if not supplied
    :type args: list
    """

    parser = argparse.ArgumentParser(
        description="RemBERT - Fine-tune a RemBERT model",
        prog="rembert_finetune",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--train', required=False, metavar="FILE", help='The text files to use for training', nargs="*")
    parser.add_argument('--train_list', required=False, metavar="FILE", help='The files listing the text files to use for training', nargs="+")
    parser.add_argument('--test_size', type=float, metavar="FLOAT", default=0.2, required=False, help='The size of the test set (0-1).')
    parser.add_argument('--resume_from', required=False, metavar="DIR", help='The text files to use for training', nargs="*")
    parser.add_argument('--block_size', type=int, metavar="INT", default=128, required=False, help='The block size to use for training.')
    parser.add_argument('--num_train_epochs', type=float, metavar="FLOAT", default=3.0, required=False, help='The number of training epochs to perform.')
    parser.add_argument('--learning_rate', type=float, metavar="FLOAT", default=2e-5, required=False, help='The learning rate to use.')
    parser.add_argument('--weight_decay', type=float, metavar="FLOAT", default=0.01, required=False, help='The weight decay to apply.')
    parser.add_argument('--max_checkpoints', type=int, metavar="INT", default=5, help='The maximum number of checkpoints to keep.')
    parser.add_argument('--output_dir', required=False, metavar="DIR", type=str, default="./output", help='The directory to store the model and tokenizer in.')
    parsed = parser.parse_args(args=args)

    train_files = determine_files(parsed.train, parsed.train_list)
    print("--> # train files: %d" % len(train_files))

    finetune(train_files, parsed.output_dir, resume_from=parsed.resume_from, block_size=parsed.block_size,
             test_size=parsed.test_size, num_train_epochs=parsed.num_train_epochs, learning_rate=parsed.learning_rate,
             weight_decay=parsed.weight_decay, max_checkpoints=parsed.max_checkpoints)


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
