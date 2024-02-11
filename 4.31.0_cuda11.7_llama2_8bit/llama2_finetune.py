# based on:
# https://www.datacamp.com/tutorial/fine-tuning-llama-2
import argparse
import traceback
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
from llama2_common import load_base_model, load_base_tokenizer


def finetune(train_data, output_dir, base_model="NousResearch/Llama-2-7b-chat-hf", lora_alpha=16, lora_dropout=0.1, lora_r=64,
             num_train_epochs=1.0, save_steps=100, logging_steps=100, learning_rate=2e-4, weight_decay=0.001,
             max_grad_norm=0.3, warmup_ratio=0.3, per_device_train_batch_size=4, gradient_accumulation_steps=1,
             quantization="4bit", dataset_text_field="text", max_checkpoints=5):
    """
    Fine-tunes a Llama2 model.

    :param train_data: the training data in jsonlines format
    :type train_data: str
    :param output_dir: the directory to store the finetuned model in (and checkpoints)
    :type output_dir: str
    :param base_model: the name/dir of the base model to use
    :type base_model: str
    :param lora_alpha: the lora alpha to use
    :type lora_alpha: float
    :param lora_dropout: the lora dropout value
    :type lora_dropout: float
    :param lora_r: the lora attention dimension
    :type lora_r: int
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
    :param max_grad_norm: maximum gradient norm (for gradient clipping).
    :type max_grad_norm: float
    :param warmup_ratio: Ratio of total training steps used for a linear warmup from 0 to learning_rate.
    :type warmup_ratio: float
    :param per_device_train_batch_size: the batch size to use
    :type per_device_train_batch_size: int
    :param gradient_accumulation_steps: to increase batch size without using more memory
    :type gradient_accumulation_steps: int
    :param dataset_text_field: the field in the jsonlines data with the text to learn from
    :type dataset_text_field: str
    :param max_checkpoints: the maximum number of checkpoints to keep
    :type max_checkpoints: int
    """
    print("--> load dataset")
    dataset = load_dataset("json", data_files=train_data, split="train")

    # loading llama 2 model
    model = load_base_model(base_model, quantization=quantization)
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    # loading tokenizer
    tokenizer = load_base_tokenizer(base_model)

    # peft parameters
    peft_params = LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        r=lora_r,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # training parameters
    training_params = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        optim="paged_adamw_32bit",
        save_steps=save_steps,
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        fp16=False,
        bf16=False,
        max_grad_norm=max_grad_norm,
        max_steps=-1,
        warmup_ratio=warmup_ratio,
        group_by_length=True,
        lr_scheduler_type="constant",
        report_to="tensorboard",
        push_to_hub=False,
        save_total_limit=max_checkpoints,
    )

    # model finetuning
    print("--> training")
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_params,
        dataset_text_field=dataset_text_field,
        max_seq_length=None,
        tokenizer=tokenizer,
        args=training_params,
        packing=False,
    )
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
        description="Llama2 - Fine-tune a model",
        prog="llama2_finetune",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--train_data', metavar="PATH", type=str, required=True, help='The training data to use (jsonlines format).')
    parser.add_argument('--dataset_text_field', type=str, metavar="FIELD", default="text", required=False, help='The name of the field containing the training text.')
    parser.add_argument('--base_model', metavar="NAME_OR_PATH", type=str, default="NousResearch/Llama-2-7b-chat-hf", required=False, help='The name/dir base model to use')
    parser.add_argument('--lora_alpha', metavar="INT", type=int, default=16, required=False, help='The lora alpha')
    parser.add_argument('--lora_dropout', metavar="FLOAT", type=float, default=0.1, required=False, help='The lora dropout')
    parser.add_argument('--lora_r', type=int, metavar="INT", default=64, required=False, help='The lora attention dimension.')
    parser.add_argument('--num_train_epochs', type=float, metavar="FLOAT", default=1.0, required=False, help='The number of training epochs to perform.')
    parser.add_argument('--save_steps', type=int, metavar="INT", default=100, required=False, help='Number of updates steps before two checkpoint saves.')
    parser.add_argument('--logging_steps', type=int, metavar="INT", default=100, required=False, help='Number of update steps between two logs.')
    parser.add_argument('--learning_rate', type=float, metavar="FLOAT", default=2e-4, required=False, help='The learning rate to use.')
    parser.add_argument('--weight_decay', type=float, metavar="FLOAT", default=0.001, required=False, help='The weight decay to apply.')
    parser.add_argument('--max_grad_norm', type=float, metavar="FLOAT", default=0.3, required=False, help='Maximum gradient norm (for gradient clipping).')
    parser.add_argument('--warmup_ratio', type=float, metavar="FLOAT", default=0.3, required=False, help='Ratio of total training steps used for a linear warmup from 0 to learning_rate.')
    parser.add_argument('--max_checkpoints', type=int, metavar="INT", default=5, help='The maximum number of checkpoints to keep.')
    parser.add_argument('--per_device_train_batch_size', type=int, metavar="INT", default=4, help='The batch size per device for training.')
    parser.add_argument('--gradient_accumulation_steps', type=int, metavar="INT", default=1, help='Number of updates steps to accumulate the gradients for, before performing a backward/update pass.')
    parser.add_argument('--quantization', choices=["4bit", "8bit"], default="4bit", help='The type of quantization to use.')
    parser.add_argument('--output_dir', required=False, metavar="DIR", type=str, default="./output", help='The directory to store the checkpoints and model in.')
    parsed = parser.parse_args(args=args)
    finetune(parsed.train_data, parsed.output_dir, base_model=parsed.base_model,
             lora_alpha=parsed.lora_alpha, lora_dropout=parsed.lora_dropout, lora_r=parsed.lora_r,
             num_train_epochs=parsed.num_train_epochs, save_steps=parsed.save_steps, logging_steps=parsed.logging_steps,
             learning_rate=parsed.learning_rate, weight_decay=parsed.weight_decay, max_grad_norm=parsed.max_grad_norm,
             warmup_ratio=parsed.warmup_ratio, per_device_train_batch_size=parsed.per_device_train_batch_size,
             gradient_accumulation_steps=parsed.gradient_accumulation_steps, dataset_text_field=parsed.dataset_text_field,
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
