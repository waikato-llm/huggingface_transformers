# based on:
# https://www.datacamp.com/tutorial/fine-tuning-llama-2
import argparse
import traceback
from llama2_common import load_finetuned_model, build_pipeline, predict


def interact(model_dir, base_model="NousResearch/Llama-2-7b-chat-hf", max_length=200):
    """
    For interacting with the llama2 model in the console.

    :param model_dir: the directory with the finetuned model
    :type model_dir: str
    :param base_model: the base model that was used for finetuning the model
    :type base_model: str
    :param max_length: the maximum length for the responses
    :type max_length: int
    """
    model, tokenizer = load_finetuned_model(base_model, model_dir)

    # predict
    pipe = build_pipeline(model, tokenizer, max_length=max_length)
    while True:
        prompt = input("\nPlease enter the text to complete by Llama2 (Ctrl+C to exit): ")
        answer = predict(prompt, pipe)
        print(answer)


def main(args=None):
    """
    The main method for parsing command-line arguments and starting the training.

    :param args: the commandline arguments, uses sys.argv if not supplied
    :type args: list
    """
    parser = argparse.ArgumentParser(
        description="Llama2 - Interact with a model",
        prog="llama2_interact",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model_dir', metavar="PATH", type=str, required=True, help='The dir with the fine-tuned model.')
    parser.add_argument('--base_model', metavar="NAME_OR_PATH", type=str, default="NousResearch/Llama-2-7b-chat-hf", required=False, help='The name/dir base model to use')
    parser.add_argument('--max_length', metavar="INT", type=int, default=200, required=False, help='The maximum length for the responses.')
    parsed = parser.parse_args(args=args)
    interact(parsed.model_dir, base_model=parsed.base_model, max_length=parsed.max_length)


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
