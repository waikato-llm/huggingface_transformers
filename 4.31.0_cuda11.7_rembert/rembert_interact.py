# Based on example here:
# https://huggingface.co/docs/transformers/tasks/language_modeling

import argparse
import traceback

from rembert_common import MODEL, load_model


def interact(repo_or_dir=MODEL, max_new_tokens=100, do_sample=True, top_k=50, top_p=0.95):
    """
    Lets the user interact with a RemBERT model.

    :param repo_or_dir: the huggingface repo or local dir to load the model from
    :type repo_or_dir: str
    :param max_new_tokens: the maximum number of new tokens to generate
    :type max_new_tokens: int
    :param do_sample: whether to perform sampling
    :type do_sample: bool
    :param top_k: top k results to return
    :type top_k: int
    :param top_p: the minimum probability for the top k results
    :type top_p: float
    """
    tokenizer, model = load_model(repo_or_dir)

    while True:
        print("\nPlease enter the text to complete (Ctrl+C to exit):")
        prompt = input()
        inputs = tokenizer(prompt, return_tensors="pt").input_ids
        outputs = model.generate(inputs, max_new_tokens=max_new_tokens, do_sample=do_sample, top_k=top_k, top_p=top_p)
        text = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        if len(text) > 0:
            print("RemBERT's response:\n")
            print(text[0])
        else:
            print("RemBERT has nothing to say...")


def main(args=None):
    """
    The main method for parsing command-line arguments and starting the training.

    :param args: the commandline arguments, uses sys.argv if not supplied
    :type args: list
    """

    parser = argparse.ArgumentParser(
        description="RemBERT - Let a model finish prompts entered by the user",
        prog="rembert_interact",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--repo_or_dir', type=str, default=MODEL, required=False, help='The huggingface repo or local dir to load the model from')
    parser.add_argument('--max_new_tokens', type=int, default=100, required=False, help='The maximum number of new tokens to generate.')
    parser.add_argument("--no_sample", action="store_true", required=False, help="Whether to avoid doing sampling.")
    parser.add_argument('--top_k', type=int, default=50, help='Top k results to return.')
    parser.add_argument('--top_p', type=float, default=.95, help='The minimum probability for the top k results.')
    parser.add_argument('--do_sample', action="store_true", help='Sampling when generating.')
    parsed = parser.parse_args(args=args)

    interact(repo_or_dir=parsed.repo_or_dir, max_new_tokens=parsed.max_new_tokens,
             do_sample=not parsed.no_sample, top_k=parsed.top_k, top_p=parsed.top_p)


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
