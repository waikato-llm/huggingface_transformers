import argparse
import json
import os
import traceback

from sfp import Poller
from rembert_common import load_model, predict, MODEL


SUPPORTED_EXTS = [".json"]
""" supported file extensions (lower case with dot). """


def check_prompt(fname, poller):
    """
    Check method that ensures the json is valid.

    :param fname: the file to check
    :type fname: str
    :param poller: the Poller instance that called the method
    :type poller: Poller
    :return: True if complete
    :rtype: bool
    """
    try:
        with open(fname, "r") as fp:
            json.load(fp)
        result = True
    except:
        result = False
    poller.debug("JSON complete:", fname, "->", result)
    return result


def format_output(raw_output):
    return raw_output.split("### Response:")[1].strip()


def process_prompt(fname, output_dir, poller):
    """
    Method for processing a JSON prompt.

    :param fname: the prompt to process
    :type fname: str
    :param output_dir: the directory to write the prompt to
    :type output_dir: str
    :param poller: the Poller instance that called the method
    :type poller: Poller
    :return: the list of generated output files
    :rtype: list
    """
    result = []
    try:
        output_response = "{}/{}{}".format(output_dir, os.path.splitext(os.path.basename(fname))[0], ".txt")

        with open(fname, "r") as fp:
            d = json.load(fp)
        prompt = d["prompt"] if ("prompt" in d) else ""
        text = predict(poller.params.tokenizer, poller.params.model, prompt,
                       max_new_tokens=poller.params.max_new_tokens, do_sample=poller.params.do_sample,
                       top_k=poller.params.top_k, top_p=poller.params.top_p)
        output = ""
        if len(text) > 0:
            output = "\n".join(text)
        with open(output_response, "w") as fp:
            fp.write(output)

        result.append(output_response)
    except KeyboardInterrupt:
        poller.keyboard_interrupt()
    except:
        poller.error("Failed to process prompt: %s\n%s" % (fname, traceback.format_exc()))
    return result


def main(args=None):
    """
    The main method for parsing command-line arguments and starting the training.

    :param args: the commandline arguments, uses sys.argv if not supplied
    :type args: list
    """

    parser = argparse.ArgumentParser(
        description="RemBERT - Prediction (file-polling)",
        prog="rembert_poll",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--repo_or_dir', type=str, default=MODEL, required=False, help='The huggingface repo or local dir to load the model from')
    parser.add_argument('--max_new_tokens', type=int, default=100, required=False, help='The maximum number of new tokens to generate.')
    parser.add_argument("--no_sample", action="store_true", required=False, help="Whether to avoid doing sampling.")
    parser.add_argument('--top_k', type=int, default=50, help='Top k results to return.')
    parser.add_argument('--top_p', type=float, default=.95, help='The minimum probability for the top k results.')
    parser.add_argument('--prediction_in', help='Path to the images to process', required=True, default=None)
    parser.add_argument('--prediction_out', help='Path to the folder for the prediction files', required=True, default=None)
    parser.add_argument('--prediction_tmp', help='Path to the temporary folder for the prediction files', required=False, default=None)
    parser.add_argument('--poll_wait', type=float, help='poll interval in seconds when not using watchdog mode', required=False, default=1.0)
    parser.add_argument('--continuous', action='store_true', help='Whether to continuously load test images and perform prediction', required=False, default=False)
    parser.add_argument('--use_watchdog', action='store_true', help='Whether to react to file creation events rather than performing fixed-interval polling', required=False, default=False)
    parser.add_argument('--watchdog_check_interval', type=float, help='check interval in seconds for the watchdog', required=False, default=10.0)
    parser.add_argument('--delete_input', action='store_true', help='Whether to delete the input images rather than move them to --prediction_out directory', required=False, default=False)
    parser.add_argument('--verbose', action='store_true', help='Whether to output more logging info', required=False, default=False)
    parser.add_argument('--quiet', action='store_true', help='Whether to suppress output', required=False, default=False)
    parsed = parser.parse_args(args=args)

    model, tokenizer = load_model(parsed.repo_or_dir)

    poller = Poller()
    poller.input_dir = parsed.prediction_in
    poller.output_dir = parsed.prediction_out
    poller.tmp_dir = parsed.prediction_tmp
    poller.extensions = SUPPORTED_EXTS
    poller.delete_input = parsed.delete_input
    poller.verbose = parsed.verbose
    poller.progress = not parsed.quiet
    poller.check_file = check_prompt
    poller.process_file = process_prompt
    poller.poll_wait = parsed.poll_wait
    poller.continuous = parsed.continuous
    poller.use_watchdog = parsed.use_watchdog
    poller.watchdog_check_interval = parsed.watchdog_check_interval
    poller.params.model = model
    poller.params.tokenizer = tokenizer
    poller.params.max_new_tokens = parsed.max_new_tokens
    poller.params.top_p = parsed.top_p
    poller.params.top_k = parsed.top_k
    poller.params.do_sample = not parsed.no_sample
    poller.poll()


def sys_main() -> int:
    """
    Runs the main function using the system cli arguments, and
    returns a system error code.
    :return:    0 for success, 1 for failure.
    """
    try:
        main()
        return 0
    except Exception:
        print(traceback.format_exc())
        return 1


if __name__ == '__main__':
    main()
