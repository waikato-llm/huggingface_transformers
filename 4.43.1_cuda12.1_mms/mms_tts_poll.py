import argparse
import os
import traceback
from sfp import Poller
from mms_common import load_tts, infer_tts, save_audio


SUPPORTED_EXTS = [".txt"]
""" supported file extensions (lower case with dot). """


def process_sample(fname, output_dir, poller):
    """
    Method for processing a text file.

    :param fname: the text file to process
    :type fname: str
    :param output_dir: the directory to write the text file to
    :type output_dir: str
    :param poller: the Poller instance that called the method
    :type poller: Poller
    :return: the list of generated output files
    :rtype: list
    """
    result = []
    try:
        output = "{}/{}{}".format(output_dir, os.path.splitext(os.path.basename(fname))[0], ".wav")
        with open(fname, "r") as fp:
            lines = fp.readlines()
        text = "\n".join(lines).strip()
        sample = infer_tts(poller.params.processor, poller.params.model, text)
        save_audio(sample, output)
        result.append(output)
    except KeyboardInterrupt:
        poller.keyboard_interrupt()
    except:
        poller.error("Failed to process text file: %s\n%s" % (fname, traceback.format_exc()))
    return result


def predict_on_samples(model_id, input_dir, output_dir, tmp_dir=None,
                       poll_wait=1.0, continuous=False, use_watchdog=False, watchdog_check_interval=10.0,
                       delete_input=False, verbose=False, quiet=False):
    """
    Generates audio from text files found in input_dir and outputs the audio files in output_dir.

    :param model_id: the checkpoint file to use
    :type model_id: str
    :param input_dir: the directory with the audio files
    :type input_dir: str
    :param output_dir: the output directory to move the audio files to and store the predictions
    :type output_dir: str
    :param tmp_dir: the temporary directory to store the predictions until finished
    :type tmp_dir: str
    :param poll_wait: the amount of seconds between polls when not in watchdog mode
    :type poll_wait: float
    :param continuous: whether to poll for files continuously
    :type continuous: bool
    :param use_watchdog: whether to react to file creation events rather than use fixed-interval polling
    :type use_watchdog: bool
    :param watchdog_check_interval: the interval for the watchdog process to check for files that were missed due to potential race conditions
    :type watchdog_check_interval: float
    :param delete_input: whether to delete the input audio files rather than moving them to the output directory
    :type delete_input: bool
    :param verbose: whether to output more logging information
    :type verbose: bool
    :param quiet: whether to suppress output
    :type quiet: bool
    """
    if verbose:
        print("Loading model: %s" % model_id)
    processor, model = load_tts(model_id)

    poller = Poller()
    poller.input_dir = input_dir
    poller.output_dir = output_dir
    poller.tmp_dir = tmp_dir
    poller.extensions = SUPPORTED_EXTS
    poller.delete_input = delete_input
    poller.verbose = verbose
    poller.progress = not quiet
    poller.process_file = process_sample
    poller.poll_wait = poll_wait
    poller.continuous = continuous
    poller.use_watchdog = use_watchdog
    poller.watchdog_check_interval = watchdog_check_interval
    poller.params.model = model
    poller.params.processor = processor
    poller.poll()


def main(args=None):
    """
    The main method for parsing command-line arguments and starting the training.

    :param args: the commandline arguments, uses sys.argv if not supplied
    :type args: list
    """

    parser = argparse.ArgumentParser(
        description="MMS - TTS/audio synthesis (file-polling)",
        prog="mms_tts_poll",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', metavar="ID", type=str, required=False, default="facebook/mms-tts-eng", help='The MMS TTS model to use.')
    parser.add_argument('--prediction_in', help='Path to the text files to process', required=True, default=None)
    parser.add_argument('--prediction_out', help='Path to the folder for the generated audio files', required=True, default=None)
    parser.add_argument('--prediction_tmp', help='Path to the temporary folder for the generated audio files', required=False, default=None)
    parser.add_argument('--poll_wait', type=float, help='poll interval in seconds when not using watchdog mode', required=False, default=1.0)
    parser.add_argument('--continuous', action='store_true', help='Whether to continuously load audio files and perform prediction', required=False, default=False)
    parser.add_argument('--use_watchdog', action='store_true', help='Whether to react to file creation events rather than performing fixed-interval polling', required=False, default=False)
    parser.add_argument('--watchdog_check_interval', type=float, help='check interval in seconds for the watchdog', required=False, default=10.0)
    parser.add_argument('--delete_input', action='store_true', help='Whether to delete the input text files rather than move them to --prediction_out directory', required=False, default=False)
    parser.add_argument('--verbose', action='store_true', help='Whether to output more logging info', required=False, default=False)
    parser.add_argument('--quiet', action='store_true', help='Whether to suppress output', required=False, default=False)
    parsed = parser.parse_args(args=args)

    predict_on_samples(parsed.model, parsed.prediction_in, parsed.prediction_out, tmp_dir=parsed.prediction_tmp,
                       poll_wait=parsed.poll_wait, continuous=parsed.continuous,
                       use_watchdog=parsed.use_watchdog, watchdog_check_interval=parsed.watchdog_check_interval,
                       delete_input=parsed.delete_input, verbose=parsed.verbose,
                       quiet=parsed.quiet)


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
