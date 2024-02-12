import argparse
import traceback
from classification_common import load_finetuned_model, predict


def interact(model_dir):
    """
    For interacting with the model in the console.

    :param model_dir: the directory with the finetuned model
    :type model_dir: str
    """
    pipe = load_finetuned_model(model_dir)

    # predict
    while True:
        text = input("\nPlease enter the text to classify (Ctrl+C to exit): ")
        answer = predict(text, pipe)
        print(answer)


def main(args=None):
    """
    The main method for parsing command-line arguments and starting the training.

    :param args: the commandline arguments, uses sys.argv if not supplied
    :type args: list
    """
    parser = argparse.ArgumentParser(
        description="Text classification - Interact with a model",
        prog="classification_interact",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model_dir', metavar="PATH", type=str, required=True, help='The directory with the fine-tuned model.')
    parsed = parser.parse_args(args=args)
    interact(parsed.model_dir)


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
