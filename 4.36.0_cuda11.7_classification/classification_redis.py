import json
import traceback

from datetime import datetime
from rdh import Container, MessageContainer, create_parser, configure_redis, run_harness, log
from classification_common import load_finetuned_model, predict


def process_text(msg_cont):
    """
    Processes the message container, loading the image from the message and forwarding the predictions.

    :param msg_cont: the message container to process
    :type msg_cont: MessageContainer
    """
    config = msg_cont.params.config

    try:
        start_time = datetime.now()
        if config.verbose:
            log("process_texts - start processing text")
        # read data
        d = json.loads(msg_cont.message['data'].decode())

        text = d["text"] if ("text" in d) else ""
        output = predict(text, config.pipe)
        if output is None:
            output = {}

        msg_cont.params.redis.publish(msg_cont.params.channel_out, json.dumps(output))
        if config.verbose:
            log("process_texts - response string published: %s" % msg_cont.params.channel_out)
            end_time = datetime.now()
            processing_time = end_time - start_time
            processing_time = int(processing_time.total_seconds() * 1000)
            log("process_texts - finished processing text: %d ms" % processing_time)
    except KeyboardInterrupt:
        msg_cont.params.stopped = True
    except:
        log("process_texts - failed to process: %s" % traceback.format_exc())


def main(args=None):
    """
    Performs the predictions.
    Use -h to see all options.

    :param args: the command-line arguments to use, uses sys.argv if None
    :type args: list
    """
    parser = create_parser('Text classification - Prediction (Redis)', prog="classification_redis", prefix="redis_")
    parser.add_argument('--model_dir', metavar="PATH", type=str, required=True, help='The directory with the fine-tuned model.')
    parser.add_argument('--verbose', required=False, action='store_true', help='whether to be more verbose with the output')

    parsed = parser.parse_args(args=args)

    pipe = load_finetuned_model(parsed.model_dir)

    config = Container()
    config.pipe = pipe
    config.verbose = parsed.verbose

    params = configure_redis(parsed, config=config)
    run_harness(params, process_text)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print(traceback.format_exc())
