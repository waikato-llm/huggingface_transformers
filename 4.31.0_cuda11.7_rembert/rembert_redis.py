import json
import traceback

from datetime import datetime
from rdh import Container, MessageContainer, create_parser, configure_redis, run_harness, log
from rembert_common import load_model, predict, MODEL


def format_output(raw_output):
    return raw_output.split("### Response:")[1].strip()


def process_prompt(msg_cont):
    """
    Processes the message container, loading the image from the message and forwarding the predictions.

    :param msg_cont: the message container to process
    :type msg_cont: MessageContainer
    """
    config = msg_cont.params.config

    try:
        start_time = datetime.now()
        if config.verbose:
            log("process_prompts - start processing prompt")
        # read data
        d = json.loads(msg_cont.message['data'].decode())

        prompt = d["prompt"] if ("prompt" in d) else ""
        text = predict(config.tokenizer, config.model, prompt,
                       max_new_tokens=config.max_new_tokens, do_sample=config.do_sample,
                       top_k=config.top_k, top_p=config.top_p)
        output = ""
        if len(text) > 0:
            output = "\n".join(text)

        msg_cont.params.redis.publish(msg_cont.params.channel_out, output)
        if config.verbose:
            log("process_prompts - response string published: %s" % msg_cont.params.channel_out)
            end_time = datetime.now()
            processing_time = end_time - start_time
            processing_time = int(processing_time.total_seconds() * 1000)
            log("process_prompts - finished processing prompt: %d ms" % processing_time)
    except KeyboardInterrupt:
        msg_cont.params.stopped = True
    except:
        log("process_prompts - failed to process: %s" % traceback.format_exc())


def main(args=None):
    """
    Performs the predictions.
    Use -h to see all options.

    :param args: the command-line arguments to use, uses sys.argv if None
    :type args: list
    """
    parser = create_parser('RemBERT - Prediction (Redis)', prog="rembert_redis", prefix="redis_")
    parser.add_argument('--repo_or_dir', type=str, default=MODEL, required=False, help='The huggingface repo or local dir to load the model from')
    parser.add_argument('--max_new_tokens', type=int, default=100, required=False, help='The maximum number of new tokens to generate.')
    parser.add_argument("--no_sample", action="store_true", required=False, help="Whether to avoid doing sampling.")
    parser.add_argument('--top_k', type=int, default=50, help='Top k results to return.')
    parser.add_argument('--top_p', type=float, default=.95, help='The minimum probability for the top k results.')
    parser.add_argument('--verbose', required=False, action='store_true', help='whether to be more verbose with the output')

    parsed = parser.parse_args(args=args)

    model, tokenizer = load_model(parsed.repo_or_dir)

    config = Container()
    config.model = model
    config.tokenizer = tokenizer
    config.max_new_tokens = parsed.max_new_tokens
    config.top_p = parsed.top_p
    config.top_k = parsed.top_k
    config.do_sample = not parsed.no_sample
    config.verbose = parsed.verbose

    params = configure_redis(parsed, config=config)
    run_harness(params, process_prompt)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print(traceback.format_exc())
