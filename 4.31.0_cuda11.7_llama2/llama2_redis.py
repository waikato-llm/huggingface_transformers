import json
import traceback

from datetime import datetime
from rdh import Container, MessageContainer, create_parser, configure_redis, run_harness, log
from llama2_common import load_finetuned_model, build_pipeline, predict


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
        output = predict(prompt, config.pipe)
        if output is None:
            output = ""

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
    parser = create_parser('Llama2 - Prediction (Redis)', prog="llama2_redis", prefix="redis_")
    parser.add_argument('--model_dir', metavar="PATH", type=str, required=True, help='The dir with the fine-tuned model.')
    parser.add_argument('--base_model', metavar="NAME_OR_PATH", type=str, default="NousResearch/Llama-2-7b-chat-hf", required=False, help='The name/dir base model to use')
    parser.add_argument('--max_length', metavar="INT", type=int, default=200, required=False, help='The maximum length for the responses.')
    parser.add_argument('--verbose', required=False, action='store_true', help='whether to be more verbose with the output')

    parsed = parser.parse_args(args=args)

    model, tokenizer = load_finetuned_model(parsed.base_model, parsed.model_dir)
    pipe = build_pipeline(model, tokenizer, max_length=parsed.max_length)

    config = Container()
    config.pipe = pipe
    config.model = model
    config.tokenizer = tokenizer
    config.max_length = parsed.max_length
    config.verbose = parsed.verbose

    params = configure_redis(parsed, config=config)
    run_harness(params, process_prompt)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print(traceback.format_exc())
