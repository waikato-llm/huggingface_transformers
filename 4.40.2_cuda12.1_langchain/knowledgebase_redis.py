import json
import traceback

from datetime import datetime
from rdh import Container, MessageContainer, create_parser, configure_redis, run_harness, log
from knowledgebase_common import (DEFAULT_PROMPT, PROMPT_PLACEHOLDERS, load_embeddings, load_tokenizer_and_model,
                                  create_prompt_template, create_qa_chain, create_retriever, create_database,
                                  create_pipeline, clean_response)


def process_prompt(msg_cont):
    """
    Processes the message container, loading the image from the message and forwarding the predictions.

    :param msg_cont: the message container to process
    :type msg_cont: MessageContainer
    """
    config = msg_cont.params.config

    def ask(question):
        context = config.retriever.invoke(question)
        return (config.chain({"input_documents": context, "question": question}, return_only_outputs=True))['output_text']

    try:
        start_time = datetime.now()
        if config.verbose:
            log("process_prompts - start processing prompt")
        # read data
        d = json.loads(msg_cont.message['data'].decode())

        text = d["prompt"] if ("prompt" in d) else ""
        answer = ask(text)
        answer = clean_response(answer, raw=config.raw)

        msg_cont.params.redis.publish(msg_cont.params.channel_out, answer)
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
    parser = create_parser('knowledgebase (Redis)', prog="kb_redis", prefix="redis_")
    parser.add_argument('--model', type=str, metavar="NAME_OR_DIR", required=False, default="microsoft/Phi-3-mini-4k-instruct", help='The name or directory of the fine-tuned model')
    parser.add_argument('--attn_implementation', type=str, required=False, default=None, help='The type of attention implementation to use, e.g., flash_attention_2')
    parser.add_argument('--device', type=str, required=False, default="cuda", help='The device to run the inference on, eg "cuda" or "cpu"')
    parser.add_argument('--embeddings', type=str, required=False, default=None, help='The name of the embeddings model to use if not the default one')
    parser.add_argument('--qna_prompt_template_file', type=str, required=False, default=None, help='The plain-text file with the prompt template for overriding the default one; supported placeholders: ' + ", ".join(PROMPT_PLACEHOLDERS))
    parser.add_argument('--prompt', type=str, required=False, default=DEFAULT_PROMPT, help='The prompt to use.')
    parser.add_argument('--plain_text_context', type=str, required=False, default=None, help='The plain-text file with the additional context to use.')
    parser.add_argument('--input', help='The path to the PDF/text file(s) or dir(s) with PDF/text files to load into the vector store and use as context', required=True, default=None, nargs="+")
    parser.add_argument('--chunk_size', type=int, default=4000, help='The size of the chunks to create from the documents.')
    parser.add_argument('--chunk_overlap', type=int, default=20, help='The overlap between the chunks.')
    parser.add_argument('--db_dir', help='The directory to store the vector store in', required=False, default="db")
    parser.add_argument('--max_new_tokens', type=int, default=300, help='The maximum number of tokens to generate with the pipeline.')
    parser.add_argument('--search_type', type=str, default="similarity", help='the search type to use: similarity (default), mmr, similarity_score_threshold.')
    parser.add_argument('--num_docs', type=int, default=3, help='The number of documents to retrieve from the vector store.')
    parser.add_argument('--fetch_k', type=int, default=20, help='The amount of documents to pass to MMR algorithm.')
    parser.add_argument('--lambda_mult', type=float, default=0.5, help='The diversity of results returned by MMR.')
    parser.add_argument('--score_threshold', type=float, default=0.8, help='The minimum relevance threshold for "similarity_score_threshold".')
    parser.add_argument('--raw', action="store_true", help='Whether to return the raw responses rather than attempting to clean them up.')
    parser.add_argument('--verbose', required=False, action='store_true', help='whether to be more verbose with the output')

    parsed = parser.parse_args(args=args)

    embeddings = load_embeddings(parsed.device, model_name=parsed.embeddings)
    prompt = create_prompt_template(parsed.prompt, plain_text_context=parsed.plain_text_context,
                                    qna_prompt_template_file=parsed.qna_prompt_template_file)
    tokenizer, model = load_tokenizer_and_model(parsed.model, attn_implementation=parsed.attn_implementation)
    pipeline = create_pipeline(tokenizer, model, parsed.max_new_tokens)
    db = create_database(parsed.input, embeddings, chunk_size=parsed.chunk_size, chunk_overlap=parsed.chunk_overlap, persist_directory=parsed.db_dir)
    retriever = create_retriever(db, num_docs=parsed.num_docs, search_type=parsed.search_type,
                                 fetch_k=parsed.fetch_k, lambda_mult=parsed.lambda_mult,
                                 score_threshold=parsed.score_threshold)
    chain = create_qa_chain(pipeline, prompt_template=prompt)

    config = Container()
    config.chain = chain
    config.retriever = retriever
    config.raw = parsed.raw
    config.verbose = parsed.verbose

    params = configure_redis(parsed, config=config)
    run_harness(params, process_prompt)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print(traceback.format_exc())
