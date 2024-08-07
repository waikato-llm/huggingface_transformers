import argparse
import json
import traceback

from knowledgebase_common import (DEFAULT_PROMPT, PROMPT_PLACEHOLDERS, load_embeddings, load_tokenizer_and_model,
                                  create_prompt_template, create_qa_chain, create_retriever, create_database,
                                  create_pipeline, clean_response)


def evaluate(chain, retriever, questions: str, output: str = None, raw: bool = False, params: argparse.Namespace = None,
             response_start: str = "<|assistant|>", response_end: str = None):
    """
    Lets the user query the document store (used as context).

    :param chain: the Q&A chain to use
    :param retriever: the document retriever for the vector store, skips RAG if None
    :param questions: the file with the questions to run through the model
    :type questions: str
    :param output: the JSON file to store the parameters/results in, outputs results on stdout if None
    :type output: str
    :param raw: whether to return the raw answers
    :type raw: bool
    :param params: the parameters that were used, can be None
    :type params: argparse.Namespace
    :param response_start: the string/tag that identifies the start of the answer
    :type response_start: str
    :param response_end: the string/tag that identifies the end of the answer, ignored if None/empty
    :type response_end: str
    """
    print("--> asking questions from: %s" % questions)

    # generate output?
    result = None
    if output is not None:
        result = dict()
        if params is not None:
            result["params"] = dict()
            for att in dir(params):
                if att.startswith("_"):
                    continue
                result["params"][att] = getattr(params, att)
        result["qna"] = list()

    count = 0

    def ask(question):
        if retriever is not None:
            context = retriever.invoke(question)
        else:
            context = []
        answer = (chain({"input_documents": context, "question": question}, return_only_outputs=True))['output_text']

        if result is not None:
            result["qna"].append({
                "index": count,
                "question": line,
                "documents": [doc.page_content for doc in context],
                "answer": {
                    "raw": answer,
                    "clean": clean_response(answer, raw=False, response_start=response_start, response_end=response_end),
                },
            })
        return answer

    # iterate questions
    with open(questions, "r") as fp:
        for line in fp.readlines():
            line = line.strip()
            if (len(line) == 0) or line.startswith("#"):
                continue
            count += 1
            print("\n--> question %d:\n%s" % (count, line))
            answer = ask(line)
            answer = clean_response(answer, raw=raw)
            print("\n--> answer %d:\n%s" % (count, answer))

    # write output
    if result is not None:
        print("--> writing results to: %s" % output)
        with open(output, "w") as fp:
            json.dump(result, fp, indent=2, ensure_ascii=False)


def main(args=None):
    """
    The main method for parsing command-line arguments and starting the training.

    :param args: the commandline arguments, uses sys.argv if not supplied
    :type args: list
    """

    parser = argparse.ArgumentParser(
        description="knowledgebase - evaluate",
        prog="kb_eval",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', type=str, metavar="NAME_OR_DIR", required=False, default="microsoft/Phi-3-mini-4k-instruct", help='The name or directory of the fine-tuned model')
    parser.add_argument('--attn_implementation', type=str, required=False, default=None, help='The type of attention implementation to use, e.g., flash_attention_2')
    parser.add_argument('--device', type=str, required=False, default="cuda", help='The device to run the inference on, eg "cuda" or "cpu"')
    parser.add_argument('--embeddings', type=str, required=False, default=None, help='The name of the embeddings model to use if not the default one')
    parser.add_argument('--qna_prompt_template_file', type=str, required=False, default=None, help='The plain-text file with the prompt template for overriding the default one; the {context} placeholder must always be present; supported placeholders: ' + ", ".join(PROMPT_PLACEHOLDERS))
    parser.add_argument('--prompt', type=str, required=False, default=DEFAULT_PROMPT, help='The prompt to use.')
    parser.add_argument('--plain_text_context', type=str, required=False, default=None, help='The plain-text file with the additional context to use.')
    parser.add_argument('--input', help='The path to the (optional) PDF/text file(s) or dir(s) with PDF/text files to load into the vector store and use as context', required=False, default=None, nargs="*")
    parser.add_argument('--chunk_size', type=int, default=4000, help='The size of the chunks to create from the documents.')
    parser.add_argument('--chunk_overlap', type=int, default=20, help='The overlap between the chunks.')
    parser.add_argument('--db_init', action="store_true", help='Whether to remove any existing vector store first.')
    parser.add_argument('--db_dir', help='The directory to store the vector store in', required=False, default="db")
    parser.add_argument('--max_new_tokens', type=int, default=300, help='The maximum number of tokens to generate with the pipeline.')
    parser.add_argument('--do_sample', action="store_true", help='Whether to perform sampling.')
    parser.add_argument('--temperature', type=float, default=0.7, help='For adjusting the probability distribution.')
    parser.add_argument('--search_type', type=str, default="similarity", help='the search type to use: similarity (default), mmr, similarity_score_threshold.')
    parser.add_argument('--num_docs', type=int, default=3, help='The number of documents to retrieve from the vector store.')
    parser.add_argument('--fetch_k', type=int, default=20, help='The amount of documents to pass to MMR algorithm.')
    parser.add_argument('--lambda_mult', type=float, default=0.5, help='The diversity of results returned by MMR.')
    parser.add_argument('--score_threshold', type=float, default=0.8, help='The minimum relevance threshold for "similarity_score_threshold".')
    parser.add_argument('--raw', action="store_true", help='Whether to return the raw responses rather than attempting to clean them up.')
    parser.add_argument('--questions', type=str, required=True, default=None, help='The plain-text file with the questions to prompt the model with; one question per line; empty lines and lines starting with # get ignored.')
    parser.add_argument('--response_start', type=str, required=False, default="<|assistant|>", help='The string/tag that identifies the start of the answer.')
    parser.add_argument('--response_end', type=str, required=False, default=None, help='The string/tag that identifies the end of the answer; ignored if not provided.')
    parser.add_argument('--output', type=str, required=False, default=None, help='The JSON file to store the parameters and results in; outputs results on stdout if not provided.')
    parsed = parser.parse_args(args=args)

    embeddings = load_embeddings(parsed.device, model_name=parsed.embeddings)
    prompt = create_prompt_template(parsed.prompt, plain_text_context=parsed.plain_text_context,
                                    qna_prompt_template_file=parsed.qna_prompt_template_file)
    tokenizer, model = load_tokenizer_and_model(parsed.model, attn_implementation=parsed.attn_implementation)
    pipeline = create_pipeline(tokenizer, model, max_new_tokens=parsed.max_new_tokens, do_sample=parsed.do_sample, temperature=parsed.temperature)
    if parsed.input is None:
        retriever = None
    else:
        db = create_database(parsed.input, embeddings, chunk_size=parsed.chunk_size, chunk_overlap=parsed.chunk_overlap,
                             persist_directory=parsed.db_dir, initialize=parsed.db_init)
        retriever = create_retriever(db, num_docs=parsed.num_docs, search_type=parsed.search_type,
                                     fetch_k=parsed.fetch_k, lambda_mult=parsed.lambda_mult,
                                     score_threshold=parsed.score_threshold)
    chain = create_qa_chain(pipeline, prompt_template=prompt)
    evaluate(chain, retriever, parsed.questions, raw=parsed.raw, params=parsed, output=parsed.output,
             response_start=parsed.response_start, response_end=parsed.response_end)


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
