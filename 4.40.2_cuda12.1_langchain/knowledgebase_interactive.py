import argparse
import traceback

from knowledgebase_common import DEFAULT_PROMPT, load_embeddings, load_tokenizer_and_model, create_prompt_template, create_qa_chain, create_retriever, create_database, create_pipeline


def query(chain, retriever):
    """
    Lets the user query the document store (used as context).

    :param chain: the Q&A chain to use
    :param retriever: the document retriever for the vector store
    """
    def ask(question):
        context = retriever.invoke(question)
        return (chain({"input_documents": context, "question": question}, return_only_outputs=True))['output_text']

    while True:
        user_question = input("\nPlease enter your question: ")
        if (user_question is None) or (user_question == ""):
            break
        answer = ask(user_question)
        answer = (answer.split("<|assistant|>")[-1]).strip()
        print("Answer:\n", answer)


def main(args=None):
    """
    The main method for parsing command-line arguments and starting the training.

    :param args: the commandline arguments, uses sys.argv if not supplied
    :type args: list
    """

    parser = argparse.ArgumentParser(
        description="knowledgebase - interactive",
        prog="kb_interactive",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', type=str, metavar="NAME_OR_DIR", required=False, default="microsoft/Phi-3-mini-4k-instruct", help='The name or directory of the fine-tuned model')
    parser.add_argument('--device', type=str, required=False, default="cuda", help='The device to run the inference on, eg "cuda" or "cpu"')
    parser.add_argument('--prompt', type=str, required=False, default=DEFAULT_PROMPT, help='The prompt to use.')
    parser.add_argument('--input', help='The path to the PDF/text file(s) or dir(s) with PDF/text files to load into the vector store and use as context', required=True, default=None, nargs="+")
    parser.add_argument('--chunk_size', type=int, default=4000, help='The size of the chunks to create from the documents.')
    parser.add_argument('--chunk_overlap', type=int, default=20, help='The overlap between the chunks.')
    parser.add_argument('--persist_directory', help='The directory to store the vector store in', required=False, default="db")
    parser.add_argument('--max_new_tokens', type=int, default=300, help='The maximum number of tokens to generate with the pipeline.')
    parser.add_argument('--num_docs', type=int, default=3, help='The number of documents to retrieve from the vector store.')
    parsed = parser.parse_args(args=args)

    embeddings = load_embeddings(parsed.device)
    prompt = create_prompt_template(parsed.prompt)
    tokenizer, model = load_tokenizer_and_model(parsed.model)
    pipeline = create_pipeline(tokenizer, model, parsed.max_new_tokens)
    db = create_database(parsed.input, embeddings, chunk_size=parsed.chunk_size, chunk_overlap=parsed.chunk_overlap, persist_directory=parsed.db_dir)
    retriever = create_retriever(db, num_docs=parsed.num_docs)
    chain = create_qa_chain(pipeline, prompt_template=prompt)
    query(chain, retriever)


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
