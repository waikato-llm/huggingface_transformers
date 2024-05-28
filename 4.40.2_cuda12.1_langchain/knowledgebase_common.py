# based on
# https://www.pragnakalp.com/leverage-phi-3-exploring-rag-based-qna-with-microsofts-phi-3/

import os
from typing import Tuple, List, Union
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.vectorstores import VectorStoreRetriever
from langchain.chains.combine_documents.base import BaseCombineDocumentsChain
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from transformers import BitsAndBytesConfig
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import torch


DEFAULT_PROMPT = 'You have been provided with the context and a question, try to find out the answer to the question only using the context information. If the answer to the question is not found within the context, return "I dont know" as the response.'


def load_embeddings(device: str) -> HuggingFaceEmbeddings:
    """
    Creates the embeddings for the device.

    :param device: the device to create the embeddings for, eg cuda or cpu
    :type device: str
    :return: the embeddings
    :rtype: HuggingFaceEmbeddings
    """
    print("--> embeddings: %s" % device)
    model_kwargs = {'device': device}
    result = HuggingFaceEmbeddings(model_kwargs=model_kwargs)
    return result


def load_tokenizer_and_model(model: str) -> Tuple:
    """
    Loads the tokenizer/model and returns them as tuple.

    :param model: the name or path to the pretrained model to use
    :type model: str
    :return: the tuple of tokenizer, model
    :rtype: tuple
    """
    print("--> tokenizer: %s" % model)
    tokenizer = AutoTokenizer.from_pretrained(model)
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=False,
    )
    print("--> model: %s" % model)
    model = AutoModelForCausalLM.from_pretrained(
        model,
        device_map='auto',
        torch_dtype="auto",
        trust_remote_code=True,
        quantization_config=quant_config,
    )
    return tokenizer, model


def create_pipeline(tokenizer, model, max_new_tokens: int) -> HuggingFacePipeline:
    """
    Creates the huggingface pipeline from model/tokenizer.

    :param tokenizer: the huggingface tokenizer to use
    :param model: the huggingface model to use
    :param max_new_tokens: the maximum number of new tokens to generate
    :type max_new_tokens: int
    :return: the generated pipeline
    :rtype: HuggingFacePipeline
    """
    print("--> pipeline: max_new_tokens=%d" % max_new_tokens)
    pipe = pipeline("text-generation", tokenizer=tokenizer, model=model, max_new_tokens=max_new_tokens)
    result = HuggingFacePipeline(pipeline=pipe)
    return result


def create_database(inputs: Union[str, List[str]], embeddings: HuggingFaceEmbeddings, chunk_size: int = 4000,
                    chunk_overlap: int = 20, persist_directory: str = "db") -> Chroma:
    """
    Loads the document(s) (pdf or txt) and turns them into a Chroma database to be stored in
    the "persist_directory" directory. The database instance gets returned.

    :param inputs: the file(s) and/or dir(s) to load PDF/text files from
    :type inputs: str or list
    :param embeddings: the huggingface embeddings to use
    :type embeddings: HuggingFaceEmbeddings
    :param chunk_size: the chunk size to use when splitting the documents
    :type chunk_size: int
    :param chunk_overlap: the overlap between the chunks
    :type chunk_overlap: int
    :param persist_directory: the directory to store the Chroma database in
    :type persist_directory: str
    :return: the Chroma database instance
    :rtype: Chroma
    """
    # locate files
    if isinstance(inputs, str):
        inputs = [inputs]
    files = []
    for inp in inputs:
        if not os.path.exists(inp):
            print("Failed to locate: %s" % inp)
            continue
        if os.path.isdir(inp):
            for f in os.listdir(inp):
                if f.lower().endswith(".pdf") or f.lower().endswith(".txt"):
                    files.append(os.path.join(inp, f))
        else:
            files.append(inp)
    print("--> knowledgebase: %d file(s)" % len(files))

    # loads files and create chunks
    chunks_total = []
    for f in files:
        print("--> load document: %s" % f)
        if f.lower().endswith(".pdf"):
            loader = PyPDFLoader(f, extract_images=False)
            pages = loader.load_and_split()
        else:
            loader = TextLoader(f)
            pages = loader.load_and_split()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            add_start_index=True,
        )
        chunks = text_splitter.split_documents(pages)
        chunks_total.extend(chunks)

    # Store data into database
    print("--> store in db: %s" % persist_directory)
    db = Chroma.from_documents(chunks_total, embedding=embeddings, persist_directory=persist_directory)
    db.persist()

    # Load the database
    print("--> load db: %s" % persist_directory)
    result = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    return result


def create_retriever(db: Chroma, num_docs: int = 3) -> VectorStoreRetriever:
    """
    Returns the retriever for the database.

    :param db: the Chroma DB instance to use
    :type db: Chroma
    :param num_docs: the number of documents to return
    :type num_docs: int
    :return: the retriever
    :rtype: VectorStoreRetriever
    """
    result = db.as_retriever(search_kwargs={"k": num_docs})
    return result


def create_prompt_template(system_prompt: str = DEFAULT_PROMPT) -> PromptTemplate:
    """
    Creates the prompt template to use.

    :param system_prompt: the system prompt to use
    :type system_prompt: str
    :return: the generated template
    :rtype: PromptTemplate
    """
    print("--> creating prompt")
    qna_prompt_template = """<|system|>
    %s<|end|>
    <|user|>
    Context:
    {context}

    Question: {question}<|end|>
    <|assistant|>""" % system_prompt
    result = PromptTemplate(template=qna_prompt_template, input_variables=["context", "question"])
    return result


def create_qa_chain(pipeline: HuggingFacePipeline, prompt_template: PromptTemplate) -> BaseCombineDocumentsChain:
    """
    Creates and returns the Q&A chain.

    :param pipeline: the huggingface pipeline to use
    :type pipeline: HuggingFacePipeline
    :param prompt_template: the prompt template to apply
    :type prompt_template: PromptTemplate
    :return: the Q&A chain
    :rtype: BaseCombineDocumentsChain
    """
    print("--> q&a chain")
    result = load_qa_chain(pipeline, chain_type="stuff", prompt=prompt_template)
    return result


def clean_response(answer: str, raw: bool = False) -> str:
    """
    Cleans up the response.

    :param answer: the response to clean up
    :type answer: str
    :param raw: if True then no cleaning is attempted
    :type raw: bool
    :return: the cleaned up response
    :rtype: str
    """
    if raw:
        return answer

    answer = (answer.split("<|assistant|>")[-1]).strip()
    return answer
