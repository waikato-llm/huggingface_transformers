# based on
# https://www.pragnakalp.com/leverage-phi-3-exploring-rag-based-qna-with-microsofts-phi-3/

import os
import shutil
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

DEFAULT_QNA_PROMPT_TEMPLATE = """<|system|>
{SYSTEM_PROMPT}<|end|>
<|user|>
Context:{PLAIN_TEXT_CONTEXT}
{context}

Question: {question}<|end|>
<|assistant|>"""

DEFAULT_PROMPT = 'You have been provided with the context and a question, try to find out the answer to the question only using the context information. If the answer to the question is not found within the context, return "I dont know" as the response.'

PH_SYSTEM_PROMPT = "{SYSTEM_PROMPT}"
PH_PLAIN_TEXT_CONTEXT = "{PLAIN_TEXT_CONTEXT}"
PROMPT_PLACEHOLDERS = [
    PH_SYSTEM_PROMPT,
    PH_PLAIN_TEXT_CONTEXT,
]


def load_embeddings(device: str, model_name: str = None) -> HuggingFaceEmbeddings:
    """
    Creates the embeddings for the device.

    :param device: the device to create the embeddings for, eg cuda or cpu
    :type device: str
    :param model_name: the name of the embeddings model to use instead of the default one (sentence-transformers/all-mpnet-base-v2)
    :type model_name: str
    :return: the embeddings
    :rtype: HuggingFaceEmbeddings
    """
    print("--> embeddings: %s/%s" % ("-default-" if (model_name is None) else model_name, device))
    model_kwargs = {'device': device}
    if model_name is None:
        result = HuggingFaceEmbeddings(model_kwargs=model_kwargs)
    else:
        result = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)
    return result


def load_tokenizer_and_model(model: str, attn_implementation: str = None) -> Tuple:
    """
    Loads the tokenizer/model and returns them as tuple.

    :param model: the name or path to the pretrained model to use
    :type model: str
    :param attn_implementation: the attention implementation to use, e.g., flash_attention_2, ignored if None
    :type attn_implementation: str
    :return: the tuple of tokenizer, model
    :rtype: tuple
    """
    print("--> tokenizer: %s" % model)
    tokenizer = AutoTokenizer.from_pretrained(model)

    print("--> model: %s" % model)
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=False,
    )
    if attn_implementation is None:
        print("    attn implementation: default")
        model = AutoModelForCausalLM.from_pretrained(
            model,
            device_map='auto',
            torch_dtype="auto",
            trust_remote_code=True,
            quantization_config=quant_config,
        )
    else:
        print("    attn implementation: %s" % attn_implementation)
        model = AutoModelForCausalLM.from_pretrained(
            model,
            device_map='auto',
            torch_dtype="auto",
            trust_remote_code=True,
            quantization_config=quant_config,
            attn_implementation=attn_implementation,
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
                    chunk_overlap: int = 20, persist_directory: str = "db", initialize: bool = False) -> Chroma:
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
    :param initialize: whether to remove any existing Chroma database first
    :type initialize: bool
    :return: the Chroma database instance
    :rtype: Chroma
    """
    # remove existing db?
    if initialize and os.path.exists(persist_directory):
        print("--> removing old db: %s" % persist_directory)
        shutil.rmtree(persist_directory, ignore_errors=True)

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


def create_retriever(db: Chroma, num_docs: int = 3, search_type: str = "similarity", fetch_k: int = 20,
                     lambda_mult: float = 0.5, score_threshold: float = 0.8) -> VectorStoreRetriever:
    """
    Returns the retriever for the database.

    :param db: the Chroma DB instance to use
    :type db: Chroma
    :param num_docs: the number of documents to return
    :type num_docs: int
    :param search_type: the search type to use: similarity (default), mmr, similarity_score_threshold
    :type search_type: str
    :param fetch_k: amount of documents to pass to MMR algorithm
    :type fetch_k: int
    :param lambda_mult: diversity of results returned by MMR
    :type lambda_mult: float
    :param score_threshold: Minimum relevance threshold for similarity_score_threshold
    :type score_threshold: float
    :return: the retriever
    :rtype: VectorStoreRetriever
    """
    search_kwargs = {
        "k": num_docs,
    }
    if search_type == "mmr":
        search_kwargs["fetch_k"] = fetch_k
        search_kwargs["lambda_mult"] = lambda_mult
    elif search_type == "similarity_score_threshold":
        search_kwargs["score_threshold"] = score_threshold

    result = db.as_retriever(
        search_type=search_type,
        search_kwargs=search_kwargs,
    )
    return result


def create_prompt_template(system_prompt: str = DEFAULT_PROMPT, plain_text_context: str = None,
                           qna_prompt_template_file: str = None) -> PromptTemplate:
    """
    Creates the prompt template to use.

    :param system_prompt: the system prompt to use
    :type system_prompt: str
    :param plain_text_context: optional plain text file with additional context
    :type plain_text_context: str
    :param qna_prompt_template_file: the file containing the prompt template to use, must contain placeholders PH_SYSTEM_PROMPT and PH_PLAIN_TEXT_CONTEXT
    :type qna_prompt_template_file: str
    :return: the generated template
    :rtype: PromptTemplate
    """
    plain_text_context_str = ""
    if plain_text_context is not None:
        if os.path.exists(plain_text_context):
            with open(plain_text_context, "r") as fp:
                lines = fp.readlines()
            plain_text_context_str = "\n" + "".join(lines).strip() + "\n"
        else:
            print("ERROR: plain context file not found: %s" % plain_text_context)
    if qna_prompt_template_file is None:
        qna_prompt_template = DEFAULT_QNA_PROMPT_TEMPLATE
    else:
        print("--> loading prompt template from: %s" % qna_prompt_template_file)
        with open(qna_prompt_template_file, "r") as fp:
            lines = fp.readlines()
        qna_prompt_template = "".join(lines)
    print("--> creating prompt template")
    qna_prompt_template = qna_prompt_template.replace(PH_SYSTEM_PROMPT, system_prompt)
    qna_prompt_template = qna_prompt_template.replace(PH_PLAIN_TEXT_CONTEXT, plain_text_context_str)
    print("--> prompt template")
    print(qna_prompt_template)
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


def clean_response(answer: str, raw: bool = False,
                   response_start: str = "<|assistant|>", response_end: str = None) -> str:
    """
    Cleans up the response.

    :param answer: the response to clean up
    :type answer: str
    :param raw: if True then no cleaning is attempted
    :type raw: bool
    :param response_start: the string/tag that identifies the start of the answer
    :type response_start: str
    :param response_end: the string/tag that identifies the end of the answer, ignored if None/empty
    :type response_end: str
    :return: the cleaned up response
    :rtype: str
    """
    if raw:
        return answer

    answer = (answer.split(response_start)[-1]).strip()
    if (response_end is not None) and (len(response_end) > 0):
        answer = (answer.split(response_end)[0]).strip()
    return answer
