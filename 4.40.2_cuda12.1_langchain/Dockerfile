FROM waikatodatamining/pytorch-huggingface-transformers:4.40.2_cuda12.1

# https://www.pragnakalp.com/leverage-phi-3-exploring-rag-based-qna-with-microsofts-phi-3/
RUN pip install --no-cache-dir \
        langchain==0.1.17 \
        langchain-community==0.0.37 \
        chromadb==0.5.0 \
        pypdf==4.2.0 \
        openai==1.26.0 \
        sentence-transformers==2.7.0 \
        tiktoken==0.7.0 \
        pytest==8.2.1

# https://huggingface.co/docs/transformers/perf_infer_gpu_one#flashattention-2
RUN pip install --no-cache-dir --no-build-isolation flash-attn==2.5.8 

COPY bash.bashrc /etc/bash.bashrc

