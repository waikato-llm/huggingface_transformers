FROM waikatodatamining/pytorch-huggingface-transformers:4.40.0_cuda11.7

# https://www.pragnakalp.com/leverage-phi-3-exploring-rag-based-qna-with-microsofts-phi-3/
RUN pip install --no-cache-dir \
        langchain==0.1.17 \
        langchain-community==0.0.37 \
        chromadb==0.5.0 \
        pypdf==4.2.0 \
        openai==1.26.0 \
        sentence-transformers==2.7.0

# https://huggingface.co/docs/transformers/perf_infer_gpu_one#flashattention-2
RUN pip install --no-cache-dir --no-build-isolation flash-attn==2.5.8

COPY bash.bashrc /etc/bash.bashrc

