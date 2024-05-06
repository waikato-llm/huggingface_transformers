FROM waikatodatamining/pytorch-huggingface-transformers:4.40.0_cuda11.7

RUN pip install --no-cache-dir \
        langchain==0.1.17 \
        chromadb=0.5.0 \
        pypdf==4.2.0 \
        openai==1.26.0 \
        sentence-transformers==2.7.0

COPY bash.bashrc /etc/bash.bashrc

