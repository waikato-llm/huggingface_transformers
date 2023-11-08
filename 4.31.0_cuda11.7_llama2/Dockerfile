FROM waikatodatamining/pytorch-huggingface-transformers:4.31.0_cuda11.7

COPY bash.bashrc /etc/bash.bashrc
COPY *.py /opt/llama2/
COPY llama2_finetune /usr/local/bin/
COPY llama2_interact /usr/local/bin/

ENV PYTHONPATH=/opt/llama2
