FROM waikatodatamining/pytorch-huggingface-transformers:4.31.0_cuda11.7

RUN pip install --no-cache-dir \
        "tensorboard==2.13.0" \
        "simple-file-poller>=0.0.9" \
        redis \
        "redis-docker-harness==0.0.4"

COPY bash.bashrc /etc/bash.bashrc
COPY *.py /opt/llama2/
COPY llama2_finetune /usr/local/bin/
COPY llama2_interact /usr/local/bin/
COPY llama2_redis /usr/local/bin/

ENV PYTHONPATH=/opt/llama2
