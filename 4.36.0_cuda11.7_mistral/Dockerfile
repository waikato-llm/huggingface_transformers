FROM waikatodatamining/pytorch-huggingface-transformers:4.36.0_cuda11.7

RUN pip install --no-cache-dir \
        "simple-file-poller>=0.0.9" \
        redis \
        "redis-docker-harness==0.0.4"

COPY bash.bashrc /etc/bash.bashrc
COPY *.py /opt/mistral/
COPY mistral_finetune /usr/local/bin/
COPY mistral_interact /usr/local/bin/
COPY mistral_redis /usr/local/bin/

ENV PYTHONPATH=/opt/mistral
