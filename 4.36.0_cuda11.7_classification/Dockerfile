FROM waikatodatamining/pytorch-huggingface-transformers:4.36.0_cuda11.7

RUN pip install --no-cache-dir \
        "simple-file-poller>=0.0.9" \
        redis \
        "redis-docker-harness==0.0.4" \
        evaluate==0.4.1 \
        scikit-learn==1.4.0

COPY bash.bashrc /etc/bash.bashrc
COPY *.py /opt/classification/
COPY classification_finetune /usr/local/bin/
COPY classification_interact /usr/local/bin/
COPY classification_redis /usr/local/bin/

ENV PYTHONPATH=/opt/classification
