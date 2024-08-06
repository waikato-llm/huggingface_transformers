FROM waikatodatamining/pytorch-huggingface-transformers:4.43.1_cuda12.1

RUN pip install --no-cache-dir redis && \
    pip install --no-cache-dir "redis-docker-harness==0.0.4" \
    pip install --no-cache-dir scipy==1.14.0

COPY bash.bashrc /etc/bash.bashrc
