FROM waikatodatamining/pytorch-huggingface-transformers:4.40.2_cuda12.1

RUN pip install --no-cache-dir \
        soundfile==0.12.1 \
        librosa==0.10.2.post1 \
        resampy==0.4.3 \
        evaluate==0.4.2 \
        jiwer==3.0.4

RUN pip install --no-cache-dir redis && \
    pip install --no-cache-dir "redis-docker-harness==0.0.4"

COPY bash.bashrc /etc/bash.bashrc
COPY kb_* /usr/bin/
COPY *.py /opt/knowledgebase/
