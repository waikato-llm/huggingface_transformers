FROM waikatodatamining/pytorch-huggingface-transformers:4.43.1_cuda12.1

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
       libsndfile1 \
       libsndfile1-dev && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
       redis \
       "redis-docker-harness==0.0.4" \
       simple-file-poller \
       librosa==0.10.2.post1 \
       soundfile==0.12.1

COPY bash.bashrc /etc/bash.bashrc
COPY mms_asr_poll /usr/bin/
COPY mms_lid_poll /usr/bin/
COPY mms_tts_poll /usr/bin/
COPY *.py /opt/mms/
