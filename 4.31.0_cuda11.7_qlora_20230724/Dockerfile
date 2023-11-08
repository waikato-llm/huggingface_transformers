FROM waikatodatamining/pytorch-huggingface-transformers:4.31.0_cuda11.7

WORKDIR /opt

RUN git clone https://github.com/artidoro/qlora.git && \
    cd qlora && \
    git reset --hard 7f4e95a68dc076bea9b3a413d2b512eca6d004e5

COPY bash.bashrc /etc/bash.bashrc
COPY qlora /usr/local/bin/

ENV PYTHONPATH=/opt/qlora
