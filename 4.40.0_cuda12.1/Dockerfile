ARG CUDA="12.1.0"
ARG CUDNN="8"
ARG PYTORCH="2.1.1"
ARG PYTORCH_CUDA="cu121"
ARG TORCH_VISION="0.16"
ARG TORCH_AUDIO="2.1.0"

ARG DOCKER_REGISTRY=public.aml-repo.cms.waikato.ac.nz:443/
FROM ${DOCKER_REGISTRY}nvidia/cuda:${CUDA}-cudnn${CUDNN}-devel-ubuntu20.04

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && \
    apt-get install -y libglib2.0-0 libsm6 libxrender-dev libxext6 libgl1-mesa-glx python3-dev python3-pip git wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir -U --extra-index-url https://download.pytorch.org/whl/$PYTORCH_CUDA \
       "torch==$PYTORCH.*" \
       "torchvision==$TORCH_VISION.*" \
       "torchaudio==$TORCH_AUDIO.*"

RUN pip install --no-cache-dir transformers==4.40.0
RUN pip install --no-cache-dir timm==0.9.9 xformers==0.0.22 && \
    pip install --no-cache-dir einops==0.7.0 accelerate==0.20.3 && \
    pip install --no-cache-dir bitsandbytes==0.41.2.post2 && \
    pip install --no-cache-dir scipy==1.11.3 && \
    pip install --no-cache-dir datasets==2.14.6 && \
    pip install --no-cache-dir peft==0.6.1 && \
    pip install --no-cache-dir git+https://github.com/huggingface/trl.git@393dbf67494771170cb8fede30b6e77c39d85841 && \
    pip install --no-cache-dir deepspeed==0.11.2 && \
    pip install --no-cache-dir sentencepiece==0.1.99 "protobuf==3.20.*" && \
    pip install --no-cache-dir "tensorboard==2.15.1"

COPY bash.bashrc /etc/bash.bashrc
