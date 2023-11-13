ARG PYTORCH="2.0.1"
ARG CUDA="11.7"
ARG CUDNN="8"

ARG DOCKER_REGISTRY=public.aml-repo.cms.waikato.ac.nz:443/
FROM ${DOCKER_REGISTRY}pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

# update NVIDIA repo key
# https://developer.nvidia.com/blog/updating-the-cuda-linux-gpg-repository-key/
ARG distro=ubuntu2004
ARG arch=x86_64
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/$distro/$arch/3bf863cc.pub

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && \
    apt-get install -y libglib2.0-0 libsm6 libxrender-dev libxext6 libgl1-mesa-glx python3-dev python3-pip git wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir transformers==4.35.0
RUN pip install --no-cache-dir timm==0.9.9 xformers==0.0.22 && \
    pip install --no-cache-dir einops==0.7.0 accelerate==0.20.3 && \
    pip install --no-cache-dir bitsandbytes==0.39.0 && \
    pip install --no-cache-dir scipy==1.11.3 && \
    pip install --no-cache-dir datasets==2.14.6 && \
    pip install --no-cache-dir peft==0.6.1 && \
    pip install --no-cache-dir trl==0.7.2 && \
    pip install --no-cache-dir deepspeed==0.11.2 && \
    pip install --no-cache-dir sentencepiece==0.1.99 "protobuf==3.20.*" && \
    pip install --no-cache-dir "tensorboard==2.15.1"

# original path, which doesn't contain any CUDA libraries:
# LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64
ENV LD_LIBRARY_PATH /usr/local/cuda-11/lib64

COPY bash.bashrc /etc/bash.bashrc
