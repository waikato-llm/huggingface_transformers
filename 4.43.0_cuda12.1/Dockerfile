ARG PYTORCH="2.3.0"
ARG CUDA="12.1"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

# update NVIDIA repo key
# https://developer.nvidia.com/blog/updating-the-cuda-linux-gpg-repository-key/
ARG distro=ubuntu2204
ARG arch=x86_64
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/$distro/$arch/3bf863cc.pub

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y libglib2.0-0 libsm6 libxrender-dev libxext6 libgl1-mesa-glx python3-dev python3-pip git wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir transformers==4.43.0
RUN pip install --no-cache-dir \
      timm==0.9.16 \
      xformers==0.0.26.post1 \
      einops==0.8.0 \
      accelerate==0.30.0 \
      bitsandbytes==0.43.1 \
      scipy==1.11.3 \
      datasets==2.19.1 \
      peft==0.10.0 \
      trl==0.8.6 \
      deepspeed==0.14.2 \
      sentencepiece==0.2.0 \
      "protobuf==3.20.*" \
      "tensorboard==2.15.1"

# original path, which doesn't contain any CUDA libraries:
# LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64
ENV LD_LIBRARY_PATH=/usr/local/cuda-11/lib64

COPY bash.bashrc /etc/bash.bashrc
