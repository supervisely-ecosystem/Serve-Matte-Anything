FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt update && apt install python3-pip -y
RUN apt-get install -y git

ARG USE_CUDA=0

ENV DEBIAN_FRONTEND=noninteractive
ENV AM_I_DOCKER True
ENV BUILD_WITH_CUDA "${USE_CUDA}"
ENV TORCH_CUDA_ARCH_LIST "8.9"
ENV CUDA_HOME /usr/local/cuda-11.8

RUN pip3 install networkx==2.8.8
RUN pip3 install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu118

RUN apt-get install ffmpeg libgeos-dev libsm6 libxext6 libexiv2-dev libxrender-dev libboost-all-dev -y

RUN git clone https://github.com/hustvl/Matte-Anything.git
RUN pip3 install git+https://github.com/facebookresearch/segment-anything.git
RUN python3 -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

WORKDIR ./Matte-Anything
RUN git clone https://github.com/IDEA-Research/GroundingDINO.git
RUN python3 -m pip install --no-cache-dir wheel
RUN python3 -m pip install --no-cache-dir --no-build-isolation -e GroundingDINO
WORKDIR /

RUN pip3 install opencv-python==4.8.0.74
RUN pip3 install gradio==3.41.2
RUN pip3 install fairscale

RUN python3 -m pip install supervisely==6.73.82
RUN pip3 install urllib3==1.26.17
RUN pip3 install starlette==0.35.1

RUN mkdir -p pretrained
ADD pretrained /pretrained/