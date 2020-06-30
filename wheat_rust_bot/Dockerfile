FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-runtime

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_VISIBLE_DEVICES=""

RUN apt-get update -y && \
    apt-get install -y libglib2.0-0 libsm6 libxext6 libxrender-dev wget curl

RUN pip install albumentations==0.3.2
RUN pip install opencv-python==4.1.1.26
RUN pip install numpy==1.17.2
RUN pip install imageio==2.5.0
RUN pip install catalyst==19.9.3
RUN pip install aiotg==1.0.0
#RUN pip install uuid

ENV PYTHONPATH=/src

WORKDIR /src

COPY ./wheat_rust_bot/requirements.txt .
RUN pip install -r requirements.txt

COPY ./model /model

COPY . .
RUN python setup.py install && rm -rf *

COPY wheat_rust_bot/ .
