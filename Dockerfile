FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04

ENV PROJECT_DIR /mlflow/projects
ENV CODE_DIR /mlflow/projects/code
WORKDIR ${PROJECT_DIR}

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update && \
    apt -y install git software-properties-common wget curl build-essential libblas-dev liblapack-dev libhdf5-dev && \
    add-apt-repository -y ppa:deadsnakes/ppa

RUN apt update && \
    apt -y install python3.9 python3.9-dev python3.9-distutils && \
    wget https://bootstrap.pypa.io/get-pip.py && \
    python3.9 get-pip.py && \
    ln -s /usr/bin/python3.9 /usr/bin/python

ADD constraints.txt ${PROJECT_DIR}/
ADD requirements.txt ${PROJECT_DIR}/
RUN pip install --no-cache-dir -r requirements.txt -c constraints.txt

WORKDIR ${CODE_DIR}
