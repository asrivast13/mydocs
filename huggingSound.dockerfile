# huggingsound
#
# VERSION               0.0.1
FROM ubuntu:18.04
MAINTAINER Amit Srivastava <amit.srivastava@talkdesk.com>
LABEL tips="to enable GPU use the command: \
            docker run --gpus=all <image>"

### Set-up main packages
RUN apt-get update && \
    apt install -y \
        wget \
        pkg-config \
        build-essential \
        git \
        unzip \
        git-lfs \
        ffmpeg

## Install Anaconda Python
#RUN wget --no-check-certificate https://repo.anaconda.com/archive/Anaconda3-2021.11-Linux-x86_64.sh
RUN wget https://repo.anaconda.com/archive/Anaconda3-2021.11-Linux-x86_64.sh
RUN bash Anaconda3-2021.11-Linux-x86_64.sh -b -f
RUN rm -f Anaconda3-2021.11-Linux-x86_64.sh
ENV PATH /root/anaconda3/bin:$PATH
#RUN easy_install pip
RUN pip install --upgrade pip
#RUN conda install pytorch
RUN pip install -U huggingsound pyctcdecode
RUN pip install https://github.com/kpu/kenlm/archive/master.zip

#Now copy the code and models into the container
RUN apt-get install -y ca-certificates
RUN cd /root && \
    git lfs clone https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-spanish && \
    mv wav2vec2-large-xlsr-53-spanish/ Models/
