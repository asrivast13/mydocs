# huggingsound
#
# VERSION               0.0.2
FROM ubuntu:18.04
MAINTAINER Amit Srivastava <amit.srivastava@talkdesk.com>
LABEL tips="to enable GPU use the command: \
            docker run --gpus=all <image>"

### Set-up main packages
RUN apt-get update
RUN apt install -y wget
RUN apt install -y pkg-config
RUN apt install -y build-essential
RUN apt install -y git
RUN apt install -y unzip
RUN apt install -y git-lfs
RUN apt install -y ffmpeg

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
