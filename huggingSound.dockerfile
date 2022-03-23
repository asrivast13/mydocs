# huggingsound
#
# VERSION               0.0.1
FROM ubuntu:18.04
MAINTAINER Amit Srivastava <amit.srivastava@talkdesk.com>

### Set-up main packages
RUN apt-get update
RUN apt install -y wget pkg-config build-essential git aptitude bc
RUN apt install -y unzip git-lfs sox libsoxr-dev libsox-fmt-all ffmpeg

## Install Anaconda Python
RUN wget https://repo.anaconda.com/archive/Anaconda3-2021.11-Linux-x86_64.sh
RUN bash Anaconda3-2021.11-Linux-x86_64.sh -b -f
RUN rm -f Anaconda3-2021.11-Linux-x86_64.sh
ENV PATH /root/anaconda3/bin:$PATH
#RUN easy_install pip
RUN pip install --upgrade pip
#RUN conda install pytorch
RUN pip install -U huggingsound pyctcdecode
RUN pip install https://github.com/kpu/kenlm/archive/master.zip
