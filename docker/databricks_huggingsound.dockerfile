# huggingsound
#
# VERSION               0.0.3
FROM databricksruntime/gpu-conda:cuda11
MAINTAINER Amit Srivastava <amit.srivastava@talkdesk.com>
LABEL tips="to enable GPU use the command: \
            docker run --gpus=all <image>"

### Set-up main packages
RUN apt-get update
RUN apt install -y build-essential
RUN apt install -y git
RUN apt install -y unzip
RUN apt install -y git-lfs
RUN apt install -y ffmpeg

## Install Anaconda Python
RUN pip install --upgrade pip
RUN pip install -U huggingsound pyctcdecode
RUN pip install https://github.com/kpu/kenlm/archive/master.zip
