#!/bin/bash

apt-get update && apt install -y git unzip git-lfs ffmpeg sox libsoxr-dev
/databricks/python3/bin/pip install --upgrade pip
/databricks/python3/bin/pip install huggingsound pyctcdecode
/databricks/python3/bin/pip install https://github.com/kpu/kenlm/archive/master.zip
git clone --recursive https://github.com/parlance/ctcdecode.git
cd ctcdecode && /databricks/python3/bin/pip install .
cd ..
rm -rf ctcdecode
/databricks/python3/bin/pip install webrtcvad
/databricks/python3/bin/pip install vosk
exit 0
