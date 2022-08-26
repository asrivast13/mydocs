import os
import sys
import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
from collections import deque
from speechbrain.pretrained import VAD
import logging
import matplotlib.pyplot as plt

logging.basicConfig(
        format='%(asctime)s.%(msecs)03d %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S')

TMPDIR = '.'
SHOWPLOTS = True
MAXPLOTDURATION = 30 #secs

myargs = deque(sys.argv)
progName = myargs.popleft()

#print(progName)
#for file in myargs:
#    print(file)

assert len(myargs) == 1
audioFile = myargs.pop()

logging.info('Creating sampler object')
resampler = T.Resample(
    8000,
    16000,
    lowpass_filter_width=16,
    rolloff=0.85,
    resampling_method='kaiser_window',
    dtype=torch.float32,
    beta=8.555504641634386
)

logging.info('Initializing VAD object from pretrained models ...')
VAD = VAD.from_hparams(source="speechbrain/vad-crdnn-libriparty", savedir="/tmp/pretrained_models/vad-crdnn-libriparty")

logging.info('loading stereo audio file %s into memory' % audioFile)
audioBase = os.path.basename(audioFile).split('.')[0]
signal, samplingRate = torchaudio.load(audioFile)
assert signal.shape[0] == 2

logging.info('splitting stereo audio into separate channels')
channels = torch.tensor_split(signal, 2, dim=0)
left = channels[0]
right = channels[1]
assert left.dtype == torch.float32

tmpAudioBase = os.path.join(TMPDIR, audioBase)

tmpAudioFile = tmpAudioBase + "_left.wav"
logging.info('resampling to 16K and writing left channel to tmp file %s' % tmpAudioFile)
left16k = resampler(left)
torchaudio.save(tmpAudioFile, left16k, 16000, encoding="PCM_S", bits_per_sample=16)

logging.info('running VAD on left channel')
#leftseg = VAD.get_speech_segments(tmpAudioFile, speech_th=0.3)
leftseg = VAD.get_speech_segments(tmpAudioFile, 
                                small_chunk_size=5, 
                                apply_energy_VAD=True,
                                double_check=True,
                                activation_th=0.25, 
                                deactivation_th=0.1,
                                en_activation_th=0.25,
                                en_deactivation_th=0.05,
                                speech_th=0.4)

logging.info("Left Segments: ")
VAD.save_boundaries(leftseg)

if SHOWPLOTS:
    left16k1ch = left16k.squeeze()
    fs = 16000
    time = torch.linspace(0, left16k1ch.shape[0]/fs, steps=left16k1ch.shape[0])
    upsampled_left = VAD.upsample_boundaries(leftseg, tmpAudioFile)

    #logging.info(time.shape)
    #logging.info(left16k1ch.shape)
    #logging.info(upsampled_left.squeeze().shape)

    prob_chunks = VAD.get_speech_prob_file(tmpAudioFile)
    logging.info(prob_chunks.shape)
    plt.plot(prob_chunks.squeeze())
    #plt.show()
    plt.savefig('leftscores.jpg')

    plt.clf()
    plt.cla() 
    plt.close()

    N = fs * MAXPLOTDURATION
    x = time
    y = left16k1ch
    z = upsampled_left.squeeze()

    if left16k1ch.shape[0] > N:
        x = time[0:N]
        y = left16k1ch[0:N]
        z = upsampled_left.squeeze()[0:N]
    
    plt.plot(x, y)
    plt.plot(x, z)
    plt.savefig('leftout.jpg')
    plt.show()

os.remove(tmpAudioFile)

tmpAudioFile = tmpAudioBase + "_right.wav"
logging.info('resampling to 16K and writing right channel to tmp file %s' % tmpAudioFile)
torchaudio.save(tmpAudioFile, resampler(right), 16000, encoding="PCM_S", bits_per_sample=16)

logging.info('running VAD on right channel')
rightseg = VAD.get_speech_segments(tmpAudioFile)

logging.info("Right Segments: ")
VAD.save_boundaries(rightseg)
os.remove(tmpAudioFile)

logging.info('Done')







