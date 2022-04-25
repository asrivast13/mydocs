# Databricks notebook source
from __future__ import annotations
import os
import sys
import torch
import warnings
import logging
import argparse
from typing import Optional, Callable
from tqdm import tqdm
from transformers import (
    Wav2Vec2Processor, 
    AutoModelForCTC
)
from huggingsound.utils import get_chunks, get_waveforms, get_dataset_from_dict_list
from huggingsound.token_set import TokenSet
from huggingsound.normalizer import DefaultTextNormalizer
from huggingsound.speech_recognition.decoder import Decoder, GreedyDecoder
import collections
import contextlib
import wave
import glob
from traceback import print_tb
import json
from vosk import Model, KaldiRecognizer, SetLogLevel
import webrtcvad
import numpy as np

# COMMAND ----------

def read_wave(path):
    """Reads a .wav file.
    Takes the path, and returns (PCM audio data, sample rate).
    """
    with contextlib.closing(wave.open(path, 'rb')) as wf:
        num_channels = wf.getnchannels()
        assert num_channels == 1
        sample_width = wf.getsampwidth()
        assert sample_width == 2
        sample_rate = wf.getframerate()
        assert sample_rate in (8000, 16000, 32000, 48000)
        pcm_data = wf.readframes(wf.getnframes())
        return pcm_data, sample_rate

def write_wave(path, audio, sample_rate):
    """Writes a .wav file.
    Takes path, PCM audio data, and sample rate.
    """
    with contextlib.closing(wave.open(path, 'wb')) as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio)

class Frame(object):
    """Represents a "frame" of audio data."""
    def __init__(self, bytes, timestamp, duration, is_speech = True):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration
        self.isSpeech = is_speech

def __frame_generator_old(frame_duration_ms, audio, sample_rate, vad=None):
    """Generates audio frames from PCM audio data.
    Takes the desired frame duration in milliseconds, the PCM data, and
    the sample rate.
    Yields Frames of the requested duration.
    """
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    timestamp = 0.0
    duration = (float(n) / sample_rate) / 2.0
    while offset + n < len(audio):
        is_speech = vad.is_speech(audio[offset:offset + n], sample_rate) if vad is not None else False
        yield Frame(audio[offset:offset + n], timestamp, duration, is_speech)
        timestamp += duration
        offset += n
        
class Utterance(object):
    """Represents an utterance of speech audio data."""
    def __init__(self, audio, timestamp, duration, bytes):
        self.audio = audio
        self.timestamp = timestamp
        self.duration = duration
        self.bytes=bytes
        

class AudioSegmenter(object):
    """Represents the WebRTC based Audio Segmentation tool"""
    def __init__(self, 
                 vadAggressiveness=2, 
                 frameDurationMs=30, 
                 numFramesInWindow=100, 
                 samplingRate=16000,
                 numBytesPerSample=2):
        assert numBytesPerSample == 2  ## for now although the algo should work for byte encodings
        self.vad = webrtcvad.Vad(vadAggressiveness)
        self.sample_rate = samplingRate
        self.frame_duration_ms = frameDurationMs
        self.num_frames_in_window = numFramesInWindow
        self.SCALE_FACTOR = 1./float(1 << ((8 * numBytesPerSample)-1))
        self.num_bytes_per_sample = numBytesPerSample
        self.minAudioFileDurationInSecs = 0

    def set_min_file_size_for_segmentation(self, minAudioFileDurInSecsToSegment):
        if(minAudioFileDurInSecsToSegment > 0):
            self.minAudioFileDurationInSecs = minAudioFileDurInSecsToSegment

    def frame_generator(self, audio):
        """Generates audio frames from PCM audio data.
        Takes the desired frame duration in milliseconds, the PCM data, and
        the sample rate.
        Yields Frames of the requested duration.
        """
        n = int(self.sample_rate * (self.frame_duration_ms / 1000.0) * 2)
        offset = 0
        timestamp = 0.0
        duration = (float(n) / self.sample_rate) / 2.0
        while offset + n < len(audio):
            is_speech = self.vad.is_speech(audio[offset:offset + n], self.sample_rate)
            yield Frame(audio[offset:offset + n], timestamp, duration, is_speech)
            timestamp += duration
            offset += n

            
    def vad_collector(self, 
                      frames, 
                      triggerToggleFactor=0.9, 
                      minSpeechInUtteranceInSecs=0.5, 
                      utteranceRunoffDuration=5, 
                      maxUtteranceDuration=30, 
                      minSilenceAtEnds=0.06, 
                      verbosity=0):
        
        """Filters out non-voiced audio frames.
        Given a webrtcvad.Vad and a source of audio frames, yields only
        the voiced audio.
        Uses a padded, sliding window algorithm over the audio frames.
        When more than (triggerToggleFactor)X% (default X=90) of the frames in the window are voiced (as
        reported by the VAD), the collector triggers and begins yielding
        audio frames. Then the collector waits until X% of the frames in
        the window are unvoiced to detrigger.
        The window is padded at the front and back to provide a small
        amount of silence or the beginnings/endings of speech around the
        voiced frames.
        Arguments:
        sample_rate - The audio sample rate, in Hz.
        frame_duration_ms - The frame duration in milliseconds.
        padding_duration_ms - The amount to pad the window, in milliseconds.
        vad - An instance of webrtcvad.Vad.
        frames - a source of audio frames (sequence or generator).
        Returns: A generator that yields PCM audio data.
        """
        frame_duration_ms = self.frame_duration_ms
        num_padding_frames = self.num_frames_in_window
        sample_rate = self.sample_rate
        min_silence_frames_at_end = int(minSilenceAtEnds * 1000 / frame_duration_ms)
        if verbosity>2: 
            print('Min Silence Frames at End: %d' % min_silence_frames_at_end)
        
        # We use a deque for our sliding window/ring buffer.
        ring_buffer = collections.deque(maxlen=num_padding_frames)
        # We have two states: TRIGGERED and NOTTRIGGERED. We start in the
        # NOTTRIGGERED state.
        triggered = False
        start_time = -1
        end_time = -1

        ## smoothing on the frames
        numFrames = len(frames)
        for index, frame in enumerate(frames):
            if(index > 0) and (index < (numFrames-1)):
                if((frames[index].isSpeech != frames[index-1].isSpeech) 
                   and (frames[index-1].isSpeech == frames[index+1].isSpeech)):
                    frames[index].isSpeech = frames[index-1].isSpeech

        voiced_frames = []
        segid = 1
        num_silence_frames_at_end = 0
        for frame in frames:
            is_speech = frame.isSpeech

            if verbosity > 3:
                sys.stdout.write('1' if is_speech else '0')
                sys.stdout.write(' %.2f\n' % frame.timestamp)
            if not triggered:
                ring_buffer.append((frame, is_speech))
                num_voiced = len([f for f, speech in ring_buffer if speech])
                # If we're NOTTRIGGERED and more than 90% of the frames in
                # the ring buffer are voiced frames, then enter the
                # TRIGGERED state.
                if num_voiced > triggerToggleFactor * ring_buffer.maxlen:
                    triggered = True
                    if verbosity > 3:
                        sys.stdout.write('+(%s)\n' % (ring_buffer[0][0].timestamp,))
                    start_time = ring_buffer[0][0].timestamp
                    # We want to yield all the audio we see from now until
                    # we are NOTTRIGGERED, but we have to start with the
                    # audio that's already in the ring buffer.
                    for f, s in ring_buffer:
                        voiced_frames.append(f)
                    ring_buffer.clear()
                    index = len(voiced_frames)-1
                    while(index>=0):
                        if(voiced_frames[index].isSpeech):
                            break
                        else:
                            num_silence_frames_at_end += 1

            else:
                # We're in the TRIGGERED state, so collect the audio data
                # and add it to the ring buffer.
                voiced_frames.append(frame)
                ring_buffer.append((frame, is_speech))
                num_unvoiced = len([f for f, speech in ring_buffer if not speech])
                if not is_speech:
                    num_silence_frames_at_end += 1
                numSpeechFramesInUtterance = sum([int(f.isSpeech) for f in voiced_frames])
                end_time = frame.timestamp + frame.duration
                # If more than 90% of the frames in the ring buffer are
                # unvoiced, then enter NOTTRIGGERED and yield whatever
                # audio we've collected.
                if((numSpeechFramesInUtterance * frame_duration_ms / 1000) < minSpeechInUtteranceInSecs):
                    continue
                    
                if ((num_unvoiced > (triggerToggleFactor * ring_buffer.maxlen)) or 
                (((end_time - start_time) > utteranceRunoffDuration) and (num_silence_frames_at_end >= min_silence_frames_at_end)) or
                   ((end_time - start_time) > maxUtteranceDuration)
                   ):
                    if verbosity > 3:
                        sys.stdout.write('-(%s)\n' % (frame.timestamp + frame.duration))
                    end_time = frame.timestamp + frame.duration
                    #start_time -= minSilenceAtEnds if start_time >= minSilenceAtEnds else 0.0
                    triggered = False
                    databytes = b''.join([f.bytes for f in voiced_frames])
                    audiosamples = np.frombuffer(databytes, dtype=np.int16).astype(np.float32)
                    audiosamples *= self.SCALE_FACTOR
                    if verbosity > 1:
                        print('Segment %d: start=%.2f end=%.2f bytes=%d duration=%.2f\n' % (segid, start_time, end_time, len(databytes), len(audiosamples)/sample_rate))
                    segid += 1
                    #databytes = None
                    yield(Utterance(audiosamples, start_time, (end_time - start_time), databytes))
                    start_time = -1
                    end_time = -1
                    ring_buffer.clear()
                    voiced_frames = []
                    num_silence_frames_at_end = 0
        if verbosity > 3:
            if triggered:
                sys.stdout.write('-(%s)\n' % (frame.timestamp + frame.duration))
            sys.stdout.write('\n')
        # If we have any leftover voiced audio when we run out of input,
        # yield it.
        end_time = frame.timestamp + frame.duration
        if voiced_frames:
            databytes = b''.join([f.bytes for f in voiced_frames])
            audiosamples = np.frombuffer(databytes, dtype=np.int16).astype(np.float32)
            audiosamples *= self.SCALE_FACTOR
            if verbosity > 1:
                print('Segment %d: start=%.2f end=%.2f bytes=%d duration=%.2f\n' % (segid, start_time, end_time, len(databytes), len(audiosamples)/sample_rate))
            #databytes = None
            yield(Utterance(audiosamples, start_time, (end_time - start_time), databytes))

    def process(self, audio, 
                triggerToggleFactor=0.9, 
                minSpeechInUtteranceInSecs=0.5, 
                utteranceRunoffDuration=5, 
                maxUtteranceDuration=30, 
                minSilenceAtEnds=0.06):
        
        frames = self.frame_generator(audio)
        frames = list(frames)
        
        totalAudioDuration = len(audio) / (self.sample_rate * self.num_bytes_per_sample)
        segmentsList = []
        
        if(totalAudioDuration < self.minAudioFileDurationInSecs):
            databytes = b''.join([f.bytes for f in frames])
            audiosamples = np.frombuffer(databytes, dtype=np.int16).astype(np.float32)
            audiosamples *= self.SCALE_FACTOR
            segid = 0
            start_time = 0.0
            end_time = totalAudioDuration
            print('Segment from File %d: start=%.2f end=%.2f bytes=%d duration=%.2f\n' % (segid, start_time, end_time, len(databytes), len(audiosamples)/sample_rate))
            segment = Utterance(audiosamples, start_time, (end_time - start_time), None)
            segmentsList.append(segment)
        else:
            totalSpeechDuration = 0.0
            totalNumSegments = 0
            maxDuration = 0.0
            speech_segments = self.vad_collector(frames, triggerToggleFactor, minSpeechInUtteranceInSecs, utteranceRunoffDuration, maxUtteranceDuration, minSilenceAtEnds)
            for segment in speech_segments:
                if segment.duration > maxDuration:
                    maxDuration = segment.duration
                totalSpeechDuration += segment.duration
                totalNumSegments += 1
                segmentsList.append(segment)

            print('Segmenter found total of %d segments with duration %.2fsecs from audio of duration %.2fsecs with max=%.2f -- compression factor: %.2f%%' 
                      % (totalNumSegments, totalSpeechDuration, totalAudioDuration, maxDuration, (100.0*totalSpeechDuration/totalAudioDuration)))
        return segmentsList


# In[ ]:


# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger.setLevel(logging.INFO)

class SpeechRecognitionModel2():
    """
    Speech Recognition Model.

    Parameters
    ----------
    model_path : str
        The path to the model or the model identifier from huggingface.co/models.
    
    device: Optional[str] = "cpu"
        Device to use for inference/evaluation/training, default is "cpu". If you want to use a GPU for that, 
        you'll probably need to specify the device as "cuda"
    """

    def __init__(self, model_path: str, device: Optional[str] = "cpu"):
        
        self.model_path = model_path
        self.device = device
        
        logger.info("Loading model...")
        self._load_model()

    @property
    def is_finetuned(self):
        return self.processor is not None

    def _load_model(self):

        self.model = AutoModelForCTC.from_pretrained(self.model_path)
        self.model.to(self.device)

        try:
            self.processor = Wav2Vec2Processor.from_pretrained(self.model_path)
            self.token_set = TokenSet.from_processor(self.processor)
        except Exception:
            logger.warning("Not fine-tuned model! You'll need to fine-tune it before use this model for audio transcription")
            self.processor = None
            self.token_set = None

    def transcribeFiles(self, paths: list[str], batch_size: Optional[int] = 1, decoder: Optional[Decoder] = None) -> list[dict]:
        """ 
        Transcribe audio files.

        Parameters:
        ----------
            paths: list[str]
                List of paths to audio files to transcribe

            batch_size: Optional[int] = 1
                Batch size to use for inference

            decoder: Optional[Decoder] = None
                Decoder to use for transcription. If you don't specify this, the engine will use the GreedyDecoder.

        Returns:
        ----------
            list[dict]:
                A list of dictionaries containing the transcription for each audio file:

                [{
                    "transcription": str,
                    "start_timesteps": list[int],
                    "end_timesteps": list[int],
                    "probabilities": list[float]
                }, ...]
        """

        if not self.is_finetuned:
            raise ValueError("Not fine-tuned model! Please, fine-tune the model first.")
        
        if decoder is None:
            decoder = GreedyDecoder(self.token_set)

        sampling_rate = self.processor.feature_extractor.sampling_rate
        result = []

        for paths_batch in tqdm(list(get_chunks(paths, batch_size))):

            waveforms = get_waveforms(paths_batch, sampling_rate)

            inputs = self.processor(waveforms, sampling_rate=sampling_rate, return_tensors="pt", padding=True, do_normalize=True)

            with torch.no_grad():
                logits = self.model(inputs.input_values.to(self.device), attention_mask=inputs.attention_mask.to(self.device)).logits

            result += decoder(logits)

        return result
    
    def transcribeAudio(self, utterances: list[Utterance], batch_size: Optional[int] = 1, decoder: Optional[Decoder] = None) -> list[dict]:
        """ 
        Transcribe audio files.

        Parameters:
        ----------
            paths: list[Utterance]
                List of audio utterances to transcribe

            batch_size: Optional[int] = 1
                Batch size to use for inference

            decoder: Optional[Decoder] = None
                Decoder to use for transcription. If you don't specify this, the engine will use the GreedyDecoder.

        Returns:
        ----------
            list[dict]:
                A list of dictionaries containing the transcription for each audio file:

                [{
                    "transcription": str,
                    "start_timesteps": list[int],
                    "end_timesteps": list[int],
                    "probabilities": list[float]
                }, ...]
        """

        if not self.is_finetuned:
            raise ValueError("Not fine-tuned model! Please, fine-tune the model first.")

        if decoder is None:
            decoder = GreedyDecoder(self.token_set)

        sampling_rate = self.processor.feature_extractor.sampling_rate
        result = []

        for utts_batch in tqdm(list(get_chunks(utterances, batch_size))):

            #waveforms = get_waveforms(paths_batch, sampling_rate)
            waveforms = []
            for utt in utts_batch:
                waveforms.append(utt.audio)

            inputs = self.processor(waveforms, sampling_rate=sampling_rate, return_tensors="pt", padding=True, do_normalize=True)

            with torch.no_grad():
                logits = self.model(inputs.input_values.to(self.device), attention_mask=inputs.attention_mask.to(self.device)).logits

            batchResults = decoder(logits)
            for index, br in enumerate(batchResults):
                if(len(br['transcription'].strip()) == 0):
                    continue;
                br['utterance_start']    = '%.2f' % utts_batch[index].timestamp
                br['utterance_duration'] = '%.2f' % utts_batch[index].duration
                br['tokens'] = self.convertToCTM(br)
                result.append(br)

        return result
    
    def convertToCTM(self, utterance: dict, extrapolate: Optional[bool] = False) -> dict:
        transcript  = utterance["transcription"]
        uttstart    = float(utterance['utterance_start'])
        duration    = float(utterance['utterance_duration'])
        char_starts = utterance["start_timestamps"]
        char_ends   = utterance["end_timestamps"]
        tokens = []

        if((char_starts is None) 
           or (char_ends is None) 
           or (len(char_starts) == 0) 
           or (len(char_ends) == 0) or 
           (len(transcript) != len(char_starts)) 
           or (len(char_starts) != len(char_ends)) 
           or (char_ends[-1] > (duration * 1000))):
            extrapolate = True

        if(extrapolate):
            words = transcript.split()
            if(len(words) == 0):
                return tokens
            numChars = len(transcript)
            start = uttstart
            start += 0.1
            dur = 0.0
            for word in words:
                dur = duration * len(word) / numChars
                #ctmline = "%s 1 %.2f %.2f %s" % (sessionId, start, dur, word)
                ctm = {"baseform": word, "start": start, "duration": dur}
                start += dur
                tokens.append(ctm)
        else:
            word = ""
            start = -1
            dur = 0.0
            for index, char in enumerate(transcript):
                if(char == " "):
                    if(len(word)>0):
                        #ctmline = "%s 1 %.2f %.2f %s" % (sessionId, (uttstart + start/1000), dur/1000, word)
                        wst = (uttstart + start/1000)
                        wd  = dur/1000
                        ctm = {"baseform": word, "start": wst, "duration": wd}
                        tokens.append(ctm)
                        word = ""
                        start = -1
                        dur = 0.0
                else:
                    if(len(word)==0):
                        start = char_starts[index]
                    word += char
                    dur = char_ends[index] - start

            if(len(word)>0):
                #ctmline = "%s 1 %.2f %.2f %s" % (sessionId, (uttstart + start/1000), dur/1000, word)
                wst = (uttstart + start/1000)
                wd  = dur/1000
                ctm = {"baseform": word, "start": wst, "duration": wd}
                tokens.append(ctm)
                word = ""
                start = -1
                dur = 0.0

        return tokens

class VoskWord:
    ''' A class representing a word from the JSON format for vosk speech recognition API '''

    def __init__(self, dict):
        '''
        Parameters:
          dict (dict) dictionary from JSON, containing:
            conf (float): degree of confidence, from 0 to 1
            end (float): end time of the pronouncing the word, in seconds
            start (float): start time of the pronouncing the word, in seconds
            word (str): recognized word
        '''

        self.conf = dict["conf"]
        self.end = dict["end"]
        self.start = dict["start"]
        self.word = dict["word"]

    def to_string(self, tag):
        ''' Returns a string describing this instance '''
        #return "{:20} from {:.2f} sec to {:.2f} sec, confidence is {:.2f}%".format(
        #    self.word, self.start, self.end, self.conf*100)
        return("%s \t 1 \t %.2f \t %.2f \t %-15s \t %.2f" % (tag, self.start, (self.end-self.start), self.word, self.conf))


# COMMAND ----------

InputFileOrFolder = '/dbfs/FileStore/audio/Spanish/Spanish_Conversational_Speech_Corpus/WAV/'
HSSttModelFolder  = '/dbfs/FileStore/models/wav2vec2-large-xlsr-53-spanish/'
VoskModelFolder   = '/dbfs/FileStore/models/vosk-model-small-es-0.22'
outFolder         = '/dbfs/FileStore/output/Spanish/Spanish_Conversational_Speech_Corpus/CTM'
sampleRate = 16000
useGPUifAvailable = True
useLM = False
useSegmentsInVosk = True

##define constants
aggressiveness = 3
frameDurationMs = 30
numFramesInWindow = 20

## initialize segmenter
print('Initializing Audio Segmenter with aggressiveness=%d frame length=%dmsec window size=%dframes and sample rate=%d\n'
     % (aggressiveness, frameDurationMs, numFramesInWindow, sampleRate))
segmenter = AudioSegmenter(aggressiveness, frameDurationMs, numFramesInWindow, sampleRate, 2)

## initialize STT model including LM if required
device = "cuda" if (useGPUifAvailable and torch.cuda.is_available()) else "cpu"
print('device = %s' % device)
print('Loading Wav2Vec2 speech recognition model from folder %s\n' % HSSttModelFolder)
HSmodel = None
decoder = None

try:
    HSmodel = SpeechRecognitionModel2(HSSttModelFolder, device=device)
    if useLM == True:
        from huggingsound import ParlanceLMDecoder
        LmModelFolder = HSSttModelFolder + "/language_model/"
        lm_path = LmModelFolder + "lm.binary"
        unigrams_path = LmModelFolder + "unigrams.txt"
        # To use this decoder you'll need to install the Parlance's ctcdecode first (https://github.com/parlance/ctcdecode)
        print('Starting to load LM file %s in ParlanceLMDecoder ...' % lm_path)
        decoder = ParlanceLMDecoder(HSmodel.token_set, lm_path=lm_path, alpha=2, beta=1, beam_width=100)    
        #decoder = KenshoLMDecoder(model.token_set, lm_path=lm_path, unigrams_path=unigrams_path, alpha=2, beta=1, beam_width=100)
        print("Finished loading Language Model")
except Exception as e:
    print('Could not load acoustic model. Failed with message: %s' % e)
    sys.exit(-1)

print('Loading Vosk Model from folder: %s' % VoskModelFolder)
VoskModel = Model(VoskModelFolder)
rec = KaldiRecognizer(VoskModel, sampleRate)
rec.SetWords(True)

# COMMAND ----------

!ls -R /dbfs/FileStore/output/Spanish/Spanish_Conversational_Speech_Corpus/CTM/
!rm -rf /dbfs/FileStore/output/Spanish/Spanish_Conversational_Speech_Corpus/CTM/
!mkdir /dbfs/FileStore/output/Spanish/Spanish_Conversational_Speech_Corpus/CTM/

# COMMAND ----------

HSOutFolder = outFolder + "/wav2vec2"
VoskOutFolder = outFolder + "/vosk"
if not os.path.exists(HSOutFolder):
    os.mkdir(HSOutFolder)
if not os.path.exists(VoskOutFolder):
    os.mkdir(VoskOutFolder)
    
audioFilesList = []
if(os.path.isfile(InputFileOrFolder)):
    audioFilesList.append(InputFileOrFolder)
elif (os.path.isdir(InputFileOrFolder)):
    audioFilesList = glob.glob(InputFileOrFolder+'/*.wav')
else:
    raise argparse.ArgumentTypeError('Invalid input: '+InputFileOrFolder)

for audioFile in audioFilesList:
    #read audio file into buffer
    audio, sample_rate = read_wave(audioFile)
    assert sample_rate == sampleRate

    sessionId = os.path.basename(audioFile).split('.')[0]
    print('Processing (%s %dHz %.2fsecs) in file %s' % (sessionId, sample_rate, (len(audio)/(sample_rate * 2)), audioFile))

    ## Run segmenter with following parameters:
    # threshold for speech/non-speech in buffer to trigger or not = 75%
    # minimum speech length in utterance = 0.5 secs (below this the segmenter will not break)
    # utterance runoff duration = 5 secs (above this the segmenter starts looking for X non-speech frames to break)
    # max duration length = 15 secs (hard-limit ... might cause a break within a word)
    # minimum number of non-speech frames needed at the end of the utterance to break (X) = 1 
    segments = segmenter.process(audio, 0.75, 0.5, 5, 15, frameDurationMs/1000)
    
    ### This part runs the HS Decoder and saves the output
    # Now run the STT decoder on all the segments sequentially (could this be parallelized?)
    transcripts = HSmodel.transcribeAudio(segments, 1, decoder)
    
    ## Write CTM lines into the output file
    outCTMFile = '%s/%s.ctm' % (HSOutFolder, sessionId)
    outJSONFile = '%s/%s.json' % (HSOutFolder, sessionId)
    print('Writing %d transcripts into file %s' % (len(transcripts), outCTMFile))
    with open(outJSONFile, "w") as ofp:
      ofp.write(json.dumps(transcripts, indent=4))
      
    with open(outCTMFile, 'w') as ofp:
        for transcript in transcripts:
            #print('\t%s (%s-%s-%s)\n' % (transcript['transcription'], sessionId, transcript['utterance_start'], transcript['utterance_duration']))
            for token in transcript["tokens"]:
                ofp.write("%s \t 1 \t %.2f \t %.2f \t %s\n" % (sessionId, token["start"], token["duration"], token["baseform"]))
    print(' ')
    
    # get the list of JSON dictionaries
    results = []
    if(useSegmentsInVosk):
        for segment in tqdm(segments):
            if(len(segment.bytes) == 0):
                continue;
            if(rec.AcceptWaveform(segment.bytes)):
                part_result = json.loads(rec.Result())
                results.append(part_result)
            part_result = json.loads(rec.FinalResult())
            if(part_result):
                results.append(part_result)
    else:
        # recognize speech using vosk model in streaming mode
        wf = wave.open(audioFile, "rb")
        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            if rec.AcceptWaveform(data):
                part_result = json.loads(rec.Result())
                results.append(part_result)
        part_result = json.loads(rec.FinalResult())
        results.append(part_result)
        wf.close()  # close audiofile

    rec.Reset()
    # convert list of JSON dictionaries to list of 'Word' objects
    ## Write CTM lines into the output file
    print('Writing Vosk Decoder output to files %s and %s' % (outCTMFile, outJSONFile))
    outCTMFile = '%s/%s.ctm' % (VoskOutFolder, sessionId)
    outJSONFile = '%s/%s.json' % (VoskOutFolder, sessionId)
    with open(outJSONFile, "w") as ofp:
      ofp.write(json.dumps(results, indent=4))

    list_of_Words = []
    for sentence in results:
        if len(sentence) == 1:
            # sometimes there are bugs in recognition 
            # and it returns an empty dictionary
            # {'text': ''}
            continue
        #print(json.dumps(sentence, indent=4))
        #print("%s  %.2f  %.2f  %s" % (sessionId, sentence['result'][0]["start"], sentence['result'][-1]["end"], sentence["text"]))
        for obj in sentence['result']:
            w = VoskWord(obj)  # create custom Word object
            list_of_Words.append(w)  # and add it to list

    # output to the screen
    with open(outCTMFile, "w") as ofp:
        for word in list_of_Words:
            ofp.write("%s\n" % word.to_string(sessionId))



# COMMAND ----------


