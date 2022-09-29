# Databricks notebook source
from __future__ import annotations
import os
import sys
import warnings
import logging
import argparse
import collections
import contextlib
import wave
import glob
import webrtcvad
import numpy as np

VERBOSITY = 0

# COMMAND ----------
# Setup logging
#logger = logging.getLogger(__name__)
#logging.basicConfig(
#    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
#    datefmt="%m/%d/%Y %H:%M:%S",
#    handlers=[logging.StreamHandler(sys.stdout)],
#)
#logger.setLevel(logging.INFO)

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
                      minSilenceAtEnds=0.06):

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

        if VERBOSITY>2:
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

            if VERBOSITY > 3:
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
                    if VERBOSITY > 3:
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
                # If more than X% of the frames in the ring buffer are
                # unvoiced, then enter NOTTRIGGERED and yield whatever
                # audio we've collected.
                #if((numSpeechFramesInUtterance * frame_duration_ms / 1000) < minSpeechInUtteranceInSecs):
                #    continue

                if VERBOSITY > 5:
                    cond1 = num_unvoiced / ring_buffer.maxlen
                    cond2 = (end_time - start_time) / utteranceRunoffDuration
                    cond3 = num_silence_frames_at_end / min_silence_frames_at_end
                    cond4 = (end_time - start_time) / maxUtteranceDuration
                    sys.stdout.write(">>>%.2f %.2f %.2f %.2f\n" % (cond1, cond2, cond3, cond4))

                if ((num_unvoiced > (triggerToggleFactor * ring_buffer.maxlen)) or
                (((end_time - start_time) > utteranceRunoffDuration) and (num_silence_frames_at_end >= min_silence_frames_at_end)) or
                   ((end_time - start_time) > maxUtteranceDuration)
                   ):
                    if VERBOSITY > 3:
                        sys.stdout.write('-(%s)\n' % (frame.timestamp + frame.duration))
                    end_time = frame.timestamp + frame.duration
                    #start_time -= minSilenceAtEnds if start_time >= minSilenceAtEnds else 0.0
                    triggered = False
                    databytes = b''.join([f.bytes for f in voiced_frames])
                    audiosamples = np.frombuffer(databytes, dtype=np.int16).astype(np.float32)
                    audiosamples *= self.SCALE_FACTOR
                    if VERBOSITY > 4:
                        print('Segment %d: start=%.2f end=%.2f bytes=%d duration=%.2f' % (segid, start_time, end_time, len(databytes), len(audiosamples)/sample_rate))
                    segid += 1
                    #databytes = None
                    yield(Utterance(audiosamples, start_time, (end_time - start_time), databytes))
                    start_time = -1
                    end_time = -1
                    ring_buffer.clear()
                    voiced_frames = []
                    num_silence_frames_at_end = 0
        if VERBOSITY > 3:
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
            if VERBOSITY > 4:
                print('Segment %d: start=%.2f end=%.2f bytes=%d duration=%.2f' % (segid, start_time, end_time, len(databytes), len(audiosamples)/sample_rate))
            #databytes = None
            yield(Utterance(audiosamples, start_time, (end_time - start_time), databytes))

    def process(self, audio,
                triggerToggleFactor=0.9,
                minSegmentDuration=0,
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
            segment = Utterance(audiosamples, start_time, (end_time - start_time), databytes)
            segmentsList.append(segment)
        else:
            totalSpeechDuration = 0.0
            totalNumSegments = 0
            maxDuration = 0.0
            speech_segments = self.vad_collector(frames, triggerToggleFactor, 0, utteranceRunoffDuration, maxUtteranceDuration, minSilenceAtEnds)
            for segment in speech_segments:
                #if len(segmentsList) == 0:
                #    print("Type(audio): %s, Type(bytes): %s" % (type(segment.audio), type(segment.bytes)))

                if minSegmentDuration > 0:
                    if len(segmentsList) > 0:
                        prevLen = segmentsList[-1].duration
                        prevEnd = segmentsList[-1].duration + segmentsList[-1].timestamp
                        currLen = segment.duration
                        currStart = segment.timestamp
                        sumLen = currLen + prevLen

                        #print("%s %s %s" % ((prevLen < minSegmentDuration), ((currStart - prevEnd) < minSilenceAtEnds), (sumLen < maxUtteranceDuration)))

                        if (prevLen < minSegmentDuration) and ((currStart - prevEnd) <= minSilenceAtEnds) and (sumLen < maxUtteranceDuration):
                            segmentsList[-1].duration += currLen
                            segmentsList[-1].bytes += segment.bytes
                            segmentsList[-1].audio = np.concatenate([segmentsList[-1].audio, segment.audio])
                            #print("MERGED: <%.2f> %.2f %.2f <%.2f> %.2f <%.2f>" % (prevLen, prevEnd, currStart, currStart-prevEnd, currLen, sumLen))
                        else:
                            segmentsList.append(segment)
                            #print("SKIPPED: <%.2f> %.2f %.2f <%.2f> %.2f <%.2f>" % (prevLen, prevEnd, currStart, currStart-prevEnd, currLen, sumLen))
                    else:
                        segmentsList.append(segment)
                        #print("FIRST: %.2f %.2f %.2f" % (segment.timestamp, segment.duration, segment.timestamp+segment.duration))
                        #print("minSegmentDuration=%.2f minSilenceAtEnds=%.2f maxUtteranceDuration=%.2f" % (minSegmentDuration, minSilenceAtEnds, maxUtteranceDuration))
                else:
                    segmentsList.append(segment)

        #print("\n")
        for index, segment in enumerate(segmentsList):
            print('Segment %d: start=%.2f end=%.2f span=%.2f bytes=%d duration=%.2f' % (index, segment.timestamp, segment.timestamp + segment.duration, segment.duration, len(segment.bytes), len(segment.audio)/sample_rate))
            if segment.duration > maxDuration:
                maxDuration = segment.duration
            totalSpeechDuration += segment.duration
            totalNumSegments += 1

        print('Segmenter found total of %d segments with duration %.2fsecs from audio of duration %.2fsecs with max=%.2f -- compression factor: %.2f%%'
                        % (totalNumSegments, totalSpeechDuration, totalAudioDuration, maxDuration, (100.0*totalSpeechDuration/totalAudioDuration)))
        return segmentsList

#__MAIN__
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inpath", "-i", type=str, required=True, help="Input Path"
    )
    parser.add_argument(
        "--outpath", "-o", type=str, required=False, help="Output Path"
    )
    parser.add_argument(
        "-v", "--verbosity", type=int, required=False, default=0, help="Verbosity Level ([0]: basic 1: more, 2: high, >=3: debug)"
    )
    parser.add_argument(
        "--vadpower", "-a", type=int, required=False, default=2, help="Aggressiveness of VAD (1: low, [2]: medium, 3: high)"
    )
    parser.add_argument(
        "--framelen", "-l", type=int, required=False, default=30, help="Length of each frame of Audio in msecs [30]"
    )
    parser.add_argument(
        "--winlen", "-w", type=int, required=False, default=20, help="Number of frames in each window for segmentation [20]"
    )
    parser.add_argument(
        "--trfactor", "-f", type=float, required=False, default=0.9, help="threshold for speech/non-speech in buffer to trigger or not; value between 0 and 1 [0.9]"
    )
    parser.add_argument(
        "--runoffdur", "-r", type=float, required=False, default=5.0, help="segment runoff duration in secs; above this the segmenter starts looking for X non-speech frames to break [5.0]"
    )
    parser.add_argument(
        "--maxsegdur", "-m", type=float, required=False, default=30, help="maximum segment duration in secs [30.0]"
    )
    parser.add_argument(
        "--minsildur", "-p", type=float, required=False, default=0.06, help="minimum silence duration in secs at the end of the segment [0.06]"
    )
    parser.add_argument(
        "--minsegdur", "-d", type=float, required=False, default=0, help="minimum segment duration in secs; positive value will trigger second pass [0.0]"
    )

    args = parser.parse_args()

    InputFileOrFolder = args.inpath
    outFolder = "./"
    if args.outpath and os.path.isdir(args.outpath):
        outFolder = args.outpath

    sampleRate = 16000

    ##define constants
    aggressiveness = args.vadpower
    frameDurationMs = args.framelen
    numFramesInWindow = args.winlen
    VERBOSITY = args.verbosity

    print(args)
    #sys.exit(0)

    segmenter = None
    ## initialize segmenter
    ## sample rate is assumed to be 16000 and bytes-per-sample is always assumed to be 2 because input is assumed to be WAV
    ## this assumption can be changed later
    print('Initializing Audio Segmenter with aggressiveness=%d frame length=%dmsec window size=%dframes and sample rate=%d\n'
          % (aggressiveness, frameDurationMs, numFramesInWindow, sampleRate))
    segmenter = AudioSegmenter(aggressiveness, frameDurationMs, numFramesInWindow, sampleRate, 2)

    ## Identify list of audio files to run on
    audioFilesList = []
    if(os.path.isfile(InputFileOrFolder)):
        audioFilesList.append(InputFileOrFolder)
    elif (os.path.isdir(InputFileOrFolder)):
        audioFilesList = glob.glob(InputFileOrFolder+'/*.wav')
    else:
        raise argparse.ArgumentTypeError('Invalid input: '+InputFileOrFolder)

    if(len(audioFilesList) == 0):
        raise Exception('No WAV files in input path: %s' % InputFileOrFolder)

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
        #segments = segmenter.process(audio, 0.9, 0, 30, 300, 0.2)
        segments = segmenter.process(audio, args.trfactor, args.minsegdur, args.runoffdur, args.maxsegdur, args.minsildur)

        for index, segment in enumerate(segments):
            outFileName = "%s/%s_%d_%d.wav" % (outFolder, sessionId, index, (int(segment.timestamp*100)))
            write_wave(outFileName, segment.bytes, sample_rate)

# p run_segmenter.py --inpath 0.wav --outpath chunks -v 3 -a 3 -l 30 -w 20 -f 0.75 -r 30.0 -m 300 -p 0.3 -d 10
