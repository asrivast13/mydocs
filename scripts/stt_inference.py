from traceback import print_tb
import torch
import os.path
import argparse
from huggingsound import SpeechRecognitionModel, KenshoLMDecoder

if __name__ == '__main__':
    
    AllowedAudioExtensions =  set(['wav', 'ogg', 'mp3'])
    parser = argparse.ArgumentParser(description='Command line Arguments for STT Decoder')

    # Directories/files
    parser.add_argument('-k', '--kensho',
                        type=bool, help='should we run the Kensho LM decoder')

    parser.add_argument('-i', '--input',
                        type=str, help='path to the input file list or audio file')

    parser.add_argument('-m', '--model', default='/root/Models',
                        type=str, help='path to the model folder')

    args = parser.parse_args()
    audio_paths=[]
    if(args.input is None):
        print('\nNo input specified -- try "-h"\n')
        exit(-1)
    else:
        filext = args.input.split('.')[-1].lower()
        if(filext in AllowedAudioExtensions):
            audio_paths.append(args.input)
        else:
            with open(args.input, 'r') as fp:
                audio_paths.append(fp.readlines())

    print("Audio Paths: ", audio_paths)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Trying device: ", device)
    batch_size = 1

    SttModelFolder = args.model
    LmModelFolder  = SttModelFolder + "language_model/"
    model = SpeechRecognitionModel(SttModelFolder, device=device)
    decoder = None
    if(args.kensho == False):        
        # The LM format used by the LM decoders is the KenLM format (arpa or binary file).
        lm_path = LmModelFolder + "lm.binary"
        unigrams_path = LmModelFolder + "unigrams.txt"

        # To use this decoder you'll need to install the Kensho's ctcdecode first (https://github.com/kensho-technologies/pyctcdecode)
        decoder = KenshoLMDecoder(model.token_set, lm_path=lm_path, unigrams_path=unigrams_path, alpha=2,     beta=1, beam_width=100)

    transcriptions = model.transcribe(audio_paths, batch_size=batch_size, decoder=decoder)
    print("\n\n", transcriptions)
    exit(0)
