import time
from nemo.collections.nlp.models import PunctuationCapitalizationModel
import argparse
import sys
import torch

# to get the list of pre-trained models
#PunctuationCapitalizationModel.list_available_models()

# Read text sentences from input file
parser = argparse.ArgumentParser()
parser.add_argument(
    "--model", "-m", type=str, required=False, default="punctuation_en_distilbert", help="Path to the model",
    choices=[m.pretrained_model_name for m in PunctuationCapitalizationModel.list_available_models()]
)
parser.add_argument(
    "--inpath", "-i", type=str, required=True, help="Input Path"
)
parser.add_argument(
    "--outpath", "-o", type=str, required=True, help="Output Path"
)
parser.add_argument(
    "--batch_size", "-b", type=int, default=128, help="Number of segments which are processed simultaneously.",
)
parser.add_argument(
    "--device",
    "-d",
    choices=['cpu', 'cuda'],
    help="Which device to use. If device is not set and CUDA is available, then GPU will be used. If device is "
    "not set and CUDA is not available, then CPU is used.",
)
     
args = parser.parse_args()

model = None
start = time.time()
try:
    # Download and load the pre-trained BERT-based model
    model = PunctuationCapitalizationModel.from_pretrained(args.model)
except Exception as e:
    print('Could not load model. Failed with message: %s' % e)
    sys.exit(-1)
end = time.time()
print("Finished loading models in %.2f secs" % (end-start))

if args.device is None:
    if torch.cuda.is_available():
        model = model.cuda()
        print("Running on GPU device: %s" % torch.cuda.get_device_name())
    else:
        model = model.cpu()
        args.batch_size = 1
else:
    model = model.to(args.device) if (args.device=='cpu') or (args.device=='cuda' and torch.cuda.is_available()) else model.to('cpu')
    if(args.device == 'cuda' and torch.cuda.is_available()):
        print("Running on GPU device: %s" % torch.cuda.get_device_name())
    else:
        args.batch_size = 1
                
print("Running with batch size= %d" % args.batch_size)
# read all sentences from input file
with open(args.inpath, "r") as fp:
    textLines = fp.readlines()
    numinp  = len(textLines)
    start = time.time()
    results = model.add_punctuation_capitalization(textLines, batch_size=args.batch_size)
    end = time.time()
    print(" Time taken overall for %d lines: %.2fsecs | time per line: %.1fmsec\n" % (numinp, (end-start), (1000*(end-start)/numinp)))
    print("Writing output to file %s" % args.outpath)
    start=time.time()
    with open(args.outpath, "w") as ofp:
        ofp.writelines(s + '\n' for s in results)
    end=time.time()
    numinp = len(results)
    print("Finished writing %d lines in %.2f secs" % (numinp, (end-start)))
