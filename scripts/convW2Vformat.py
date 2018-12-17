#!/usr/bin/env python

# External dependencies
import gensim.models.keyedvectors as word2vec
import os.path
import argparse

parser = argparse.ArgumentParser(description='Parser for word2vec model file converter')

# Directories/files
parser.add_argument('-t', '--text',
                    type=str, help='path to the model file in text format')

parser.add_argument('-b', '--bin',
                    type=str, help='path to the model file in binary format')


args = parser.parse_args()

textfile   = args.text
binfile    = args.bin

if textfile is None and binfile is None:
    print('\nNo inputs - try "-h"')
    exit(-1)
    
istextfile = textfile and os.path.exists(textfile)
isbinfile  = binfile and os.path.exists(binfile)

if istextfile and isbinfile:
    print('\nBoth textfile "{}" and binary file "{}" already exist -- cannot do anything'.format(textfile, binfile))
    exit(-1)

if binfile is None:
    binfile = textfile + ".bin"

if textfile is None:
    textfile = binfile + ".txt"

model = None
if istextfile:
    print('Reading model from text file "{}" and writing to binary file "{}" ...'.format(textfile, binfile))
    model = word2vec.KeyedVectors.load_word2vec_format(textfile, binary=False)
    model.save_word2vec_format(binfile, binary=True)
else:
    print('Reading model from binary file "{}" and writing to text file "{}" ...'.format(binfile, textfile))
    model = word2vec.KeyedVectors.load_word2vec_format(binfile, binary=True)
    model.save_word2vec_format(textfile, binary=False)

exit(0)
