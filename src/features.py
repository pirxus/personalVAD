#!/usr/bin/python3

import numpy as np
import python_speech_features as psf
from librosa import load
from glob import glob

# Paths to librispeech
DEV_CLEAN = 'data/LibriSpeech/dev-clean/'
DEV_OTHER = 'data/LibriSpeech/dev-other/'
TEST_CLEAN = 'data/LibriSpeech/test-clean/'
TEST_OTHER = 'data/LibriSpeech/test-other/'

# Loads all flac files from a specified directory and extracts mfcc features
# (20 coefficients)
def flac16khz2mfcc(directory):
    features = {}
    for f in glob(directory + '/*.flac'):
        x, sr = load(f, sr=160000)
        mfccs = psf.base.mfcc(x, numcep=20, nfilt=40, winfunc=np.hamming)
        print(mfccs.shape)
        features[f] = mfccs

    return features

# Loads all flac files from a specified directory and extracts log Mel-filterbank energies
# (40 dimensions)
def flac16khz2lmfbank(directory): 
    features = {}
    for f in glob(directory + '/*.flac'):
        x, sr = load(f, sr=160000)

        # NOTE: fiddle with the nfft param?
        lmfbs = psf.base.logfbank(x, nfilt=40)
        print(lmfbs.shape)
        features[f] = lmfbs

    return features

if __name__ == '__main__':
    #flac16khz2mfcc('data/LibriSpeech/dev-clean/1272/128104/')
    flac16khz2lmfbank('data/LibriSpeech/dev-clean/1272/128104/')
