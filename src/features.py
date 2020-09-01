#!/usr/bin/python3

import numpy as np
from librosa import load, feature
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
        x, sr = load(f, sr = 160000)
        n_fft = int(sr * 0.02) # 20 ms window
        hop_len = n_fft // 2

        #TODO n_mfcc
        mfccs = feature.mfcc(x, sr=sr, n_mfcc=20, hop_length=hop_len, n_fft=n_fft)
        print(mfccs.shape)
        features[f] = mfccs

    return features

if __name__ == '__main__':
    flac16khz2mfcc('data/LibriSpeech/dev-clean/1272/128104/')
