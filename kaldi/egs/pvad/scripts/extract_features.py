#!/usr/bin/python

"""
This script is a replacement for the fbank script kaldi
offers for feature extraction.

The input signal is loaded via kaldiio from an scp recipe
and is further processed. First the input has to be normalized
to the floating point format, then the filterbank features
are extracted along with the ground truth annotations and
SV scores. All of the previous is then saved into an ark file.

"""

# TODO: use conda for librosa and kaldiio/other 3rd party libraries???

import os
import sys
import numpy as np
import kaldiio

DATA_ROOT = '../data/augmented/'

for i, (utt, (sr, arr)) in enumerate(kaldiio.load_scp_sequential('data/augmented/wav.scp')):
    print(arr.astype(np.float64, order='C') / 32768)
    if i == 4:
        break
