import numpy as np
import sys
import os
import kaldiio
import kaldi_io
import soundfile as sf


with kaldiio.ReadHelper('scp:data/reverb/wav.scp') as reader:
    for key, (r, a) in reader:
        kaldi = a
        print(kaldi.astype(np.float64, order='C') / 32768)
        break

copy, sr = sf.read('reverb_test.wav')
# normalize to float32
print(kaldi.astype(np.float64, order='C') / 32768)
print(copy)
#print(kaldi)
