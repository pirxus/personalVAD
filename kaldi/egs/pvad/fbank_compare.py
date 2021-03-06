import numpy as np
import sys
import os
import kaldi_io
import python_speech_features as psf
import librosa
import soundfile as sf
import scipy

UTT = '1272-128104-0001_1993-147965-0003_2078-142845-0041'
AUDIO = f'data/clean/1_concat/{UTT}.flac'
SCP = 'data/clean/data/raw_fbank_clean.1.scp'

# load the actual audio file
x, sr = sf.read(AUDIO, dtype='int16')
assert (sr == 16000), f'Invalid source audio sample rate {sr}'
logfbanks, energy = psf.base.fbank(x, nfilt=40, winfunc=np.hamming)
one = np.log(logfbanks)


#specto = librosa.feature.melspectrogram(x, sr=16000, n_fft=400, hop_length=160, n_mels=40,
#        window='hamming')
#two = np.log10(specto.T)
#three = librosa.core.amplitude_to_db(specto).T
#four = librosa.core.amplitude_to_db(logfbanks.T).T


# load the extracted kaldi fbanks
for utt, mat in kaldi_io.read_mat_scp(SCP):
    if utt == UTT:
        print(logfbanks.shape, x.shape)
        print(mat.shape)
        #print(one.shape, two.shape, three.shape, four.shape)
        #print(logfbanks - specto.T[:-2])
        #print(logfbanks - specto.T[1:-1])
        #print(logfbanks - specto.T[2:])
        #print('')
        #print(three[100:200])
        #print(four[100:200])
        #print(mat)

        print(one)
        print(mat)
        break
