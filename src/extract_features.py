import numpy as np
import os
import io
import sys
import pickle
import shlex, subprocess
import soundfile as sf
import python_speech_features as psf
from glob import glob


"""
- for each utterance id
    1) load the signal
    2) extract the log mel-filterbanks
    3) generate the ground truth labels
    4) save both the annotations and the features to a separate folder
"""

# This function helps us midigate some artifats in the label vector - replaces all sequences
# of zeros of given length in a given array with ones
def replace_zero_sequences(arr, seq_len):
    inside = False
    for i in range(arr.size):
        if arr[i] == 0:
            if not inside:
                inside = True
                curr = 1
            else:
                curr += 1
        else:
            if inside:
                if curr < seq_len: # replace the short sequence
                    arr[i - curr:i] = 1.
                inside = False
    return arr

# Path to the concatenated utterances
DATA = 'data/concat/'
DEST = 'data/features/'

def features_from_flac():
    with os.scandir(DATA) as folders:
        for folder in folders:
            print(f'Entering folder {folder.name}')
            os.mkdir(DEST + folder.name)

            for f in glob(folder.path + '/*.flac'):
                utt_id = f.rpartition('/')[2].split('.')[0]

                # first, extract the log mel-filterbank energies
                x, sr = sf.read(f)
                assert (sr == 16000), f'Invalid source audio sample rate {sr}'
                fbanks, energy = psf.base.fbank(x, nfilt=40, winfunc=np.hamming)
                logfbanks = np.log10(fbanks)

                # now load the transcription and the alignment timestamps
                with open(folder.path + '/' + utt_id + '.txt') as txt:
                    line = txt.readline()
                gtruth = line.partition(' ')[0].split(',')
                tstamps = np.array([int(float(stamp)*1000) for stamp in line.split(' ')[1:]],
                        dtype=np.int)
                gt_len = len(gtruth)
                assert (gt_len == tstamps.size), f"gtruth and tstamps arrays have to be the same"

                # now generate n ground truth labels based on the gtruth and tstamps labels
                # where n is the number of feature frames we extracted
                n = logfbanks.shape[0]

                # NOTE: the timestamp doesn't really match the value of n. Keep an eye out..
                if tstamps[-1] < n*10:
                    tstamps[-1] = n * 10

                labels = np.ones(n)
                stamp_prev = 0
                tstamps = tstamps // 10
                for (stamp, label) in zip(tstamps, gtruth):
                    if label in ['', '$']: labels[stamp_prev:stamp] = 0
                    stamp_prev = stamp

                with open(DEST + folder.name + '/' + utt_id + '.fea', 'wb') as f:
                    pickle.dump((logfbanks, replace_zero_sequences(labels, 8)), f)

# first create the destination directory
if __name__ == '__main__':
    if os.path.exists(DEST):
        if not os.path.isdir(DEST) or os.listdir(DEST):
            print('The specified destination folder is an existing file/directory')
            sys.exit()
    try:
        os.mkdir(DEST)
    except OSError:
        print(f'Could not create destination directory {DEST}')

    features_from_flac() # the old way...
