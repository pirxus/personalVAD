import numpy as np
import os
import sys
import pickle
import soundfile as sf
import python_speech_features as psf
from glob import glob
from features import flac16khz2lmfbank


"""
- for each utterance
    1) load the signal
    2) extract the log mel-filterbanks
    3) generate the ground truth labels
    4) save both the annotations and the features to a separate folder
"""

# Path to the concatenated utterances
DATA = 'data/concat/'
DEST = 'data/features/'

# first create the destination directory
if os.path.exists(DEST):
    print('The specified destination folder is an existing file/directory')
    sys.exit()
else:
    try:
        os.mkdir(DEST)
    except OSError:
        print(f'Could not create destination directory {DEST}')

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
            logfbanks = np.log(fbanks)

            # now load the transcription and the alignment timestamps
            txt = open(folder.path + '/' + utt_id + '.txt')
            line = txt.readline(); txt.close()
            gtruth = line.partition(' ')[0].split(',')
            tstamps = [int(float(stamp)*1000) for stamp in line.split(' ')[1:]]
            gt_len = len(gtruth)
            assert (gt_len == len(tstamps)), f"gtruth and tstamps arrays have to be the same"

            # now generate n ground truth labels based on the gtruth and tstamps labels
            # where n is the number of feature frames we extracted
            n = logfbanks.shape[0]

            # NOTE: the timestamp doesn't really match the value of n. Keep an eye out..
            if tstamps[-1] < n*10:
                tstamps[-1] = n * 10

            labels = np.ones(n)
            j = 0 # current index to the gtruth/tstamps array
            for i in range(n): # TODO: SLOW!!!! think of something better...
                if i * 10 > tstamps[j]: j += 1
                assert (j < gt_len), f"j={j} out of range {gt_len}"
                # set the label, 1==speech frame, 0==non-speech frame
                if gtruth[j] in ['', '$']: labels[i] = 0

            with open(DEST + folder.name + '/' + utt_id + '.fea', 'wb') as f:
                pickle.dump((logfbanks, labels), f)
                f.close()
