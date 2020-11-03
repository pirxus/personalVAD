import numpy as np
import os
import sys
import pickle
import shlex, subprocess
from glob import glob
from extract_features import replace_zero_sequences
import kaldi_io

FBANKS = 'kaldi/egs/pvad/fbank/'
TSTAMPS = 'kaldi/egs/pvad/data/clean/'

def load_timestamp_paths(root_dir):
    utterances = dict()
    for dir_name, subdir_list, file_list in os.walk(root_dir):
        if dir_name.rpartition('_')[2] != 'concat': continue
        print(f"Entering the {dir_name} directory")

        # load the paths to the ground truth labels
        for f in glob(dir_name + '/*.txt'):
            utterances[f.rpartition('/')[2].split('.')[0]] = f

    return utterances

# first, let's load the paths to the timestamp files for each utterance
transcripts = load_timestamp_paths(TSTAMPS)

# now let's generate a ground truth label array for each utterance
for f in glob(FBANKS + '/*.scp'):
    print(f"Opening file {f}")
    one_scp_file = list()
    for utt, mat in kaldi_io.read_mat_scp(f):
        # first get the real utterance id.. throw away the augmentation prefix
        utt_id = utt.partition('-')[2] if utt[0] == 'r' else utt

        # now load the transcription and the alignment timestamps
        with open(transcripts[utt_id]) as txt:
            line = txt.readline()
        gtruth = line.partition(' ')[0].split(',')
        tstamps = np.array([int(float(stamp)*1000) for stamp in line.split(' ')[1:]],
                dtype=np.int)
        gt_len = len(gtruth)
        assert (gt_len == tstamps.size), f"gtruth and tstamps arrays have to be the same"

        # now generate n ground truth labels based on the gtruth and tstamps labels
        # where n is the number of feature frames we extracted
        n = mat.shape[0]

        # NOTE: the timestamp doesn't really match the value of n. Keep an eye out..
        if tstamps[-1] < n*10:
            tstamps[-1] = n * 10

        labels = np.ones(n)
        stamp_prev = 0
        tstamps = tstamps // 10
        for (stamp, label) in zip(tstamps, gtruth):
            if label in ['', '$']: labels[stamp_prev:stamp] = 0
            stamp_prev = stamp

        # write the ground truth labels into an ark file

        one_scp_file.append((utt, np.expand_dims(replace_zero_sequences(labels, 8), axis=1)))

    # TODO: investigate the scp files...
    ark_file = f.replace('raw_fbank', 'gtruth').replace('scp', 'ark')
    with open(ark_file, 'wb') as ark:
        for utt, labels in one_scp_file:
            kaldi_io.write_mat(ark, labels, key=utt)
