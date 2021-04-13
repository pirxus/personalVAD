
import numpy as np
import os
import soundfile as sf
import tempfile
import torch
import argparse as ap
from glob import glob
from resemblyzer import trim_long_silences

LIBRI_ROOT = 'LibriSpeech'
DEST = 'ivec_extr'
DIRECTORIES = ['dev-clean', 'dev-other', 'test-clean', 'test-other',
               'train-clean-100', 'train-clean-360', 'train-other-500']

# command line arguments
parser = ap.ArgumentParser(description="Extract speaker embeddings for the LibriSpeech dataset.")
parser.add_argument('--libri_root', type=str, required=False, default=LIBRI_ROOT,
        help="Specify the path to the LibriSpeech dataset")
parser.add_argument('--list_out', type=str, required=False, default=DEST,
        help="Specify the destination of the file_list")
parser.add_argument('parts', type=str, nargs='*', default=DIRECTORIES,
        help="Specify which LibriSpeech folders to process")
args = parser.parse_args()

LIBRI_ROOT = args.libri_root
DEST = args.list_out
DIRECTORIES = args.parts
N_WAVS = 3

# create the destination directory
try:
    os.makedirs(DEST + '/files')
except:
    pass

with open(DEST + '/file_list', 'w') as file_list:

    for directory in DIRECTORIES:
        print(f"Processing directory: {directory}")
        with os.scandir(LIBRI_ROOT + '/' + directory) as speakers:
            for speaker in speakers:
                print(f"Processing speaker: {speaker.name}")
                if not os.path.isdir(speaker.path): continue

                with os.scandir(speaker.path) as sessions:
                    sessions = list(sessions)
                    # select a random session
                    session = sessions[np.random.randint(0, len(sessions))]

                    # get the files for the current speaker
                    files = glob(session.path + '/*.flac')
                    n_files = len(files)
                    wavs = []

                    # choose n random files for the current speaker and load the waveforms
                    for i in range(N_WAVS):
                        random_file = files[np.random.randint(0, n_files)]
                        x, sr = sf.read(random_file)
                        wavs.append(x)

                    wav = trim_long_silences(np.concatenate(wavs)) # apply vad

                    # save the final flac to the destination folder
                    sf.write(DEST + '/files/' + speaker.name + '.flac', wav, sr)
                    file_list.write(speaker.name + '\n')
