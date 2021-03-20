"""Extracts and saves speaker embedding vectors for each speaker in the LibriSpeech dataset
and saves them into an ark and an scp file.
"""

import numpy as np
import os
import soundfile as sf
import tempfile
import torch
import argparse as ap
from glob import glob
from kaldiio import WriteHelper
from resemblyzer import VoiceEncoder, preprocess_wav, normalize_volume
from speechbrain.pretrained import SpeakerRecognition

LIBRI_ROOT = 'LibriSpeech'
DEST = 'embeddings'
DIRECTORIES = ['dev-clean', 'dev-other', 'test-clean', 'test-other',
               'train-clean-100', 'train-clean-360', 'train-other-500']

N_WAVS = 2

# program line arguments
parser = ap.ArgumentParser(description="Extract speaker embeddings for the LibriSpeech dataset.")
parser.add_argument('--libri_root', type=str, required=False, default=LIBRI_ROOT,
        help="Specify the path to the LibriSpeech dataset")
parser.add_argument('--embed_out', type=str, required=False, default=DEST,
        help="Specify the embedding output folder")
parser.add_argument('--n_wavs', type=int, required=False, default=N_WAVS,
        help="Number of processed utterances for each speaker")
parser.add_argument('--dvector', action='store_true',
        help="Extract d-vectors")
parser.add_argument('--xvector', action='store_true',
        help="Extract x-vectors")
parser.add_argument('--use_numpy', action='store_true',
        help="Save the individual embeddings in the *.npy format instead of scp/ark")
parser.add_argument('parts', type=str, nargs='*', default=DIRECTORIES,
        help="Specify which LibriSpeech folders to process")
args = parser.parse_args()

LIBRI_ROOT = args.libri_root
DEST = args.embed_out
DVECTORS = args.dvector
XVECTORS = args.xvector
N_WAVS = args.n_wavs
DIRECTORIES = args.parts
USE_NUMPY = args.use_numpy

if DEST[-1] == '/': DEST = DEST[:-1]

# create a temporary directory for the xvector model
tmpdir = tempfile.TemporaryDirectory()

# d-vector interface, x-vector interface
dvector_model = VoiceEncoder()
xvector_model = SpeakerRecognition.from_hparams(
        source="speechbrain/spkrec-xvect-voxceleb",
        savedir=tmpdir.name)

if DVECTORS and not USE_NUMPY:
    dvector_writer = WriteHelper(f'ark,scp:{DEST}/dvectors.ark,{DEST}/dvectors.scp')
if XVECTORS and not USE_NUMPY:
    xvector_writer = WriteHelper(f'ark,scp:{DEST}/xvectors.ark,{DEST}/xvectors.scp')

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
                wavs_dvector = []
                wavs_xvector = []

                # choose n random files for the current speaker and load the waveforms
                for i in range(N_WAVS):
                    random_file = files[np.random.randint(0, n_files)]
                    x, sr = sf.read(random_file)
                    if DVECTORS:
                        wavs_dvector.append(preprocess_wav(x))
                    if XVECTORS:
                        wavs_xvector.append(torch.from_numpy(x).unsqueeze(dim=0))

                if DVECTORS:
                    # extract and save the dvector
                    dvector = dvector_model.embed_speaker(wavs_dvector)
                    if not USE_NUMPY:
                        dvector_writer(speaker.name, dvector)
                    else:
                        np.save(DEST + '/' + speaker.name + '.dvector', dvector)

                if XVECTORS:
                    # extract and save the xvector
                    utt = torch.cat(wavs_xvector, dim=1)
                    xvector = xvector_model.encode_batch(utt).numpy().squeeze().squeeze()
                    if not USE_NUMPY:
                        xvector_writer(speaker.name, xvector)
                    else:
                        np.save(DEST + '/' + speaker.name + '.xvector', xvector)

if DVECTORS and not USE_NUMPY: dvector_writer.close()
if XVECTORS and not USE_NUMPY: xvector_writer.close()

# delete the temporary directory
tmpdir.cleanup()
