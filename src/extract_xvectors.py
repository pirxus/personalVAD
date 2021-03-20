"""Extracts and saves x-vectors for each speaker in the LibriSpeech dataset.
"""

import numpy as np
import os
import soundfile as sf
from glob import glob
import torchaudio
import speechbrain
from speechbrain.pretrained import SpeakerRecognition
import torch
from kaldiio import WriteHelper
import tempfile

LIBRI_SOURCE = 'LibriSpeech/'
DIRECTORIES = ['dev-clean']#, 'dev-other', 'test-clean', 'test-other']
               #'train-clean-100', 'train-clean-360'],
               #'train-other-500']
DEST = 'embeddings'

N_WAVS = 2
XVECTORS = True

# create a temporary directory for the xvector model
tmpdir = tempfile.TemporaryDirectory()

# x-vector interface
xvector_model = SpeakerRecognition.from_hparams(
        source="speechbrain/spkrec-xvect-voxceleb",
        savedir=tmpdir.name)


# open the ark and scp for writing
if XVECTORS:
    xvector_writer = WriteHelper(f'ark,scp:{DEST}/xvectors.ark,{DEST}/xvectors.scp')

for directory in DIRECTORIES:
    print(f"Processing directory: {directory}")
    with os.scandir(LIBRI_SOURCE + directory) as speakers:
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
                for i in range(N_WAVS):
                    random_file = files[np.random.randint(0, n_files)]
                    x, sr = torchaudio.load(random_file)
                    wavs.append(x)

                # concatenate the utterances..
                utt = torch.cat(wavs, dim=1)

                # compute the xvector
                xvector = xvector_model.encode_batch(utt).numpy().squeeze().squeeze()

                # save the embedding
                if XVECTORS:
                    xvector_writer(speaker.name, xvector)

if XVECTORS:
    xvector_writer.close()

# delete the temporary directory
tmpdir.cleanup()
