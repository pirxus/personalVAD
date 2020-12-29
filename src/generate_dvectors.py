"""Extracts and saves speaker embedding vectors for each speaker in the LibriSpeech dataset.
"""

import numpy as np
import os
import soundfile as sf
from glob import glob
from resemblyzer import VoiceEncoder, preprocess_wav, normalize_volume

from extract_features_embed import get_speaker_embedding

LIBRI_SOURCE = 'data/LibriSpeech/'
DIRECTORIES = ['dev-clean', 'dev-other', 'test-clean', 'test-other', 'train-clean-100']
EMBED_OUT = 'data/embeddings/'

N_WAVS = 2

encoder = VoiceEncoder()

for directory in DIRECTORIES:
    with os.scandir(LIBRI_SOURCE + directory) as speakers:
        for speaker in speakers:
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
                    wavs.append(preprocess_wav(sf.read(random_file)[0]))

                # extract the embedding
                embedding = encoder.embed_speaker(wavs)

                # save the embedding
                np.save(EMBED_OUT + speaker.name + '.dvector', embedding)
