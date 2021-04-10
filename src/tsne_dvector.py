# This script prepares dvector data for tsne analysis. The tsne analysis is supposed to
# compare the discriminative quality of dvectors that are continuously extracted at frame-level
# and dvectors extracted on window-level with an arbitrary sliding step

import numpy as np
import os
import soundfile as sf
import tempfile
import torch
import argparse as ap
from glob import glob
from kaldiio import WriteHelper
from resemblyzer import preprocess_wav, normalize_volume
from speechbrain.pretrained import SpeakerRecognition
from resemblyzer_mod import VoiceEncoderMod

LIBRI_ROOT = 'LibriSpeech'
DEST = 'tsne_embeddings'
DIRECTORIES = ['dev-clean']

N_SPEAKERS = 10
N_WAVS = 3 # use N wavs for each speaker

# program line arguments
parser = ap.ArgumentParser(description="Extract speaker embeddings for the LibriSpeech dataset.")
parser.add_argument('--libri_root', type=str, required=False, default=LIBRI_ROOT,
        help="Specify the path to the LibriSpeech dataset")
parser.add_argument('--embed_out', type=str, required=False, default=DEST,
        help="Specify the embedding output folder")
args = parser.parse_args()

LIBRI_ROOT = args.libri_root
DEST = args.embed_out

# resemblyzer settings..
rate = 2.5
samples_per_frame = 160
frame_step = int(np.round((16000 / rate) / samples_per_frame))
min_coverage = 0.5

if DEST[-1] == '/': DEST = DEST[:-1]

# d-vector interface, x-vector interface
encoder = VoiceEncoderMod()

for directory in DIRECTORIES:
    print(f"Processing directory: {directory}")
    with os.scandir(LIBRI_ROOT + '/' + directory) as speakers:
        for i, speaker in enumerate(speakers):
            if i == 10: break

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

                # choose n random files for the current speaker and load the waveforms
                for i in range(N_WAVS):
                    random_file = files[np.random.randint(0, n_files)]
                    x, sr = sf.read(random_file)
                    wavs_dvector.append(x)

                # extract and save the dvector
                for arr in wavs_dvector:
                    # extract the filterbank features
                    fbanks = librosa.feature.melspectrogram(arr, 16000, n_fft=400,
                            hop_length=160, n_mels=40).astype('float32').T

                    wav = arr.copy()
                    wav_slices, mel_slices = encoder.compute_partial_slices(
                            wav.size, rate, min_coverage)
                    max_wave_length = wav_slices[-1].stop
                    if max_wave_length >= wav.size:
                        wav = np.pad(arr, (0, max_wave_length - wav.size), "constant")
                    mels = librosa.feature.melspectrogram(wav, 16000, n_fft=400,
                            hop_length=160, n_mels=40).astype('float32').T
                    # create the fbanks slices...
                    fbanks_sliced = np.array([mels[s] for s in mel_slices])

                    fbanks = fbanks.to(encoder.device)
                    fbanks_sliced = fbanks_sliced.to(encoder.device)

                    with torch.no_grad():
                        # pass the tensors through the model (two forward methods...) and get
                        # the embeddings
                        embeds_stream, _ = encoder.forward_stream(fbanks, None)
                        embeds_stream = embeds_stream.cpu().numpy().squeeze()

                        # windowed embeddings..
                        embeds_slices = encoder(fbanks_sliced).cpu().numpy()


                    # get the utterance embeddings
                    embed_win = embeds_slices.mean()
                    print(f"embed_win mean shape: {embed_win.shape}")

                    #TODO stream
                    #TODO save stream


                    np.save(DEST + '/' + speaker.name + '.dvector_win', dvector)

