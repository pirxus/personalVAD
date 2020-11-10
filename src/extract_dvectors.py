from resemblyzer import VoiceEncoder, preprocess_wav
from pathlib import Path
import numpy as np

np.set_printoptions(precision=3, suppress=True)
encoder = VoiceEncoder()
for audio in ['../data/LibriSpeech/dev-clean/1272/128104/1272-128104-0001.flac',
        '../data/LibriSpeech/dev-clean/1272/128104/1272-128104-0002.flac',
        '../data/LibriSpeech/dev-clean/1272/128104/1272-128104-0003.flac']:

    fpath = Path(audio)
    wav = preprocess_wav(fpath)

    embed = encoder.embed_utterance(wav)
    print(embed)
