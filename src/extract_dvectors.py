from resemblyzer import VoiceEncoder, preprocess_wav
from pathlib import Path
from typing import Union, Optional
import numpy as np
import librosa
import python_speech_features as psf


def preprocess_wav_my (fpath_or_wav: Union[str, Path, np.ndarray]):
    """
    Applies preprocessing operations to a waveform either on disk or in memory such that  
    The waveform will be resampled to match the data hyperparameters.

    :param fpath_or_wav: either a filepath to an audio file (many extensions are supported, not 
    just .wav), either the waveform as a numpy array of floats.
    :param source_sr: if passing an audio waveform, the sampling rate of the waveform before 
    preprocessing. After preprocessing, the waveform's sampling rate will match the data 
    hyperparameters. If passing a filepath, the sampling rate will be automatically detected and 
    this argument will be ignored.
    """
    # Load the wav from disk if needed
    if isinstance(fpath_or_wav, str) or isinstance(fpath_or_wav, Path):
        wav, source_sr = librosa.load(fpath_or_wav, sr=None)
    else:
        wav = fpath_or_wav
    
    ## Apply the preprocessing: normalize volume and shorten long silences 
    #wav = normalize_volume(wav, audio_norm_target_dBFS, increase_only=True)
    #wav = trim_long_silences(wav)
    
    return wav

np.set_printoptions(precision=3, suppress=True)
encoder = VoiceEncoder()
for audio in ['../data/LibriSpeech/dev-clean/1272/128104/1272-128104-0001.flac',
        '../data/LibriSpeech/dev-clean/1272/128104/1272-128104-0002.flac',
        '../data/LibriSpeech/dev-clean/1272/128104/1272-128104-0003.flac',
        '../data/LibriSpeech/dev-clean/1272/128104/1272-128104-0004.flac',
        '../data/LibriSpeech/dev-clean/1272/128104/1272-128104-0005.flac',
        '../data/LibriSpeech/dev-clean/1272/128104/1272-128104-0006.flac',
        '../data/LibriSpeech/dev-clean/1272/128104/1272-128104-0007.flac',
        '../data/LibriSpeech/dev-clean/1272/128104/1272-128104-0008.flac',
        '../data/LibriSpeech/dev-clean/1272/128104/1272-128104-0009.flac']:

    fpath = Path(audio)
    wav1 = preprocess_wav(fpath)

    wav2 = preprocess_wav_my(fpath)
    # test the log mel-filterbanks produced by psf against the librosa mel-filterbanks
    #print(16000 * 0.25)
    #print(16000 * 0.01)
    mel = librosa.feature.melspectrogram(wav2, 16000,
        n_fft=int(16000*0.25),
        hop_length=160,
        n_mels=40,
        window='hamming')
    log_mel, energy = psf.base.fbank(wav2, nfilt=40, winfunc=np.hamming)

    #print(mel.T[:10] - log_mel[:10])

    embed1, partials1, slices1 = encoder.embed_utterance(wav1, return_partials=True)
    embed2, partials2, slices2 = encoder.embed_utterance(wav2, return_partials=True, rate=2, min_coverage=0.2)
    print(len(partials2), len(partials1))
    for slc in slices2:
        print(wav2[slc].shape)
        pass
    #print(wav2.size/(16000), len(partials1), len(partials2), wav1.size/wav2.size)
