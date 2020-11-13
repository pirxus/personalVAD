import numpy as np
import os
from enum import Enum
import sys
import pickle
import random
import soundfile as sf
import python_speech_features as psf
from glob import glob
from resemblyzer import VoiceEncoder, preprocess_wav

from extract_features import replace_zero_sequences

# Path to the dataset
DATA = 'data/concat/'
DEST = 'data/features/'
TEXT = 'data/concat/text'
LIBRI_SOURCE = 'data/LibriSpeech/train-clean-100/'
TS_DROPOUT = True
CACHE_DVECTORS = True
embedding_cache = dict()

# feature extraction mode based on the target architecture
class Mode(Enum):
    VAD = 0
    SC = 1
    ST = 2
    ET = 3
    SET = 4

MODE = Mode.ET

def get_speaker_embedding(utt_id, spk_idx, encoder, n_wavs=2, use_cache=True):
    """Computes a d-vector speaker embedding for a target speaker and saves it to the 
    embedding_cache dictionary. If an embedding for our target is already present there, then
    that d-vector is returned instead.

    Args:
        utt_id (str): The whole concatenated utterance id.
        spk_idx (int): Which speaker from that utterance id is our target.
        encoder (resemblyzer.VoiceEncoder): Speaker encoder object that allows us to extract
            the speaker embedding.
        n_wavs (int, optional): Number of audio utterances used to calculate the embedding.
            Defaults to 2.
        use_caceh(bool, optional): Tells the function whether to cache the extracted embeddings
            or not. Defaults to True.

    Returns:
        numpy.ndarray: The extracted speaker embedding

    """

    # first remove the augmentation prefix...
    if 'rev' in utt_id:
        utt_id = utt_id.partition('-')[2]

    # get the speaker id
    spk_id = utt_id.split('_')[spk_idx].split('-')

    # check whether the embedding is already present in the embedding cache
    if use_cache and spk_id[0] in embedding_cache:
        embedding = embedding_cache[spk_id[0]]

    else:

        # compute the speaker embedding from a few audio files in their librispeech folder..
        files = glob(LIBRI_SOURCE + spk_id[0] + '/' + spk_id[1] + '/*.flac')
        wavs = list()
        for i in range(n_wavs):
            random_file = np.random.randint(0, n_wavs)
            wavs.append(preprocess_wav(sf.read(files[random_file])[0]))

        embedding = encoder.embed_speaker(wavs)
        # cache the d-vector
        if use_cache: embedding_cache[spk_id[0]] = embedding

    return embedding

def features_from_flac(text):
    """This function goes through the entire audio dataset specified by `DATA` and creates
    a new dataset of extracted features for a vad architecture specified in the `MODE` parameter.

    Args:
        text (file):
    """

    if MODE != Mode.VAD:
        encoder = VoiceEncoder()

    with os.scandir(DATA) as folders:
        for folder in folders:
            if not os.path.isdir(folder.path): continue
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
                try:
                    gtruth, tstamps = text[utt_id]
                except:
                    print(f"Error: key {utt_id} not found.")
                    continue

                gt_len = len(gtruth)
                assert (gt_len == tstamps.size), f"gtruth and tstamps arrays have to be the same"

                # now generate n ground truth labels based on the gtruth and tstamps labels
                # where n is the number of feature frames we extracted
                n = logfbanks.shape[0]

                # NOTE: the timestamp doesn't really match the value of n. Keep an eye out..
                if tstamps[-1] < n*10:
                    tstamps[-1] = n * 10

                # classic vad
                if MODE == Mode.VAD:
                    labels = np.ones(n)
                    stamp_prev = 0
                    tstamps = tstamps // 10

                    for (stamp, label) in zip(tstamps, gtruth):
                        if label in ['', '$']: labels[stamp_prev:stamp] = 0
                        stamp_prev = stamp

                    with open(DEST + folder.name + '/' + utt_id + '.vad.fea', 'wb') as f:
                        pickle.dump((logfbanks, replace_zero_sequences(labels, 8)), f)

                elif MODE == Mode.SC:
                    pass #TODO
                elif MODE == Mode.ST:
                    pass #TODO

                # target embedding vad
                elif MODE == Mode.ET:

                    # now onto d-vector extraction...
                    #wav = preprocess_wav(f, source_sr=sr)
                    #_, embeds, wav_slices = encoder.embed_utterance(wav, return_partials=True)
                    # choose a speaker at random
                    n_speakers = gtruth.count('$') + 1

                    # now, based on TS_DROPOUT, decide with a certain probability, whether to 
                    # make a one speaker utterance without a target speaker to mitigate
                    # overfitting for the target speaker class
                    if TS_DROPOUT and n_speakers == 1 and CACHE_DVECTORS:
                        if use_target := bool(np.random.randint(0, 2)) or embedding_cache == {}:
                            # target speaker
                            which = 0
                            spk_embed = get_speaker_embedding(utt_id, which, encoder)

                        else:
                            # get a random speaker embedding ?? other than the current one ??
                            if 'rev' in utt_id: spk_id = utt_id.partition('-')[2]
                            spk_id = utt_id.split('-')[0]
                            rnd_spk_id, spk_embed = random.choice(list(embedding_cache.items()))
                            which = -1 if rnd_spk_id != spk_id else 0

                    else:
                        which = np.random.randint(0, n_speakers) 
                        spk_embed = get_speaker_embedding(utt_id, which, encoder,
                                use_cache=CACHE_DVECTORS)

                    # now relabel the ground truths to three classes... (tss, ntss, ns) -> {0, 1, 2}
                    labels = np.ones(n, dtype=np.long)
                    stamp_prev = 0
                    tstamps = tstamps // 10

                    for (stamp, label) in zip(tstamps, gtruth):
                        if label == '':
                            labels[stamp_prev:stamp] = 2
                        elif label == '$':
                            which -= 1;
                            labels[stamp_prev:stamp] = 2
                        else:
                            if which == 0: # tss
                                labels[stamp_prev:stamp] = 0
                            #else: # ntss
                                #labels[stamp_prev:stamp] = 1

                        stamp_prev = stamp

                    with open(DEST + folder.name + '/' + utt_id + '.et_vad.fea', 'wb') as f:
                        pickle.dump((logfbanks, spk_embed, labels), f)

                elif MODE == Mode.SET:
                    pass #TODO


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

    # first, load the utterance transcriptions
    text = dict()
    with open(TEXT) as text_file:
        for utterance in text_file:
            utt_id, _, rest = utterance.partition(' ')
            labels, _, tstamps = rest.partition(' ')
            # save them as preprocessed tuples...
            text[utt_id] = (labels.split(','),
                    np.array([int(float(stamp)*1000) for stamp in tstamps.split(' ')], dtype=np.int))

    # extract the features
    features_from_flac(text)
