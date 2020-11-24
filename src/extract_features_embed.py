import numpy as np
import os
from enum import Enum
import sys
import pickle
import random
import soundfile as sf
import python_speech_features as psf
from glob import glob
from resemblyzer import VoiceEncoder, preprocess_wav, normalize_volume

from extract_features import replace_zero_sequences

# Path to the dataset
DATA = 'data/concat/'
DEST = 'data/features/'
TEXT = 'data/concat/text' # ground truth annotations for each utterance
LIBRI_SOURCE = 'data/LibriSpeech/train-clean-100/'
TS_DROPOUT = False
CACHE_DVECTORS = False
embedding_cache = dict()

# feature extraction mode based on the target architecture
class Mode(Enum):
    VAD = 0
    SC = 1
    ET = 2
    ST = 3
    SET = 4

MODE = Mode.ST

def cos(a, b):
    """Computes the cosine similarity of two vectors"""
    return np.dot(a, b) / (np.sqrt(np.dot(a, a)) * np.sqrt(np.dot(b, b)))

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

def preprocess_wav (wav):
    """ Applies preprocessing operations to a waveform either on disk or in memory such that  
    The waveform will be resampled to match the data hyperparameters.
    """
    wav = normalize_volume(wav, -30, increase_only=True)
    return wav

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

                    # TODO save as numpy file...
                    with open(DEST + folder.name + '/' + utt_id + '.vad.fea', 'wb') as f:
                        pickle.dump((logfbanks, replace_zero_sequences(labels, 8)), f)

                # the baseline - combine speaker verification score and vad output
                elif MODE == Mode.SC:
                    pass #TODO

                # score based training
                elif MODE == Mode.ST or MODE == Mode.ET or MODE == Mode.SET:
                    # we need to extract partial embeddings for each utterance - each representing
                    # a certain time window. Then those embeddings are compared with the target
                    # speaker embedding via cosine similarity and this score is then used as
                    # a feature.

                    # randomly select a target speaker and compute his embedding
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

                    # get the partial utterances for the current utterance, but bypas the
                    # resemblyzer's wav_preprocess function - we don't want any vad preprocessing
                    rate = 3
                    x = preprocess_wav(x)
                    _, utt_embeds, slices = encoder.embed_utterance(x, return_partials=True,
                            rate=rate, min_coverage=0.5)

                    # compute the cosine similarity between the partial embeddings and the target
                    # speaker embedding
                    scores_raw = np.array([ cos(spk_embed, cur_embed) for cur_embed in utt_embeds ])

                    # span the extracted scores to the whole utterance length
                    # - the first 160 frames are the first score as the embedding is computed from
                    #   a 1.6s long window
                    # - all the other scores have frame_step frames between them
                    samples_per_frame = 160
                    frame_step = int(np.round((16000 / rate) / samples_per_frame))
                    scores = np.append(np.kron(scores_raw[0], np.ones(160, dtype=scores_raw.dtype)),
                        np.kron(scores_raw, np.ones(frame_step, dtype=scores_raw.dtype)))
                    assert scores.size >= logfbanks.shape[0],\
                        "Error: The score array was longer than the actual feature vector."

                    # trim the score vector to be the same length as the acoustic features
                    scores = scores[:logfbanks.shape[0]]

                    # now relabel the ground truths to three classes... (ns, ntss, tss) -> {0, 1, 2}
                    labels = np.ones(n, dtype=np.long)
                    stamp_prev = 0
                    tstamps = tstamps // 10

                    for (stamp, label) in zip(tstamps, gtruth):
                        if label == '':
                            labels[stamp_prev:stamp] = 0
                        elif label == '$':
                            which -= 1;
                            labels[stamp_prev:stamp] = 0
                        else:
                            if which == 0: # tss
                                labels[stamp_prev:stamp] = 2
                            #else: # ntss - no need to label, the array is already filled with ones
                                #labels[stamp_prev:stamp] = 1

                        stamp_prev = stamp

                    # save the extracted features
                    np.savez(DEST + folder.name + '/' + utt_id + '.pvad.fea',
                            x=logfbanks, scores=scores, embed=spk_embed, y=labels)


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
