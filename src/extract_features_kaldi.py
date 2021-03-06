#!/usr/bin/python

"""
This script is a replacement for the fbank script kaldi
offers for feature extraction.

The input signal is loaded via kaldiio from an scp recipe
and is further processed. First the input has to be normalized
to the floating point format, then the filterbank features
are extracted along with the ground truth annotations and
SV scores. All of the previous is then saved into an ark file.

"""

# TODO: use conda for librosa and kaldiio/other 3rd party libraries???

import os
import sys
import numpy as np
import kaldiio
import random
import librosa
from kaldiio import ReadHelper, WriteHelper
from glob import glob
import python_speech_features as psf
import multiprocessing as mp

from extract_features_embed import preprocess_wav, get_speaker_embedding,\
    Mode, embedding_cache, cos
from resemblyzer import VoiceEncoder

KALDI_ROOT = 'kaldi/egs/pvad/'
DATA_ROOT = 'data/augmented/'
REPO_ROOT = ''
EMBED_PATH = 'data/embeddings/'

""":"""
# Path to the dataset
DATA = 'data/concat/'
DEST = 'data/features/'
TEXT = 'data/concat/text' # ground truth annotations for each utterance
LIBRI_SOURCE = 'data/LibriSpeech/train-clean-100/'
TS_DROPOUT = True
CACHE_DVECTORS = True

# indicates whether the speaker embeddings are derived from the dataset
# or whether they are pre-generated and stored in the EMBED_PATH folder and should be pulled
# from there...
GEN_SPK_EMBEDDINGS = False

""":"""
MODE = Mode.ET

txt = dict()

def gpu_worker(q_send, q_return):
    # first, initialize the model
    encoder = VoiceEncoder()
    rate = 3

    while True:
        # fetch the tensor
        x, pid = q_send.get()

        # compute the partial embeddings
        _, utt_embeds, slices = encoder.embed_utterance(x, return_partials=True,
                rate=rate, min_coverage=0.5)

        # return the tensor back to the original process
        q_return[pid].put(utt_embeds)

def extract_features(scp, q_send, q_return):
    #encoder = VoiceEncoder()
    encoder = None
    wav_scp = ReadHelper(f'scp:{scp}')
    pid = int(scp.rpartition('.')[0].rpartition('_')[2]) # NOTE: critical for queue functionality
    array_writer = WriteHelper(f'ark,scp:{DEST}fbanks_{pid}.ark,{DEST}fbanks_{pid}.scp')
    score_writer = WriteHelper(f'ark,scp:{DEST}scores_{pid}.ark,{DEST}scores_{pid}.scp')
    embed_writer = WriteHelper(f'ark,scp:{DEST}embed_{pid}.ark,{DEST}embed_{pid}.scp')
    label_writer = WriteHelper(f'ark,scp:{DEST}labels_{pid}.ark,{DEST}labels_{pid}.scp')
    label_vad_writer = WriteHelper(f'ark,scp:{DEST}labels_vad_{pid}.ark,{DEST}labels_vad_{pid}.scp')

    i = 0
    for utt_id, (sr, arr) in wav_scp:

        # now load the transcription and the alignment timestamps
        try:
            gtruth, tstamps = text[utt_id]
        except:
            print(f"Error: key {utt_id} not found.")
            continue

        gt_len = len(gtruth)
        assert (gt_len == tstamps.size), f"gtruth and tstamps arrays have to be the same"

        # load the wav and normalize to float32
        arr = arr.astype(np.float32, order='C') / 32768

        # extract the filterbank features
        fbanks = librosa.feature.melspectrogram(arr, 16000, n_fft=int(16000*0.25),
                hop_length=160, n_mels=40, window='hamming')
        logfbanks = np.log10(fbanks + 1e-6).T[:-2]
            
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

            # save the extracted features
            np.savez(DEST + folder.name + '/' + utt_id + '.vad.fea',
                    x=logfbanks, y=replace_zero_sequences(labels, 8))

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
                use_target = bool(np.random.randint(0, 3))
                if use_target or embedding_cache == {}:
                    # target speaker
                    which = 0
                    spk_embed = get_speaker_embedding(utt_id, which,
                            encoder, path=EMBED_PATH)

                else:
                    # get a random speaker embedding ?? other than the current one ??
                    if 'rev' in utt_id: spk_id = utt_id.partition('-')[2]
                    spk_id = utt_id.split('-')[0]
                    rnd_spk_id, spk_embed = random.choice(list(embedding_cache.items()))
                    which = -1 if rnd_spk_id != spk_id else 0

            else:
                which = np.random.randint(0, n_speakers) 
                spk_embed = get_speaker_embedding(utt_id, which, encoder,
                        use_cache=CACHE_DVECTORS, path=EMBED_PATH)

            # get the partial utterances for the current utterance, but bypas the
            # resemblyzer's wav_preprocess function - we don't want any vad preprocessing

            # send the datata to be processed on the gpu and retreive the result
            q_send.put((preprocess_wav(arr), pid))
            utt_embeds = q_return.get()

            rate = 3
            """
            x = preprocess_wav(arr)
            _, utt_embeds, slices = encoder.embed_utterance(x, return_partials=True,
                    rate=rate, min_coverage=0.5)
            """

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
            labels = np.ones(n, dtype=np.float32)
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

            # now create one more label array for the base VAD system
            labels_vad = (labels != 0).astype('float32')

            # write the extracted features to the scp and ark files..
            array_writer(utt_id, logfbanks)
            score_writer(utt_id, scores)
            embed_writer(utt_id, spk_embed)
            label_writer(utt_id, labels)
            label_vad_writer(utt_id, labels_vad)

            # flush the results...
            if i % 100 == 0:
                array_writer.fark.flush()
                array_writer.fscp.flush()
                score_writer.fark.flush()
                score_writer.fscp.flush()
                embed_writer.fark.flush()
                embed_writer.fscp.flush()
                label_writer.fark.flush()
                label_writer.fscp.flush()
                label_vad_writer.fark.flush()
                label_vad_writer.fscp.flush()

    # close all the scps..
    wav_scp.close()
    array_writer.close()
    score_writer.close()
    embed_writer.close()
    label_writer.close()
    label_vad_writer.close()

def process_init(txt, embed_path):
    global text
    text = txt
    global EMBED_PATH
    EMBED_PATH = embed_path

if __name__ == '__main__':
    # change the cwd to the kaldi root in order to access the kaldi
    # required binaries...
    REPO_ROOT = os.getcwd() + '/' 
    EMBED_PATH = REPO_ROOT + EMBED_PATH
    os.chdir(KALDI_ROOT)

    # load the sv model
    if MODE != Mode.VAD:
        encoder = 'test'
        #encoder = VoiceEncoder()

    # first, load the utterance transcriptions
    with open('data/augmented/text') as text_file:
        for utterance in text_file:
            utt_id, _, rest = utterance.partition(' ')
            labels, _, tstamps = rest.partition(' ')
            # save them as preprocessed tuples...
            txt[utt_id] = (labels.split(','),
                    np.array([int(float(stamp)*1000) for stamp in tstamps.split(' ')], dtype=np.int))

    # get the file list for processing
    files = glob('data/augmented/split_*.scp')
    files.sort()
    nj = len(files)

    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        print("Cuda failed....")

    # create the communication queues
    manager = mp.Manager()
    q_send = manager.Queue() # main send queue
    q_return = [] # return queues
    for i in range(nj):
        q_return.append(manager.Queue())

    # create the gpu worker process
    worker = mp.Process(target=gpu_worker, args=(q_send, q_return,))
    worker.daemon = True
    worker.start()
    # create the process pool
    pool = mp.Pool(processes=nj, initializer=process_init, initargs=(txt, EMBED_PATH,))
    pool.starmap(extract_features, zip(files, [q_send] * nj, q_return))
    pool.close()
    pool.join()
