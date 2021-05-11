#!/usr/bin/python

"""@package extract_features

Author: Simon Sedlacek
Email: xsedla1h@stud.fit.vutbr.cz

This script is responsible for the whole feature extraction process,
right after the augmentation process is done.

The script utilizes mutliprocessing - N* workers are responsible
for loading the waveform information from the input .scp recipes,
and extracting the acoustic features. The acoustic features are then
passed to one GPU-worker, which extracts frame-level d-vector embeddings
for each utterance and passes them back to the original CPU worker.

The CPU worker then computes the cosine similarity scores for the whole
utterance and converts the utterance annotations to PVAD ground truth labels.

The acoustic features are saved to fbanks.scp/.ark, the ground truth labels
to labels.scp/.ark, and the scores to scores.scp/.ark files.

The information about the target speaker for each utterance is saved into a
targets.scp file.

IMPORTANT! the wav-reverberate Kaldi program has to be accessible via PATH if
augmentation was used. The kaldiio library needs this program to retrieve the
waveforms from the .scp recipes.

============

*N - the number of CPU workers is dependent on the number of .scp files generated
in the augmentation process...

"""

import os
import sys
import numpy as np
import kaldiio
import random
import librosa
import argparse as ap
import torch
from kaldiio import ReadHelper, WriteHelper
from glob import glob
import multiprocessing as mp

from resemblyzer import VoiceEncoder
from resemblyzer_mod import VoiceEncoderMod

KALDI_ROOT = 'kaldi/egs/pvad/'
# Path to the dataset
DATA_ROOT = 'data/augmented/'
EMBED_PATH = 'data/embeddings/'
DEST = 'data/features/'
EMBED = 'data/embeddings/'

# if True, a single speaker utterance has a chance to randomly choose a
# different speaker as the target, rather than the speaker present in the utterance
TS_DROPOUT = False

txt = dict()

# some resemblyzer d-vector extraction parameters...
rate = 2.5
samples_per_frame = 160
frame_step = int(np.round((16000 / rate) / samples_per_frame))
min_coverage = 0.5

def cos(a, b):
    """Compute the cosine similarity of two vectors"""
    return np.dot(a, b) / (np.sqrt(np.dot(a, a)) * np.sqrt(np.dot(b, b)))


def load_dvector(utt_id, spk_idx, embed_scp, select_random=False):
    """Load the dvector for the target speaker.

    Load the d-vector representation for a selected speaker. Can
    also randomly select a d-vector representation of a different speaker.

    Args:
        utt_id (str): The id of the utterance in for which the speaker is selected.
        spk_id (int): The index of the speaker in the processed utterance.
        embed_scp (file): D-vector embedding scp stash. From this file, the d-vectors
            are loaded.
        select_random (bool, optional) Indicates whether a random speaker should be chosen
            instead of the one present in the utterance. Defaults to False.

    Returns:
        tuple: A tuple containing:
            embedding (np.array): The d-vector representation of the selected speaker.
            spk_id (str): The id of the selected speaker, whose d-vector is returned.
    """

    # get the speaker id
    if "rev1-" in utt_id: # just care for the old reverberation prefix...
        utt_id = utt_id[5:]
    spk_id = utt_id.split('_')[spk_idx].split('-')[0]

    if select_random:
        spk_id2 = spk_id
        while spk_id2 == spk_id:
            spk_id2, embedding = random.choice(list(embed_scp.items()))

        spk_id = spk_id2
    else:
        embedding = embed_scp[spk_id]
    return embedding, spk_id


def gpu_worker(q_send, q_return):
    """GPU worker process.

    This process is responsible for taking the acoustic features from the CPU
    workers and extracting frame-level d-vector embeddings for them. Extracts
    the d-vectors both in the baseline streaming manner and also utilizing the
    window-level d-vector inference for the scoring method modifications. After
    extraction, the d-vectors are returned to the original process.

    Args:
        q_send (mp.Manager.Queue): Input queue object shared by all CPU workers. 
        q_return (list of mp.Manager.Queue): A list of queue objects sorted by
            the CPU worker pid used to pass the extracted d-vectors back to the
            original CPU worker.
    """

    # first, initialize the encoder model
    encoder = VoiceEncoderMod()
    device = encoder.device

    while True:

        # retrieve the incoming fbanks
        fbanks, fbanks_sliced, pid = q_send.get()
        
        # move the incoming tensors to cuda
        fbanks = fbanks.to(device)
        fbanks_sliced = fbanks_sliced.to(device)

        with torch.no_grad():
            # pass the tensors through the model (two forward methods...) and get
            # the embeddings
            embeds_stream, _ = encoder.forward_stream(fbanks, None)
            embeds_stream = embeds_stream.cpu()

            # windowed embeddings..
            embeds_slices = encoder(fbanks_sliced).cpu()

        # return the tensor back to the original process
        q_return[pid].put((embeds_stream, embeds_slices))


def extract_features(scp, q_send, q_return):
    """CPU worker PVAD feature extraction proces.

    Args:
        scp (str): Path to the generated augmented dataset scp file this worker is supposed
            to process in its entirety.
        q_send (mp.Manager.Queue): The queue object used to pass the fbanks
            to the GPU worker.
        q_return (mp.Manager.Queue): The queue object used to retrieve the d-vectors
            from GPU worker.
    
    """

    # prepare the scp and and ark files
    wav_scp = ReadHelper(f'scp:{scp}')
    pid = int(scp.rpartition('.')[0].rpartition('_')[2]) # NOTE: critical for queue functionality
    array_writer = WriteHelper(f'ark,scp:{DEST}/fbanks_{pid}.ark,{DEST}/fbanks_{pid}.scp')
    score_writer = WriteHelper(f'ark,scp:{DEST}/scores_{pid}.ark,{DEST}/scores_{pid}.scp')
    label_writer = WriteHelper(f'ark,scp:{DEST}/labels_{pid}.ark,{DEST}/labels_{pid}.scp')
    target_writer = open(f'{DEST}/targets_{pid}.scp', 'w')
    embed_scp = kaldiio.load_scp(f'{EMBED}/dvectors.scp')

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
        fbanks = librosa.feature.melspectrogram(arr, 16000, n_fft=400,
                hop_length=160, n_mels=40).astype('float32').T[:-2]
        logfbanks = np.log10(fbanks + 1e-6)

        # use resemblyzer to generate filterbank slices for scoring..
        # NOTE: this piece of code was taken from the VoiceEncoder class and repurposed.
        # The method I implemented had some problems with utterances that were below 1.6s long
        wav = arr.copy()
        wav_slices, mel_slices = VoiceEncoder.compute_partial_slices(wav.size, rate, min_coverage)
        max_wave_length = wav_slices[-1].stop
        if max_wave_length >= wav.size:
            wav = np.pad(arr, (0, max_wave_length - wav.size), "constant")
        mels = librosa.feature.melspectrogram(wav, 16000, n_fft=400,
                hop_length=160, n_mels=40).astype('float32').T

        # create the fbanks slices...
        fbanks_sliced = np.array([mels[s] for s in mel_slices])
            
        # now generate n ground truth labels based on the gtruth and tstamps labels
        # where n is the number of feature frames we extracted
        n = logfbanks.shape[0]

        # NOTE: the timestamp doesn't really match the value of n. Keep an eye out..
        if tstamps[-1] < n*10:
            tstamps[-1] = n * 10

        # we need to extract partial embeddings for each utterance - each representing
        # a certain time window. Then those embeddings are compared with the target
        # speaker embedding via cosine similarity and this score is then used as
        # a feature.

        # randomly select a target speaker and compute his embedding
        n_speakers = gtruth.count('$') + 1

        # now, based on TS_DROPOUT, decide with a certain probability, whether to 
        # make a one speaker utterance without a target speaker to mitigate
        # overfitting for the target speaker class
        if TS_DROPOUT and n_speakers == 1:
            use_target = bool(np.random.randint(0, 3))
            if use_target:
                # target speaker
                which = 0
                spk_embed, spk_id = load_dvector(utt_id, which, embed_scp)

            else:
                which = -1 
                spk_embed, spk_id = load_dvector(utt_id, which, embed_scp, select_random=True)

        else:
            which = np.random.randint(0, n_speakers) 
            spk_embed, spk_id = load_dvector(utt_id, which, embed_scp)

        # send the datata to be processed on the gpu and retreive the result
        # prepare the fbanks tensor
        fbanks_tensor = torch.unsqueeze(torch.from_numpy(fbanks), 0)
        q_send.put((fbanks_tensor, torch.from_numpy(fbanks_sliced), pid))
        embeds_stream, embeds_slices = q_return.get()

        # convert to numpy arrays
        embeds_stream = embeds_stream.numpy().squeeze()
        embeds_slices = embeds_slices.numpy()

        # now generate three types of scores...
        # 1) with np.kron, just cloning the score value for each fbank slice (40 frames)
        # 2) linearly interpolate the slice scores with np.linspace
        # 3) just use the frame-level embeddings to score each frame individually

        # cosine similarities...
        scores_slices = np.array([ cos(spk_embed, cur_embed) for cur_embed in embeds_slices ])
        scores_stream = np.array([ cos(spk_embed, cur_embed) for cur_embed in embeds_stream ])

        try:
            # span the extracted scores to the whole utterance length
            # - the first 160 frames are the first score as the embedding is computed from
            #   a 1.6s long window
            # - all the other scores have frame_step frames between them

            scores_kron = np.kron(scores_slices[0], np.ones(160, dtype='float32'))
            if scores_slices.size > 1:
                scores_kron = np.append(scores_kron,
                        np.kron(scores_slices[1:], np.ones(frame_step, dtype='float32')))
            assert scores_kron.size >= n,\
                "Error: The scores_kron array was shorter than the actual feature vector."
            scores_kron = scores_kron[:n] # trim the score array

            # scores, linearly interpolated, starting from 0.5 every time
            # first 160 frames..
            scores_lin = np.kron(scores_slices[0], np.ones(160, dtype='float32'))
            # now the rest...
            for i, s in enumerate(scores_slices[1:]):
                scores_lin = np.append(scores_lin,
                        np.linspace(scores_slices[i], s, frame_step, endpoint=False))

            assert scores_lin.size >= n,\
                "Error: The scores_lin array was shorter than the actual feature vector."
            scores_lin = scores_lin[:n] # trim the score array

            # stack the three score arrays
            # legend: scores[0,:] -> scores_stream, 1 -> scores_slices, 2 -> scores_lin
            scores = np.stack((scores_stream, scores_kron, scores_lin))
            assert scores.shape[1] >= n,\
                "Error: The score array was shorter than the actual feature vector."

        except Exception as e:
            print(type(e), e, e.args)
            print(f"kron/lin: fbanks.shape {fbanks.shape}, arr.shape {arr.shape}")
            print(f"scores_stream {scores_stream.shape} embeds_slices {embeds_slices.shape}")
            print(f"scores_slices {scores_slices.shape} scores_kron {scores_kron.shape}")
            print(f"scores_lin {scores_lin.shape}")
            print("===============")
            continue

        # now relabel the ground truths to three classes... (ns, ntss, tss) -> {0, 1, 2}
        labels = np.ones(n, dtype=np.float32)
        stamp_prev = 0
        tstamps = tstamps // 10

        for (stamp, label) in zip(tstamps, gtruth):
            if label == '':
                labels[stamp_prev:stamp] = 0
            elif label == '$':
                which -= 1; # decrement the target speaker indicator
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
        label_writer(utt_id, labels)
        target_writer.write(f"{utt_id} {spk_id}\n") # write the target speaker too..

        # flush the results... just to be sure really...
        if i % 100 == 0:
            array_writer.fark.flush()
            array_writer.fscp.flush()
            score_writer.fark.flush()
            score_writer.fscp.flush()
            label_writer.fark.flush()
            label_writer.fscp.flush()
            target_writer.flush()

    # close all the scps..
    wav_scp.close()
    array_writer.close()
    score_writer.close()
    label_writer.close()
    target_writer.close()

def process_init(txt):
    """Prepare some shared memory objects for multiprocessing."""
    global text
    text = txt

if __name__ == '__main__':
    # command line arguments
    parser = ap.ArgumentParser(description="Extract speaker embeddings for the LibriSpeech dataset.")
    parser.add_argument('--kaldi_root', type=str, required=False, default=KALDI_ROOT,
            help="Specify the Kaldi pvad project root path")
    parser.add_argument('--data_root', type=str, required=False, default=DATA_ROOT,
            help="Specify the directory, where the target .scp files are situated")
    parser.add_argument('--dest_path', type=str, required=False, default=DEST,
            help="Specify the feature output directory")
    parser.add_argument('--embed_path', type=str, required=False, default=EMBED,
            help="Specify the path to the embeddings folder")
    parser.add_argument('--ts_dropout', action='store_true')
    parser.add_argument('--use_kaldi', action='store_true',
            help="Set this flag if the source dataset was augmented with Kaldi."\
                    "WARNING: If set, all other path specifications are counted "\
                    "as relative to --kaldi_root!")
    args = parser.parse_args()

    USE_KALDI = args.use_kaldi
    KALDI_ROOT = args.kaldi_root
    DATA_ROOT = args.data_root
    DEST = args.dest_path
    EMBED = args.embed_path
    TS_DROPOUT = args.ts_dropout

    # if USE_KALDI was set, move to KALDI_ROOT to gain access to MUSAN
    # and RIRS_NOISES
    if USE_KALDI:
        os.chdir(KALDI_ROOT)

    # first, load the utterance transcriptions
    with open(DATA_ROOT + '/text') as text_file:
        for utterance in text_file:
            utt_id, _, rest = utterance.partition(' ')
            labels, _, tstamps = rest.partition(' ')
            # save them as preprocessed tuples...
            txt[utt_id] = (labels.split(','),
                    np.array([int(float(stamp)*1000) for stamp in tstamps.split(' ')],
                        dtype='int32'))

    # get the file list for processing
    files = glob(DATA_ROOT + '/split_*.scp')
    files.sort()
    nj = len(files)

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

    # create the CPU worker pool - one for each scp file
    pool = mp.Pool(processes=nj, initializer=process_init, initargs=(txt,))
    pool.starmap(extract_features, zip(files, [q_send] * nj, q_return))
    pool.close()
    pool.join()
