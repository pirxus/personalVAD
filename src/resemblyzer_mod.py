import torch
import numpy as np
import librosa
from kaldiio import ReadHelper
from resemblyzer import VoiceEncoder
from extract_features_embed import preprocess_wav, get_speaker_embedding,\
    Mode, embedding_cache, cos
from numpy.lib.stride_tricks import sliding_window_view
import python_speech_features as psf
import random

DATA = '/home/pirx/Devel/bp/personalVAD/kaldi/egs/pvad/data/clean'
EMBED_PATH = 'data/embeddings/'
text = {}
TS_DROPOUT = True
CACHE_DVECTORS = True
GEN_SPK_EMBEDDINGS = False

class VoiceEncoderMod(VoiceEncoder):
    def __init__(self):
        super().__init__()

    def forward_stream(self, x, hidden):
        """Modified to return embeddings for all inputs and to be able to
        remember the hidden state between batches.."""
        out, hidden = self.lstm(x, hidden)
        embeds_raw = self.relu(self.linear(out[:,:]))
        norm = torch.norm(embeds_raw, dim=2, keepdim=True)
        return embeds_raw / norm, hidden

if __name__ == '__main__':
    model = VoiceEncoderMod()
    encoder = VoiceEncoder()
    wav_scp = ReadHelper(f'scp:{DATA}/wav.scp')
    text_file = open(DATA + '/text')
    for utterance in text_file:
        utt_id, _, rest = utterance.partition(' ')
        labels, _, tstamps = rest.partition(' ')
        # save them as preprocessed tuples...
        text[utt_id] = (labels.split(','),
                np.array([int(float(stamp)*1000) for stamp in tstamps.split(' ')], dtype=np.int8))

    for utt_id, (sr, arr) in wav_scp:
        print(f"utt_id: {utt_id}")

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
        rate = 2.5
        samples_per_frame = 160
        frame_step = int(np.round((16000 / rate) / samples_per_frame))

        x = arr
        _, utt_embeds, slices = encoder.embed_utterance(x, return_partials=True,
                rate=rate, min_coverage=0.5)
        utt_embeds = utt_embeds[:-1]
        with torch.no_grad():
            fbanks = np.expand_dims(fbanks, 0)
            embeds_mod, _ = model.forward_stream(torch.from_numpy(fbanks).to(model.device), None)
            embeds_mod = embeds_mod.cpu().numpy().squeeze()

            fbanks = fbanks.squeeze()
            # slice up the fbanks
            print(fbanks.shape)
            fbanks_sliced = sliding_window_view(fbanks, (160, 40)).squeeze(axis=1)[::frame_step].copy()
            #print(fbanks_sliced[::40].shape)
            print(utt_embeds.shape)
            tensor = torch.from_numpy(fbanks_sliced)
            print(tensor.shape)
            test_embed = model(torch.from_numpy(fbanks_sliced).to(model.device)).cpu().numpy()
            print(test_embed.shape)


        # compute the cosine similarity between the partial embeddings and the target
        # speaker embedding
        scores_raw = np.array([ cos(spk_embed, cur_embed) for cur_embed in utt_embeds ])
        scores_mod = np.array([ cos(spk_embed, cur_embed) for cur_embed in embeds_mod ])
        scores_new = np.array([ cos(spk_embed, cur_embed) for cur_embed in test_embed ])
        print("embed shape: ", utt_embeds.shape, test_embed.shape)

        # span the extracted scores to the whole utterance length
        # - the first 160 frames are the first score as the embedding is computed from
        #   a 1.6s long window
        # - all the other scores have frame_step frames between them and the last one
        #   is stretched to match the logfbanks length
        scores = np.append(np.kron(scores_raw[0], np.ones(160, dtype=scores_raw.dtype)),
                np.kron(scores_raw[1:-1], np.ones(frame_step, dtype=scores_raw.dtype)))
        scores = np.append(scores, np.kron(scores_raw[-1],
            np.ones(logfbanks.shape[0] - scores.size, dtype=scores_raw.dtype)))

        print(scores.size, logfbanks.shape)
        assert scores.size >= logfbanks.shape[0],\
            "Error: The score array was shorter than the actual feature vector."

        scores_new_kron = np.append(np.kron(scores_new[0], np.ones(160, dtype=scores_raw.dtype)),
                np.kron(scores_new[1:-1], np.ones(frame_step, dtype=scores_new.dtype)))
        scores_new_kron = np.append(scores_new_kron, np.kron(scores_new[-1],
            np.ones(logfbanks.shape[0] - scores_new_kron.size, dtype=scores_new.dtype)))
        
        print(f"SUM: {(scores - scores_new_kron).sum()}")

        #print(scores[:162], scores_mod[:162])

        # scores, linearly interpolated, starting from 0.5 every time
        # first 160 frames..
        lin_scores = np.linspace(0.5, scores_new[0], 160, endpoint=False)
        # now the rest...
        for i, s in enumerate(scores_new[1:]):
            lin_scores = np.append(lin_scores,
                    np.linspace(scores_new[i], s, frame_step, endpoint=False))
        print(logfbanks.shape[0], lin_scores.size, logfbanks.shape[0] - lin_scores.size)
        lin_scores = np.append(lin_scores, np.kron(scores_new[-1],
            np.ones(logfbanks.shape[0] - lin_scores.size, dtype=scores_new.dtype)))

        print(f"Scores size: {scores.size} {scores_mod.size} {lin_scores.size} {logfbanks.shape[0]}")

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

    text_file.close()
    wav_scp.close()
