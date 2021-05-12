#!/usr/bin/python

"""@package concatenate_utterances

Author: Simon Sedlacek
Email: xsedla1h@stud.fit.vutbr.cz

This module is used to generate the concatenated utterance dataset from the
LibriSpeech dataset.

"""

import sys
import os
from multiprocessing import Process
import argparse as ap
import time
import re
import random
import math
import soundfile as sf
import numpy as np
from glob import glob

KEEP_TEXT = False
N = 10000 # The number of generated utterances
FILES_PER_DIR = 2000
FLAC = True # if true, flac will be used to load the waveform. If false, sox will be used.
UNIQUE = True # if true, each source utterance can only be used once...

SETS = ['dev-clean', 'dev-other', 'test-clean', 'test-other']
wav_scp_prefix = 'data/clean/'

def parse_alignments(path):
    """Parse the LibriSpeech alignments files.

    Create a list containing tuples of paths to the utterances
    (full_path, utterance_name, aligned_text, tstamps), and their
    actual full transcriptions and time alignments.

    Args:
        path (str): Path to the target alignment file.

    Returns:
        list of tuples: tuples containing:
            full_path (str): Path to the utterance audio file.
            name (str): The utterance id.
            aligned_text (str): The transcript of the utterance conwerted to a combination
                of 'W' denoting words and ' ' denoting silence.
            tstamps (str): The utterance's text alignment time stamps.
    """

    transcripts = [] 
    directory = path.rpartition('/')[0] + '/'
    with open(path) as f:
        for line in f.readlines():
            name = line.split(' ')[0]
            full_path = directory + name + '.flac'
            aligned_text = line.split(' ')[1][1:-1]
            tstamps = line.split(' ')[2][1:-2] # without the newline..

            # throw away the actual words if not needed...
            if not KEEP_TEXT:
                aligned_text = re.sub(r"[A-Z']+", 'W', aligned_text)

            # store the aligned transcript in the list
            transcripts.append((full_path, name, aligned_text, tstamps))

    return transcripts

def load_dataset_structure(root, sets):
    """Load the structure of the LibriSpeech dataset.

    Args:
        root (str): The path to the root of the LibriSpeech dataset, e.g. 'data/LibriSpeech/'
        sets (list of str): List of LibriSpeech subsets from which to generate the resulting
            concatenated utterances.

    Returns:
        list of tuples: tuples containing:
            speaker_name (str): The id of the speaker.
            transcripts (list of tuples): All transcripts obtained for current
                speaker, as returned by the parse_alignments function.
    """

    utterances = []
    for subset in sets:
        with os.scandir(root + subset) as speakers:
            for speaker in speakers:
                #print(f'Now processing the speaker {speaker.name}')

                # extract the paths to the individual files and their transcriptions
                transcripts = []
                for dirName, subdirList, fileList in os.walk(root + subset + '/' + speaker.name):
                    #print(f'Entering the {dirName} directory')
            
                    # load the paths to the files and their transcriptions
                    if fileList != []:
                        for f in glob(dirName + '/*.txt'):

                            # extract the transcriptions and store them in a list
                            if f.split('.')[-2] == 'alignment':
                                transcripts.extend(
                                        parse_alignments(f))

                utterances.append((speaker.name, transcripts))

    return utterances

def trim_utt_end(x, sr, tstamps):
    """Trim the end of the utterance.

    Trim the end of the utterance so that the alignment timestamps for the other
    utterances can be offset by exactly n frames and so that the utterance's length
    is divisible by 10ms.

    Args:
        x (np.array): The source waveform to be trimmed.
        sr (int): Sample rate of x.
        tstamps (list of strings): The time stamp array corresponding to x.
    """

    end_stamp = math.trunc(float(tstamps[-1]) * 100) / 100
    end = end_stamp * sr
    if x.size != end:
        assert (x.size > end), "Signal length was smaller than the end timestamp"
        x = x[:int(end-x.size)]

    return x, end_stamp


def generate_concatenations(dataset, dest, n, wav_scp, utt2spk, text):
    """Generate the concatenated utterance dataset.

    Args:
        dataset (list of tuples): The loaded dataset structure as returned by the
            load_dataset_structure function.
        dest (str): Generated dataset destination path.
        wav_scp (file): The wav.scp file, which will describe the whole generated
            dataset.
        utt2spk (file): A Kaldi-specific file containing information about which speaker
            is present in each utterance. It is generated only as a placeholder for some
            Kaldi scripts and has no real use down the feature extraction pipeline.
        text (file): The file which will contain the aligned transcripts of each resulting
            utterance in the dataset.

    """

    random.seed()

    if wav_scp == None or utt2spk == None:
        print("Wav.scp and utt2spk files have to be created for concatenations to be generated")
        sys.exit(2)

    iteration = 0 # split files into directories by 1000
    cur_dir = ''
    scp_path = ''
    for iteration in range(n):
        if iteration % FILES_PER_DIR == 0:
            # create a new destination subdirectory
            scp_path = str(iteration // FILES_PER_DIR) + '_concat' + '/'
            cur_dir = dest + scp_path
            os.mkdir(cur_dir)

        # now randomly select the number of speaker utterances that are to be concatenated
        n_utter = np.random.randint(1, 4)
        try:
            speakers = random.sample(dataset, n_utter) # randomly select n speakers
        except ValueError:
            print("Ran out of utterances, ending...")
            # no utterances left in the dataset, just leave...
            return

        utterances = []
        for speaker in speakers: # and for each speaker select a random utterance
            utt = random.choice(speaker[1])
            utterances.append(utt)
            if UNIQUE:
                # delete the used utterance
                speaker[1].remove(utt)
                if not speaker[1]:
                    # also delete the speaker if there are no utterances left
                    dataset.remove(speaker)

        data = np.array([])
        file_name = ''
        transcript = ''
        alignment = ''

        # concatenate the waveforms, save the new concatenated file and its
        # transcription and alignment timestamps
        tstamps = [] 
        prev_end_stamp = 0
        for utterance in utterances:
            x, sr = sf.read(utterance[0])
            assert (sr == 16000), f'Invalid source audio sample rate {sr}'
            stamps = utterance[3].split(',')
            x, end_stamp = trim_utt_end(x, sr, stamps)
            data = np.append(data, x)

            # check if the waveform isn't corrupt -__-
            if x.size < 100:
                data = np.array([])
                break

            # offset the timestamps
            if tstamps != []:
                tstamps.pop()
                tstamps.extend(
                        [str(round(float(stamp) + prev_end_stamp, 2))
                            for stamp in stamps])
            else:
                tstamps = stamps

            prev_end_stamp += end_stamp
            tstamps[-1] = str(round(prev_end_stamp))

            file_name += utterance[1] + '_'
            transcript += utterance[2] + '$'

        alignment = ' '.join(tstamps)
        file_name = file_name[:-1]
        transcript = transcript[:-1]


        # check if the waveform isn't corrupt -__- skipp
        if data.size < 100:
            print(f"An utterance with path {file_name} was too short.")
            continue

        # save the new file
        sf.write(cur_dir + file_name + '.flac', data, 16000)

        # and write an entry to our wav.scp, utt2spk and text files
        if FLAC:
            wav_scp.write(file_name + ' flac -d -c -s ' + wav_scp_prefix +
                    scp_path + file_name + '.flac |\n')
        else: # use sox instead of flac in the wav.scp
            wav_scp.write(file_name + ' sox ' + wav_scp_prefix + scp_path +
                    file_name + '.flac -b 16 -e signed -c 1 -t wav - |\n')

        utt2spk.write(file_name + ' ' + file_name + '\n')
        text.write(file_name + ' ' + transcript + ' ' + alignment + '\n')


if __name__ == '__main__':
    parser = ap.ArgumentParser(description="Generate LibriSpeech concatenations.",
            usage="concatenate_utterances.py [options]")
    parser.add_argument('--libri_root', type=str, required=True,
            help="Specify the path to the LibriSpeech dataset")
    parser.add_argument('--concat_dir', type=str, required=True,
            help="Specify the output folder")
    parser.add_argument('--count', type=int, default=N,
            help="Generated utterance count")
    parser.add_argument('--scp_prefix', type=str, default=wav_scp_prefix,
            help="wav.scp path prefix")
    parser.add_argument('parts', type=str, nargs='*',
            help="Specify which LibriSpeech folders to process")
    args = parser.parse_args()

    root = args.libri_root
    dest = args.concat_dir
    N = args.count
    wav_scp_prefix = args.scp_prefix
    SETS = args.parts

    if root[-1] != '/': root += '/'
    if wav_scp_prefix[-1] != '/': wav_scp_prefix += '/'
    if dest[-1] != '/': dest += '/'

    # check if the destination path is an absolute path
    if not os.path.isabs(dest):
        print("The destination folder path has to be specified as absolute")
        sys.exit(1)

    # load the dataset structure
    dataset = load_dataset_structure(root, SETS)

    # now create the destination directory
    if os.path.exists(dest):
        if not os.path.isdir(dest) or os.listdir(dest):
            print('The specified destination folder is an existing file/directory')
            sys.exit(1)
    else:
        try:
            os.mkdir(dest)
        except OSError:
            print(f'Could not create destination directory {dest}')
            sys.exit(1)

    # create the wav.scp, utt2spk, text files...
    with open(dest + '/wav.scp', 'w') as wav_scp, \
         open(dest + '/utt2spk', 'w') as utt2spk, \
         open(dest + '/text', 'w') as text:
        # and generate our dataset
        generate_concatenations(dataset, dest, N, wav_scp, utt2spk, text)

