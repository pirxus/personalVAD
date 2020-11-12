#!/usr/bin/python

import sys
import os
from multiprocessing import Process
import time
import re
import random
import math
import soundfile as sf
import numpy as np
from glob import glob

ALIGNED = True
MPROCESS = False
KEEP_TEXT = False
N = 6000 # The number of generated utterances
FILES_PER_DIR = 1000


# This function creates a list that contains tuples of the paths to the utterances
# and their actual full transcriptions and alignments
# (full_path, name, aligned_text, tstamps)
def parse_alignments(path):
    transcripts = [] 
    directory = path.rpartition('/')[0] + '/'
    f = open(path)
    for line in f.readlines():
        name = line.split(' ')[0]
        full_path = directory + name + '.flac'
        aligned_text = line.split(' ')[1][1:-1]
        tstamps = line.split(' ')[2][1:-2] # without the newline..

        # throw away the actual words if not needed...
        if not KEEP_TEXT:
            aligned_text = re.sub(r'[A-Z]+', 'W', aligned_text)

        # store the aligned transcript in the list
        transcripts.append((full_path, name, aligned_text, tstamps))

    f.close()
    return transcripts

def load_dataset_structure(root_dir):
    utterances = []
    with os.scandir(root) as speakers:
        for speaker in speakers:
            #print(f'Now processing the speaker {speaker.name}')

            # extract the paths to the individual files and their transcriptions
            transcripts = []
            for dirName, subdirList, fileList in os.walk(root + speaker.name):
                #print(f'Entering the {dirName} directory')
        
                # load the paths to the files and their transcriptions
                if fileList != []:
                    for f in glob(dirName + '/*.txt'):

                        # extract the transcriptions and store them in a list
                        if f.split('.')[-2] == 'alignment':
                            transcripts.extend(
                                    parse_alignments(f))

                        # NOTE: I ended up using the alignment files only, since currently
                        # they are all I need and contain all the information from the
                        # original transcript files.

            utterances.append((speaker.name, transcripts))

    return utterances

# Trims the end of the utterance so that the alignment timestamps for the other
# utterances can be offset by exactly n frames and so that the utterance's length is
# divisible by 10ms
def trim_utt_end(x, sr, tstamps):
    end_stamp = math.trunc(float(tstamps[-1]) * 100) / 100
    end = end_stamp * sr
    if x.size != end:
        assert (x.size > end), "Signal length was smaller than the end timestamp"
        x = x[:int(end-x.size)]

    return x, end_stamp


def generate_concatenations(dataset, dest, proc_name='', n=1300,
        wav_scp=None, utt2spk=None, text=None):
    if proc_name != '':
        print(f'process {proc_name} starting...')

    if wav_scp == None or utt2spk == None:
        # TODO: create the wav.scp file for multiprocessing cases...
        print("Wav.scp and utt2spk files have to be created for concatenations to be generated")
        sys.exit(2)

    iteration = 0 # split files into directories by 1000
    cur_dir = ''
    for iteration in range(n):
        if iteration % FILES_PER_DIR == 0:
            # create a new destination subdirectory
            cur_dir = dest + proc_name + str(iteration // FILES_PER_DIR) + '_concat' + '/'
            os.mkdir(cur_dir)

        # now randomly select the number of speaker utterances that are to be concatenated
        n_utter = np.random.randint(1, 4)
        speakers = random.sample(dataset, n_utter) # randomly select n speakers
        utterances = []
        for speaker in speakers: # and for each speaker select a random utterance
            utterances.append(random.choice(speaker[1]))

        data = np.array([])
        file_name = ''
        transcript = ''
        alignment = ''

        if ALIGNED is True:
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

        else:

            # concatenate the waveforms, save the new concatenated file and its
            # transcription
            for utterance in utterances:
                x, sr = sf.read(utterance[0])
                assert (sr == 16000), f'Invalid source audio sample rate {sr}'
                data = np.append(data, x)
                file_name += utterance[1] + '_'
                transcript += utterance[2]

            file_name = file_name[:-1]
            transcript = transcript[:-1]

        # save the new file and transcription
        sf.write(cur_dir + file_name + '.flac', data, 16000)
        #with open(cur_dir + file_name + '.txt', 'w') as txt: #TODO: throw away?
        #    if ALIGNED: txt.write(transcript + ' ' + alignment + '\n')
        #    else: txt.write(transcript + '\n')

        # and write an entry to our wav.scp, utt2spk and text files
        wav_scp.write(file_name + ' flac -d -c -s ' + cur_dir + file_name + '.flac |\n')
        utt2spk.write(file_name + ' ' + file_name + '\n')
        if ALIGNED: text.write(file_name + ' ' + transcript + ' ' + alignment + '\n')

if __name__ == '__main__':

    if len(sys.argv) != 3:
        print("Incorrect number of parameters for the concatenation script")
        print("Please specify the path (i.e. data/LibriSpeech/dev-clean) to")
        print("the dataset's root direcotory as the first parameter and the name")
        print("of the destination folder that will contain the generated utterances.")
        sys.exit(0)
    else:
        root = sys.argv[1]
        dest = sys.argv[2]
        if root[-1] != '/': root += '/'
        if dest[-1] != '/': dest += '/'
        # check if the destination path is an absolute path
        if not os.path.isabs(dest):
            print("The destination folder path has to be specified as absolute")
            sys.exit(1)

    dataset = load_dataset_structure(root)

    # now create the destination directory
    if os.path.exists(dest):
        if not os.path.isdir(dest) or os.listdir(dest):
            print('The specified destination folder is an existing file/directory')
            sys.exit()
    else:
        try:
            os.mkdir(dest)
        except OSError:
            print(f'Could not create destination directory {dest}')
            sys.exit(1)

    # generate the dataset - TODO: multiprocessing and wav.scp
    if MPROCESS == True:
        gen1 = Process(target=generate_concatenations, args=(dataset, dest, 'a', N/4,))
        gen2 = Process(target=generate_concatenations, args=(dataset, dest, 'b', N/4,))
        gen3 = Process(target=generate_concatenations, args=(dataset, dest, 'c', N/4,))
        gen4 = Process(target=generate_concatenations, args=(dataset, dest, 'd', N/4,))
        gen1.start(); gen2.start(); gen3.start(); gen4.start()
    else:

        # create the wav.scp file
        with open(dest + 'wav.scp', 'w') as wav_scp, open(dest + 'utt2spk', 'w') as utt2spk, open(dest + 'text', 'w') as text:
            # and generate our dataset
            generate_concatenations(dataset, dest, n=N, wav_scp=wav_scp, utt2spk=utt2spk, text=text)

