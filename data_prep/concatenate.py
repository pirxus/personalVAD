#!/usr/bin/python

import sys
import os
import random
import soundfile as sf
import numpy as np

# This function creates a list that contains tuples of the paths to the utterances
# and their actual full transcriptions (full_path, name, text)
def parse_transcriptions(path):
    transcripts = [] 
    directory = path.rpartition('/')[0] + '/'
    f = open(path)
    for line in f.readlines():
        name = line.split(' ')[0]
        full_path = directory + name + '.flac'
        text = line.partition(' ')[2][:-1]
        transcripts.append((full_path, name, text)) # store the transcript in the list

    f.close()
    return transcripts

def load_dataset_structure(root_dir):
    utterances = []
    with os.scandir(root) as speakers:
        for speaker in speakers:
            print(f'Now processing the speaker {speaker.name}')

            # extract the paths to the individual files and their transcriptions
            transcripts = []
            for dirName, subdirList, fileList in os.walk(root + speaker.name):
                print(f'Entering the {dirName} directory')
        
                # load the paths to the files and their transcriptions
                if fileList != []:
                    for f in fileList:

                        # extract the transcriptions and store them in a list
                        if f.split('.')[1] == 'trans':
                            transcripts += parse_transcriptions(dirName + '/' + f)

            utterances.append((speaker.name, transcripts))

    return utterances

def generate_concatenations(dataset, dest, n=1300):

    # first, create the destination directory
    if os.path.exists(dest):
        print('The specified destination folder is an existing file/directory')
        return
    else:
        try:
            os.mkdir(dest)
        except OSError:
            print('Could not create destination directory %s' % dest)


    iteration = 0 # split files into directories by 1000
    cur_dir = ''
    while iteration < n:
        if iteration % 1000 == 0:
            # create a new destination subdirectory
            cur_dir = dest + str(iteration) + '/'
            os.mkdir(cur_dir)

        # now randomly select the number of speaker utterances that are to be concatenated
        n_utter = np.random.randint(1, 4)
        speakers = random.sample(dataset, n_utter) # randomly select n speakers
        utterances = []
        for speaker in speakers: # and for each speaker select a random utterance
            utterances.append(random.choice(speaker[1]))

        # concatenate the waveforms, save the new concatenated file and its transcription
        data = np.array([])
        sr = 16000
        transcript = ''
        file_name = ''
        for utterance in utterances:
            x, sr = sf.read(utterance[0])
            data = np.append(data, x)
            file_name += utterance[1] + '_'
            transcript += utterance[2] + ' '

        file_name = file_name[:-1]
        transcript = transcript[:-1] + '\n' # don't forget the newline

        # save the new file and transcription
        sf.write(cur_dir + file_name + '.flac', data, sr)
        with open(cur_dir + file_name + '.txt', 'w') as txt:
            txt.write(transcript)
            txt.close()
        iteration += 1

if __name__ == '__main__':

    if len(sys.argv) != 3:
        print("Incorrect number of parameters for the concatenation script")
        print("Please specify the path (i.e. ~/data/LibriSpeech/dev-clean) to")
        print("the dataset's root direcotory as the first parameter and the name")
        print("of the destination folder that will contain the generated utterances.")
        sys.exit(0)
    else:
        root = sys.argv[1]
        dest = sys.argv[2]
        if root[-1] != '/': root += '/'
        if dest[-1] != '/': dest += '/'

    dataset = load_dataset_structure(root)
    generate_concatenations(dataset, dest, n=15)
