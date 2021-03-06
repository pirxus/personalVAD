#!/usr/bin/python
import sys
import os

# first argument -> scp file from which we need to generate the flac files.
# second argument -> path to a destination folder

FILES_PER_DIR = 1000
DELETE_ORIGINAL = False

if len(sys.argv) != 3:
    print("Please provide the target scp file and the destination path")
    sys.exit(0)
else:
    scp_path = sys.argv[1]
    dest = sys.argv[2]

# create the destination directory
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

# now read the scp file and for every FILES_PER_DIR utterances create a new directory
with open(scp_path) as scp:
    cur_dir = ''
    for i, entry in enumerate(scp):
        if i % FILES_PER_DIR == 0:
            # create a new destination subdirectory
            cur_dir = dest + '/' + str(i // FILES_PER_DIR) + '_flac' + '/'
            os.mkdir(cur_dir)

        # now alter and execute the command to get the flac file
        utt_id, _, command = entry.partition(' ')
        command = command[:-1] + f" flac -s - -o {cur_dir}{utt_id}.flac"

        # and delete the source utterance..
        if DELETE_ORIGINAL:
            pass

        # and execute the command to get the flac file...
        os.system(command)
