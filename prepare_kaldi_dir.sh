#!/bin/bash
#
# File: prepare_kaldi_dir.sh
# Author: Simon Sedlacek
# Email: xsedla1h@stud.fit.vutbr.cz
#
# NOTE: before running this script, please make SURE that you have kaldi downloaded
# and compiled in the personal VAD project root folder (the kaldi/ folder should be on
# the same level as the data/ and src/ directories among others..) 
# 
# This script will create a kaldi project folder for personal VAD data augmentation
# purposes, copy and create necessary symlinks.
#
# The created kaldi project folder will be (relative to the pvad repo root) kaldi/egs/pvad/.
#
# After running this script, data augmentation will be possible
#


# first, check if the kaldi folder exists..
if [[ ! -d "kaldi/" ]]; then
  echo "the 'kaldi/' folder does not exist."
  echo "Please download and install kaldi to the project root directory."
  exit 1
fi

cd kaldi/egs

cp -r ../../src/kaldi/egs/pvad .

cd pvad
# create symlinks for kaldi binaries and utilities
ln -s ../wsj/s5/steps .
ln -s ../wsj/s5/utils .
ln -s ../../src .
