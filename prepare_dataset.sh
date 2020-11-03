#!/bin/bash

# This script takes care of creating the dataset that we use for vad/pvad training.
#
# First, concatenated utterances are generated from the Librispeech dataset along with kaldi
# recipes for reverberation and augmentation.
#
# Then features and ground truth labels are extracted for each utterance.

# some colors..
red=`tput setaf 1`
green=`tput setaf 2`
reset=`tput sgr0`

concat_dir=/home/pirx/Devel/bp/personalVAD/kaldi/egs/pvad/data/clean
kaldi_root=kaldi/egs/pvad

# generate the concatenations
echo "${green}Generating concatenated utterances...${reset}"
python src/concatenate_utterances.py data/LibriSpeech/dev-clean $concat_dir

# move to the kaldi directory
echo "${green}Moving to kaldi directory...${reset}"
cd $kaldi_root

# make sure there are no duplicate entries in the wav.scp and utt2spk files
sort $concat_dir/utt2spk | uniq -u >$concat_dir/tmp && mv $concat_dir/tmp $concat_dir/utt2spk
sort $concat_dir/wav.scp | uniq -u >$concat_dir/tmp && mv $concat_dir/tmp $concat_dir/wav.scp
utils/fix_data_dir.sh data/clean

# create the spk2utt for the clean data
utils/utt2spk_to_spk2utt.pl data/clean/utt2spk > data/clean/spk2utt

# reverberate the data and extract log mel-filterbank features
data_prep.sh
