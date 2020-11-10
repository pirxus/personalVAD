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

if [ -e $1 ]; then
  stage=2
else
  stage=$1
fi

repo_root=/home/prix/Devel/bp/personalVAD
concat_dir=/home/pirx/Devel/bp/personalVAD/kaldi/egs/pvad/data/clean
kaldi_root=/home/pirx/Devel/bp/personalVAD/kaldi/egs/pvad


if [ $stage -le 0 ]; then

  # generate the concatenations
  echo "${green}Generating concatenated utterances...${reset}"
  python src/concatenate_utterances.py data/LibriSpeech/dev-clean $concat_dir

fi

if [ $stage -le 1 ]; then

  # move to the kaldi directory
  echo "${green}Moving to kaldi directory...${reset}"
  cd $kaldi_root

  # make sure there are no duplicate entries in the wav.scp utt2spk and text files
  sort $concat_dir/utt2spk | uniq -u >$concat_dir/tmp && mv $concat_dir/tmp $concat_dir/utt2spk
  sort $concat_dir/wav.scp | uniq -u >$concat_dir/tmp && mv $concat_dir/tmp $concat_dir/wav.scp
  sort $concat_dir/text | uniq -u >$concat_dir/tmp && mv $concat_dir/tmp $concat_dir/text
  utils/fix_data_dir.sh data/clean

  # create the spk2utt for the clean data
  utils/utt2spk_to_spk2utt.pl data/clean/utt2spk > data/clean/spk2utt

  # reverberate the data and extract log mel-filterbank features
  data_prep.sh
  cd ../../../

fi

# move the extracted feature scp files to the data folder and split them into train and test
# directories

#if [ $stage -le 2 ]; then
#
#  echo "${green}Moving the feature scp files...${reset}"
#  mkdir -p data/features/train
#  mkdir -p data/features/test
#
#  cp "kaldi/egs/pvad/fbanks/"*".scp" "data/features/train/"
#
#  for filename in "data/features/train/"*; do
#    if [[ "$filename" == *".1."* ]]; then
#      mv "$filename" data/features/test
#    fi
#  done
#fi
