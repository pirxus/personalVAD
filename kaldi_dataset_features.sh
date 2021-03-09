#!/bin/bash

# =======
# This script handles all the data preparation steps prior to feature extraction
# 
# STEPS:
# 1) generate the concatenations and the ground truth labels
# 2) 
#
#
#
#


# first, some data-prep flags..

augment=false
reverberate=true

# some colors..
red=`tput setaf 1`
green=`tput setaf 2`
reset=`tput sgr0`
nj_features=4

if [ -e $1 ]; then
  echo "Please specifiy the data preparation stage."
  exit 0
else
  stage=$1
fi

repo_root=/home/pirx/Devel/bp/personalVAD
concat_dir=$repo_root/kaldi/egs/pvad/data/clean
kaldi_root=$repo_root/kaldi/egs/pvad

if [ $stage -le 0 ]; then

  # generate the concatenations
  echo "${green}Generating concatenated utterances...${reset}"
  python src/concatenate_utterances.py data/LibriSpeech $concat_dir

fi

if [ $stage -le 1 ]; then

  # move to the kaldi directory
  echo "${green}Moving to kaldi directory...${reset}"
  cd $kaldi_root

  # make sure there are no duplicate entries in the wav.scp utt2spk and text files
  #sort $concat_dir/utt2spk | uniq -u >$concat_dir/tmp && mv $concat_dir/tmp $concat_dir/utt2spk
  #sort $concat_dir/wav.scp | uniq -u >$concat_dir/tmp && mv $concat_dir/tmp $concat_dir/wav.scp
  #sort $concat_dir/text | uniq -u >$concat_dir/tmp && mv $concat_dir/tmp $concat_dir/text
  utils/fix_data_dir.sh data/clean

  # create the spk2utt for the clean data
  utils/utt2spk_to_spk2utt.pl data/clean/utt2spk > data/clean/spk2utt

fi

if [ $stage -le 2 ]; then
  cd $kaldi_root

  # reverberate the data, augment the data
  ./data_prep_new.sh 0
  #cd ../../../data/concat
  #cat text_flac >> text

  # TODO: fix the reverberation prefix so that utterances with the same transcriptions are
  # grouped together...
fi

if [ $stage -le 3 ]; then # feature extraction
  cd $kaldi_root

  # first, split the wav.scp into multiple files to allow multiprocessing
  cd data/augmented
  split -n l/$nj_features --additional-suffix .scp -d wav.scp split_
  
  cd $repo_root
  # extract features, dest: $kaldi_root/data/features
  python3 src/extract_features_kaldi.py

  # combine back the feature scps
  cd $kaldi_root/data/features/
  cat fbanks_*.scp > fbanks.scp
  cat embed_*.scp > embed.scp
  cat scores_*.scp > scores.scp
  cat labels_*.scp > labels.scp
  cat labels_vad_*.scp > labels_vad.scp
  cd $repo_root
fi

if [ $stage -le 4 ]; then # split into train and test
  cd $kaldi_root/data/
  mkdir -p train
  mkdir -p test
  cd features
  for name in fbanks embed scores labels labels_vad
  do
    awk 'NR % 10 == 0' $name.scp > ../test/$name.scp
    cp $name.scp ../train/$name.scp #TODO: remove the lines
  done
  cd $repo_root
  #todo
fi
