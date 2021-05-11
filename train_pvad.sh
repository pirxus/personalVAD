#!/bin/bash
#
# File: train_pvad.sh
# Author: Simon Sedlacek
# Email: xsedla1h@stud.fit.vutbr.cz
#
# This is a demonstration script for personal VAD training meant to show examples
# of how to begin training the different personal VAD architectures.
#
# The dataset used for training is the sample validation dataset provided for model
# evaluation. To properly train the models, a new training dataset has to be 
# generated first. When it is ready, feel free take inspiration in this script to
# write your own training script. The scripts will not save the models to avoid cluttering
# the repository.
#

#================ EDIT HERE =========================

# here, the scoring method for the st and set architectures can be specified
# 0 (baseline), 1 (PC), 2 (LI)
st_score_type=1
set_score_type=1

#===================================================

# some colors..
red=`tput setaf 1`
green=`tput setaf 2`
yellow=`tput setaf 3`
reset=`tput sgr0`

if [ -e $1 ]; then
  echo "Please, in order to train a model, specify the target architecture."
  echo "Choose one of: vad, et, st, set"
  exit 0
else
  arch=$1
fi

# move to the eval directory..
BASE=$PWD
TRAIN_NAME=test_clean
TEST_NAME=test_clean
cd data/eval_dir

if [[ $arch == "vad" ]]; then
  echo "${green}Beginning training the classic vad architecture.${reset}"
  python $BASE/src/vad.py --train_dir data/$TRAIN_NAME --test_dir data/$TEST_NAME \
    --nsave_model
fi

if [[ $arch == "et" ]]; then
  echo "${green}Beginning training the ET vad architecture.${reset}"
  python $BASE/src/vad_et.py --nsave_model --embed_path embeddings \
    --train_dir data/$TRAIN_NAME --test_dir data/$TEST_NAME
fi

if [[ $arch == "st" ]]; then
  echo "${green}Beginning training the ST vad architecture.${reset}"
  echo "${yellow}Using scoring method" $st_score_type "${reset}"
  python $BASE/src/vad_st.py --nsave_model --score_type $st_score_type \
    --train_dir data/$TRAIN_NAME --test_dir data/$TEST_NAME
fi

if [[ $arch == "set" ]]; then
  echo "${green}Beginning training the SET vad architecture.${reset}"
  echo "${yellow}Using scoring method" $set_score_type "${reset}"
  python $BASE/src/vad_set.py --nsave_model --embed_path embeddings --score_type $set_score_type \
    --train_dir data/$TRAIN_NAME --test_dir data/$TEST_NAME
fi
