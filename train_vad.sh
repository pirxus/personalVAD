#!/bin/bash

# This is a launching script used to train one of the vad architectures.
#

if [ -e $1 ]; then
  echo "Please, in order to train a model, specify the target architecture."
  exit 0
else
  arch=$1
fi


if [[ $arch == "vad" ]]; then
  echo "Beginning training the classic vad architecture."
  python src/vad.py
fi

if [[ $arch == "et" ]]; then
  echo "Beginning training the ET vad architecture."
  python src/vad_et.py
fi

if [[ $arch == "st" ]]; then
  echo "Beginning training the ST vad architecture."
  python src/vad_st.py
fi

if [[ $arch == "set" ]]; then
  echo "Beginning training the SET vad architecture."
  python src/vad_set.py
fi
