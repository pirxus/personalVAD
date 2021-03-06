#!/bin/bash

red=`tput setaf 1`
green=`tput setaf 2`
reset=`tput sgr0`

train_cmd="run.pl"
decode_cmd="run.pl"
stage=0

if [ ! -d "RIRS_NOISES" ]; then
  # Download and unzip the rirs 
  wget --no-check-certificate http://www.openslr.org/resources/28/rirs_noises.zip
  unzip rirs_noises.zip
fi

if [ $stage -le 0 ]; then
  echo "${green}Reverberating the dataset...${reset}"
   # Make a version with reverberated speech
  rvb_opts=()
  rvb_opts+=(--rir-set-parameters "0.5, RIRS_NOISES/simulated_rirs/smallroom/rir_list")
  rvb_opts+=(--rir-set-parameters "0.5, RIRS_NOISES/simulated_rirs/mediumroom/rir_list")

  # Reverberate our data folder
  steps/data/reverberate_data_dir.py \
    "${rvb_opts[@]}" \
    --prefix "rev" \
    --speech-rvb-probability 0.6 \
    --pointsource-noise-addition-probability 0 \
    --isotropic-noise-addition-probability 0 \
    --num-replications 1 \
    --source-sampling-rate 16000 \
    data/clean data/reverb
fi

# generate the wav files
python ./scripts/scp2flac.py data/reverb/wav.scp data/reverb/flac

# move the data to the root folder above...
mkdir -p ../../../data/concat
mv data/reverb/text data/reverb/text_flac
mv data/reverb/flac/* data/reverb/text_flac ../../../data/concat
mv data/clean/*_concat data/clean/text ../../../data/concat

exit 0

#cp data/swbd_sre/vad.scp data/swbd_sre_reverb/
#utils/copy_data_dir.sh --utt-suffix "-reverb" data/swbd_sre_reverb data/swbd_sre_reverb.new
#rm -rf data/swbd_sre_reverb
#mv data/swbd_sre_reverb.new data/swbd_sre_reverb

# extract 40-dimensional log-fbank features
if [ $stage -le 1 ]; then
  echo "${green}Extracting filterbank features...${reset}"

  #mkdir -p data/fbank
  fbankdir=fbank
  for part in clean reverb; do
    steps/make_fbank.sh  --cmd "$train_cmd" --nj 4 data/$part exp/make_fbank/$part $fbankdir
    steps/compute_cmvn_stats.sh $part exp/make_fbank/$part $fbankdir
  done  
fi

# 
if [ $stage -le 2 ]; then
  echo "${green}Merging extracted features...${reset}"
fi
