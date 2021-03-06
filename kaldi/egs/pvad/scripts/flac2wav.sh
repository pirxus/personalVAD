#!/usr/bin/bash
for i in *.flac;
do
  name=`echo "$i" | cut -d'.' -f1`
  ffmpeg -i "$i" "${name}.wav"
done
