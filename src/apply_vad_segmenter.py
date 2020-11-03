#!/usr/bin/env python3
# Copyright 2020  Brno University of Technology (author: Karel Vesely)
# Licensed under the Apache License, Version 2.0 (the "License")

from __future__ import print_function

import sys, os, re

import numpy as np
import matplotlib
matplotlib.use('Agg') # no X server required,
import matplotlib.pyplot as plt

try:
    import kaldi_io
except:
    print("Missing kaldi_io, please install it by: 'python3 -m pip install kaldi_io'")
    raise


# Show the command line log,
print(' '.join(sys.argv))

# Option parsing,
import argparse
parser = argparse.ArgumentParser(description='Convert score-matrix to "segments" in kaldi format.')
# optional,
parser.add_argument('-s', '--smooth-window', type=int, default=31, help='length of the score-smoothing window')
parser.add_argument('-t', '--speech-threshold', type=float, default=-0.5, help="speech detection threshold (<0: more 'speech', 0.0=default, >0: less 'speech')")
parser.add_argument('-e', '--extra-speech-frames', type=int, default=30, help='speech frames added to the begining/end of each segment (in 10ms frames)')
parser.add_argument('-m', '--min-segment-dur', type=int, default=10, help='minimal length of a segment (in 10ms frames)')
parser.add_argument('-l', '--max-speech-segment-dur', type=int, default=1500, help='maximal length of a segment (in 10ms frames)')
parser.add_argument('-g', '--graph-dir', type=str, default='', help='directory for storing the score histogram plot')
# positional,
parser.add_argument('log_post_scp', help='scp with score matrix (log-posteriors), scp with 1 matrix only...')
parser.add_argument('segments_file', help='location of "segments" file (output)')
# parse,
args = parser.parse_args()


segments=[]
for (utt, m) in kaldi_io.read_mat_scp(args.log_post_scp):
  # Asuming m is pre-softmax, get posterior-logit,
  logit = m[:,0] - m[:,1]
  # Average-smoothing,
  N = args.smooth_window
  s = np.cumsum(logit)
  logit_smooth = np.hstack([ logit[:N//2], (s[N:] - s[:-N]) // N, logit[-N//2:] ])

  # Store the histogram plot,
  if args.graph_dir != "":
    f = plt.figure(figsize=(8,4));
    # histogram
    y,x = np.histogram(logit_smooth, bins=50, density=True)
    plt.plot(x[1:],y,'b',label='scores in unsegmented recording');
    plt.plot([args.speech_threshold,args.speech_threshold],[0,np.max(y)],'r--', label='decision-threshold');
    plt.xlabel('score, smoothed per-frame logit-posteiors'); plt.ylabel('p(score)');
    plt.title(utt); plt.grid(); plt.legend(); plt.tight_layout();
    f.savefig('%s/%s.png' % (args.graph_dir,utt))

  # Get the segments,
  score = logit_smooth # will use smoothed logit-posterior scores,
  thr = args.speech_threshold # the decision threshold
  num_frames = logit_smooth.shape[0]
  #
  decisions = np.r_[False, score > thr, False] # score bigger than threshold is speech,
  speech_segs_raw = np.nonzero(decisions[1:] != decisions[:-1])[0].reshape(-1,2)
  if len(speech_segs_raw) == 0 : continue # no-speech-found!

  # Frame extension,
  Next = args.extra_speech_frames
  speech_segs_ext = np.array(list(zip(speech_segs_raw[:,0] - Next, speech_segs_raw[:,1] + Next)))
  speech_segs_ext[speech_segs_ext<0] = 0
  speech_segs_ext[speech_segs_ext>=num_frames] = num_frames

  # Merge overlapping segments,
  decisions_ext = np.zeros(num_frames + 2)
  for (b,e) in speech_segs_ext : decisions_ext[b+1:e+1] = 1
  speech_segs_ext = np.nonzero(decisions_ext[1:] != decisions_ext[:-1])[0].reshape(-1,2)

  # Make sure no raw-segment is shorter than 0.10s,
  Tmin = args.min_segment_dur + 2*(Next if Next > 0 else 0)
  speech_segs = speech_segs_ext[(speech_segs_ext[:,1]-speech_segs_ext[:,0])>Tmin]
  if len(speech_segs) == 0 : continue # no-speech-found! (all segments dropped, min duration)

  # Make sure no segment is longer than 15.00s,
  Tmax = args.max_speech_segment_dur
  while 1:
    long_segs = (speech_segs[:,1]-speech_segs[:,0]) > Tmax # bool
    if np.sum(long_segs) == 0: break
    for i in np.array(range(len(long_segs)))[long_segs][::-1]: # Desceding order, we will be inserting!
      s = np.array(speech_segs[i])
      min = (s[0]+Tmax//3) + np.argmin(logit_smooth[s[0]+Tmax//3:s[1]-Tmax//3]) # Search in middle 5s
      assert(min > s[0]); assert(min < s[1]);
      speech_segs[i] = [min+1, s[1]] # at 'i'
      speech_segs = np.insert(speech_segs, i, [s[0], min-1], axis=0) # Before 'i'
      #print(i,speech_segs[i-1],speech_segs[i],speech_segs[i+1],speech_segs[i+2],min,speech_segs.shape)

  # Add to segments
  segments.append(['%s!%07d-%07d %s %06.2f %06.2f' % (utt,beg,end,utt,beg/100.,end/100.) for beg,end in speech_segs])

  # Print some statistical info,
  final_speech_frames = np.sum(speech_segs[:,1]-speech_segs[:,0])/float(decisions.shape[0])
  dur_seconds = decisions.shape[0]/100.
  print(utt, ', duration-in-seconds', dur_seconds, ', raw-speech-frames', np.mean(decisions), ', final-speech-frames', final_speech_frames)

# Create output-dir (if needed),
d = os.path.dirname(args.segments_file)
if not os.path.exists(d):
  try : os.makedirs(d)
  except : pass

# Save segments
output = []
if segments : output = np.hstack(segments)
with open(args.segments_file, 'w') as f:
  np.savetxt(f, output, delimiter='\n', fmt='%s')
  print("Segments saved :", args.segments_file)

# Done!
