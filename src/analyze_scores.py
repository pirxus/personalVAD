""" This little script analyzes the score values extracted using the full-on frame-level approach
vs the window-level approach
"""

import numpy as np
import os
import sys
from glob import glob
import argparse as ap
import matplotlib.pyplot as plt
import seaborn as sns
import kaldiio

DATA_DIR = 'data/score_analysis'

which = 'tss'

# move to the analyzed directory
os.chdir(DATA_DIR)

# load the scps and get the keys
scores_scp = kaldiio.load_scp('scores.scp')
labels_scp = kaldiio.load_scp('labels.scp')
keys = np.array(list(scores_scp))

#if which == 'tss':
#    tss = [[], [], []]
#elif which == 'ntss':
#    ntss = [[], [], []]
#elif which == 'ns':
#    ns = [[], [], []]
#else:
#    print("Please specify which class to analyze...")
#    sys.exit()

tss = [[], []]
ntss = [[], []]
ns = [[], []]


for i, key in enumerate(keys):
    scores = scores_scp[key]
    labels = labels_scp[key]

    # sub-sample the arrays, indexes 159 + k*40, since that's where the methods line up
    scores = scores[:2, 159::40]
    labels = labels[159::40]

    labels_ns = labels == 0
    labels_ntss = labels == 1
    labels_tss = labels == 2
    
    # get the sample score values
    for score_method in range(2):
        ns[score_method].append(scores[score_method, labels_ns])
        ntss[score_method].append(scores[score_method, labels_ntss])
        tss[score_method].append(scores[score_method, labels_tss])


ns_mean = []
ntss_mean = []
tss_mean = []

ns_var = []
ntss_var = []
tss_var = []

# now collapse the accumulated lists into one array and compute mean and variance
for i in range(2):

    ns[i] = np.concatenate(ns[i])
    ns_mean.append(np.mean(ns[i]))
    ns_var.append(np.var(ns[i]))

    ntss[i] = np.concatenate(ntss[i])
    ntss_mean.append(np.mean(ntss[i]))
    ntss_var.append(np.var(ntss[i]))

    tss[i] = np.concatenate(tss[i])
    tss_mean.append(np.mean(tss[i]))
    tss_var.append(np.var(tss[i]))

print(f"tss mean = {tss_mean[0]}, {tss_mean[1]}; tss var = {tss_var[0]}, {tss_var[1]}")
print(f"ntss mean = {ntss_mean[0]}, {ntss_mean[1]}; ntss var = {ntss_var[0]}, {ntss_var[1]}")
print(f"ns mean = {ns_mean[0]}, {ns_mean[1]}; ns var = {ns_var[0]}, {ns_var[1]}")
bins = 50 

fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(
    nrows=5, ncols=1, figsize=(7, 9), sharex=True)


sns.distplot(tss[0], hist=True, kde=True, bins=100, color='darkblue', hist_kws={'edgecolor': 'black'},kde_kws={'linewidth': 4}, ax=ax1)
sns.distplot(tss[1], hist=True, kde=True, bins=100, color='red', hist_kws={'edgecolor': 'black'},kde_kws={'linewidth': 4}, ax=ax1)


sns.distplot(ntss[0], hist=True, kde=True, bins=100, color='darkblue', hist_kws={'edgecolor': 'black'},kde_kws={'linewidth': 4}, ax=ax2)
sns.distplot(ntss[1], hist=True, kde=True, bins=100, color='red', hist_kws={'edgecolor': 'black'},kde_kws={'linewidth': 4}, ax=ax2)

sns.distplot(ns[0], hist=True, kde=True, bins=100, color='darkblue', hist_kws={'edgecolor': 'black'},kde_kws={'linewidth': 4}, ax=ax3)
sns.distplot(ns[1], hist=True, kde=True, bins=100, color='red', hist_kws={'edgecolor': 'black'},kde_kws={'linewidth': 4}, ax=ax3)



sns.distplot(ns[0], hist=True, kde=True, bins=100, color='lime', hist_kws={'edgecolor': 'black'},kde_kws={'linewidth': 4}, ax=ax4)
sns.distplot(tss[0], hist=True, kde=True, bins=100, color='darkblue', hist_kws={'edgecolor': 'black'},kde_kws={'linewidth': 4}, ax=ax4)
sns.distplot(ntss[0], hist=True, kde=True, bins=100, color='red', hist_kws={'edgecolor': 'black'},kde_kws={'linewidth': 4}, ax=ax4)


sns.distplot(ns[1], hist=True, kde=True, bins=100, color='lime', hist_kws={'edgecolor': 'black'},kde_kws={'linewidth': 4}, ax=ax5)
sns.distplot(tss[1], hist=True, kde=True, bins=100, color='darkblue', hist_kws={'edgecolor': 'black'},kde_kws={'linewidth': 4}, ax=ax5)
sns.distplot(ntss[1], hist=True, kde=True, bins=100, color='red', hist_kws={'edgecolor': 'black'},kde_kws={'linewidth': 4}, ax=ax5)


ax1.set(xlabel='', title='tss: frame-level (blue) vs. window-level (red)')
ax2.set(xlabel='', title='ntss: frame-level (blue) vs. window-level (red)')
ax3.set(xlabel='', title='ns: frame-level (blue) vs. window-level (red)')
ax4.set(xlabel='', title='frame-level: tss (blue) vs. ntss (red) vs. ns (green)')
ax5.set(xlabel='cosine distance', title='window-level: tss (blue) vs. ntss (red) vs. ns (green)')

plt.show()
