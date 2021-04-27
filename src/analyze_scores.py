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
import pandas as pd
from sklearn.metrics import roc_curve, auc

DATA_DIR = 'data/score_analysis'

which = 'tss'

# move to the analyzed directory
os.chdir(DATA_DIR)

# load the scps and get the keys
scores_scp = kaldiio.load_scp('scores.scp')
labels_scp = kaldiio.load_scp('labels.scp')
keys = np.array(list(scores_scp))

# scoring method comparison plot

# get a suitable utterance with three speakers
"""
for i, key in enumerate(keys):
    if key.count('_') == 2 and key.rpartition('_')[2] not in ['music', 'noise', 'reverb']:
        if i == 20:
            break
        else:
            i += 1

scores = scores_scp[key]
labels = labels_scp[key]

fig = plt.figure(figsize=(10,4))
stop = -250

plt.plot(scores[0,:stop], label='baseline')
plt.plot(scores[1,:stop], label='partially-constant', linewidth=2.0)
plt.plot(scores[2,:stop], label='linearly-interpolated')
index = np.argmax(labels == 1.0)
plt.fill_between(np.arange(index), labels[:index]*0.5, step='pre', alpha=0.1)
plt.fill_between(np.arange(index, labels.size + stop), labels[index:stop], step='pre', alpha=0.1)

plt.ylim(0.4, 0.9)
plt.legend(loc="upper right", title='Scoring method')
plt.xlabel('Frame number')
plt.ylabel('Cosine similarity score')
plt.title('Scoring method comparison')
plt.show()
"""

# dvector extraction method comparisons

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


# 0 - ntss, 1 - tss
y = np.concatenate((np.zeros(ntss[0].size), np.ones(tss[0].size)))
fig = plt.figure(figsize=(5,4))

# frame-level
y_pred = np.concatenate((ntss[0], tss[0]))
fpr, tpr, threshold = roc_curve(y, y_pred)
roc_auc = auc(fpr, tpr)
fnr = 1 - tpr
eer_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]
EER = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
EER2 = fnr[np.nanargmin(np.absolute((fnr - fpr)))]
print("frame-level", EER, EER2, eer_threshold)
plt.plot(fpr, tpr, label=f'frame-level, baseline (AUC = {roc_auc:0.2f})')

# window-level
y_pred = np.concatenate((ntss[1], tss[1]))
fpr, tpr, threshold = roc_curve(y, y_pred)
roc_auc = auc(fpr, tpr)
fnr = 1 - tpr
eer_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]
EER = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
EER2 = fnr[np.nanargmin(np.absolute((fnr - fpr)))]
print("window-level", EER, EER2, eer_threshold)

plt.plot(fpr, tpr, color='darkorange', label=f'window-level (AUC = {roc_auc:0.2f})')

plt.xlim([-0.01, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Scoring method ROC comparison')
plt.legend(loc="lower right")
plt.savefig('roc.pdf')
#plt.show()



print(f"tss mean = {tss_mean[0]}, {tss_mean[1]}; tss var = {tss_var[0]}, {tss_var[1]}")
print(f"ntss mean = {ntss_mean[0]}, {ntss_mean[1]}; ntss var = {ntss_var[0]}, {ntss_var[1]}")
print(f"ns mean = {ns_mean[0]}, {ns_mean[1]}; ns var = {ns_var[0]}, {ns_var[1]}")
bins = 50
binwidth = 0.02

kde_kws={'linewidth': 3}

#fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(
#    nrows=5, ncols=1, figsize=(7, 9), sharex=True)
fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(7, 5), sharex=True)

data = {'frame-level': tss[0], 'window-level': tss[1]}
df = pd.DataFrame.from_dict(data)
df = pd.melt(df, id_vars=[], value_vars=['frame-level', 'window-level'])
df = df.rename(columns={'variable': 'scoring method'})

sns.histplot(data=df, x='value', hue='scoring method', binwidth=binwidth, stat='density', common_norm=False, kde=True, line_kws=kde_kws, ax=ax1)

data = {'frame-level': ntss[0], 'window-level': ntss[1]}
df = pd.DataFrame.from_dict(data)
df = pd.melt(df, id_vars=[], value_vars=['frame-level', 'window-level'])
df = df.rename(columns={'variable': 'scoring method'})
sns.histplot(data=df, x='value', hue='scoring method', binwidth=binwidth, stat='density', common_norm=False, kde=True, line_kws=kde_kws, ax=ax2)
#sns.distplot(df, hue='o', hist=True, kde=True, bins=100, color='darkblue', hist_kws={'edgecolor': 'black'},kde_kws={'linewidth': 4}, ax=ax1)
#sns.distplot(tss[0], hist=True, kde=True, bins=100, color='darkblue', hist_kws={'edgecolor': 'black'},kde_kws={'linewidth': 4}, ax=ax1)
#sns.distplot(tss[1], hist=True, kde=True, bins=100, color='red', hist_kws={'edgecolor': 'black'},kde_kws={'linewidth': 4}, ax=ax1)


#sns.distplot(ntss[0], hist=True, kde=True, bins=100, color='darkblue', hist_kws={'edgecolor': 'black'},kde_kws={'linewidth': 4}, ax=ax2)
#sns.distplot(ntss[1], hist=True, kde=True, bins=100, color='red', hist_kws={'edgecolor': 'black'},kde_kws={'linewidth': 4}, ax=ax2)


#sns.distplot(ns[0], hist=True, kde=True, bins=100, color='darkblue', hist_kws={'edgecolor': 'black'},kde_kws={'linewidth': 4}, ax=ax3)
#sns.distplot(ns[1], hist=True, kde=True, bins=100, color='red', hist_kws={'edgecolor': 'black'},kde_kws={'linewidth': 4}, ax=ax3)
#
#
#
#sns.distplot(ns[0], hist=True, kde=True, bins=100, color='lime', hist_kws={'edgecolor': 'black'},kde_kws={'linewidth': 4}, ax=ax4)
#sns.distplot(tss[0], hist=True, kde=True, bins=100, color='darkblue', hist_kws={'edgecolor': 'black'},kde_kws={'linewidth': 4}, ax=ax4)
#sns.distplot(ntss[0], hist=True, kde=True, bins=100, color='red', hist_kws={'edgecolor': 'black'},kde_kws={'linewidth': 4}, ax=ax4)
#
#
#sns.distplot(ns[1], hist=True, kde=True, bins=100, color='lime', hist_kws={'edgecolor': 'black'},kde_kws={'linewidth': 4}, ax=ax5)
#sns.distplot(tss[1], hist=True, kde=True, bins=100, color='darkblue', hist_kws={'edgecolor': 'black'},kde_kws={'linewidth': 4}, ax=ax5)
#sns.distplot(ntss[1], hist=True, kde=True, bins=100, color='red', hist_kws={'edgecolor': 'black'},kde_kws={'linewidth': 4}, ax=ax5)


ax1.set(xlabel='', title='Target speaker speech')
ax2.set(xlabel='Cosine similarity score', title='Non-target speaker speech')

#ax3.set(xlabel='', title='ns: frame-level (blue) vs. window-level (red)')
#ax4.set(xlabel='', title='frame-level: tss (blue) vs. ntss (red) vs. ns (green)')
#ax5.set(xlabel='cosine distance', title='window-level: tss (blue) vs. ntss (red) vs. ns (green)')
plt.savefig('scores_histogram.pdf')

plt.show()
