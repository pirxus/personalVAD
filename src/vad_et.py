"""@package vad_et

This module implements the ET vad architecture from {paper_link}. The input for this architecture
consists of a 296-dimensional feature vector of which 40 values are our extracted logfbank
energies and the other 256 values is the target speaker embedding d-vector.

"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.nn.functional import one_hot
import torch.nn.functional as F
import kaldiio
import argparse as ap

from sklearn.metrics import average_precision_score

import numpy as np
import pickle
import os
import sys
from glob import glob

from vad import pad_collate

# model hyper parameters
num_epochs = 10
batch_size = 128
batch_size_test = 128

input_dim = 296
hidden_dim = 64
out_dim = 3
num_layers = 2
lr = 1e-3
SCHEDULER = True

DATA_TRAIN = 'data/train'
DATA_TEST = 'data/test'
EMBED_PATH = 'embeddings'
MODEL_PATH = 'vad_et.pt'
SAVE_MODEL = True

USE_KALDI = False
USE_WPL = False
MULTI_GPU = False
DATA_TRAIN_KALDI = 'data/train'
DATA_TEST_KALDI = 'data/test'

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
WPL_WEIGHTS = torch.tensor([1.0, 0.2, 1.0]).to(device)

class VadETDatasetArk(Dataset):
    """VadET training dataset. Uses kaldi scp and ark files."""

    def __init__(self, root_dir, embed_path):
        self.root_dir = root_dir
        self.embed_path = embed_path
        self.fbanks = kaldiio.load_scp(f'{self.root_dir}/fbanks.scp')
        self.labels = kaldiio.load_scp(f'{self.root_dir}/labels.scp')
        self.keys = np.array(list(self.fbanks)) # get all the keys
        self.embed = kaldiio.load_scp(f'{self.embed_path}/dvectors.scp')

        # load the target speaker ids
        self.targets = {}
        with open(f'{self.root_dir}/targets.scp') as targets:
            for line in targets:
                (utt_id, target) = line.split()
                self.targets[utt_id] = target

    def __len__(self):
        return self.keys.size

    def __getitem__(self, idx):
        key = self.keys[idx]
        target = self.targets[key]
        x = self.fbanks[key]
        embed = self.embed[target]
        y = self.labels[key]

        # add the dvector array to the feature vector
        x = np.hstack((x, np.full((x.shape[0], 256), embed)))

        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).long()
        return x, y

class VadETDataset(Dataset):
    """VadET training dataset."""

    def __init__(self, root_dir):
        """Initializes the dataset object and loads the paths to the feature files into
        the file_list attribute.

        Args:
            root_dir (str): Path to the root directory of the dataset. In this folder,
                there can be several other folders containing the data.
        """

        self.file_list = list()

        # first load the paths to the feature files
        with os.scandir(root_dir) as folders:
            for folder in folders:
                self.file_list.extend(glob(folder.path + '/*.fea.npz'))
        self.n_utterances = len(self.file_list)

    def __len__(self):
        return self.n_utterances

    def __getitem__(self, index):
        with np.load(self.file_list[index]) as f:
            x = f['x']
            embed = f['embed']
            y = f['y']

            # add the dvector array to the feature vector
            x = np.hstack((x, np.full((x.shape[0], 256), embed)))

            x = torch.from_numpy(x).float()
            y = torch.from_numpy(y)
            return x, y

        return None

class VadET(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, out_dim):
        super(VadET, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.out_dim = out_dim

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim) #
        self.relu = nn.LeakyReLU() #
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, x_lens, hidden):
        x_packed = pack_padded_sequence(x, x_lens, batch_first=True, enforce_sorted=False)
        out_packed, hidden = self.lstm(x_packed, hidden)
        out_padded, _ = pad_packed_sequence(out_packed, batch_first=True)

        out_padded = self.fc1(out_padded) #
        out_padded = self.relu(out_padded) #
        out_padded = self.fc2(out_padded)
        return out_padded, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.num_layers, batch_size, self.hidden_dim)
        cell = weight.new(self.num_layers, batch_size, self.hidden_dim)
        return torch.stack([hidden, cell])


class WPL(nn.Module):
    """Weighted pairwise loss
    The weight pairs are interpreted as follows:
    [<ns,tss> ; <ntss,ns> ; <tss,ntss>]

    target contains indexes, output is a tensor of probabilites for each class
    (ns, ntss, tss) -> {0, 1, 2} 

    """

    def __init__(self, weights):
        super(WPL, self).__init__()
        self.weights = weights
        assert len(weights) == 3, "The wpl is defined for three classes only."

    def forward(self, output, target):
        minibatch_size = output.size(0)
        label_mask = one_hot(target) > 0.5 # boolean mask
        label_mask_r1 = torch.roll(label_mask, 1, 1) # if ntss, then tss
        label_mask_r2 = torch.roll(label_mask, 2, 1) # if ntss, then ns

        # get the probability of the actual label and the other two into one array
        actual = torch.exp(torch.masked_select(output, label_mask))
        plus_one = torch.exp(torch.masked_select(output, label_mask_r1))
        minus_one = torch.exp(torch.masked_select(output, label_mask_r2))

        # arrays of the first pair weight and the second pair weight used in the equation
        w1 = torch.masked_select(self.weights, label_mask) # if ntss, w1 is <ntss, ns>
        w2 = torch.masked_select(self.weights, label_mask_r1) # if ntss, w2 is <tss, ntss>

        # first pair
        first_pair = w1 * torch.log(actual / (actual + minus_one))
        second_pair = w2 * torch.log(actual / (actual + plus_one))

        # get the negative mean value for the two pairs
        wpl = -0.5 * (first_pair + second_pair)

        # sum and average for minibatch
        return wpl.sum() / minibatch_size


if __name__ == '__main__':
    # default data path
    data_train = DATA_TRAIN_KALDI if USE_KALDI else DATA_TRAIN
    data_test = DATA_TEST_KALDI if USE_KALDI else DATA_TEST

    # program arguments
    parser = ap.ArgumentParser(description="Train the VAD ET model.")
    parser.add_argument('--train_dir', type=str, default=data_train)
    parser.add_argument('--test_dir', type=str, default=data_test)
    parser.add_argument('--embed_path', type=str, default=EMBED_PATH)
    parser.add_argument('--model_path', type=str, default=MODEL_PATH)
    parser.add_argument('--load_model', type=str, default='')
    parser.add_argument('--use_kaldi', action='store_true')
    parser.add_argument('--use_wpl', action='store_true')
    args = parser.parse_args()

    MODEL_PATH = args.model_path
    data_train = args.train_dir
    data_test = args.test_dir
    EMBED_PATH = args.embed_path
    USE_KALDI = args.use_kaldi
    USE_WPL = args.use_wpl

    # Load the data and create DataLoader instances
    if USE_KALDI:
        train_data = VadETDatasetArk(data_train, EMBED_PATH)
        test_data = VadETDatasetArk(data_test, EMBED_PATH)
    else:
        train_data = VadETDataset(data_train)
        test_data = VadETDataset(data_test)

    train_loader = DataLoader(
            dataset=train_data, num_workers=4, pin_memory=True,
            batch_size=batch_size, shuffle=True, collate_fn=pad_collate)
    test_loader = DataLoader(
            dataset=test_data, num_workers=4, pin_memory=True,
            batch_size=batch_size_test, shuffle=False, collate_fn=pad_collate)

    model = VadET(input_dim, hidden_dim, num_layers, out_dim).to(device)

    if USE_WPL:
        print("Using the wpl..")
        criterion = WPL(WPL_WEIGHTS)
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if SCHEDULER:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

    softmax = nn.Softmax(dim=1)

    # Train!!! hype!!!
    for epoch in range(num_epochs):
        print(f"====== Starting epoch {epoch} ======")
        for batch, (x_padded, y_padded, x_lens, y_lens) in enumerate(train_loader):
            y_padded = y_padded.to(device)

            # pass the data through the model
            out_padded, _ = model(x_padded.to(device), x_lens, None)

            # compute the loss
            loss = 0
            for j in range(out_padded.size(0)):
                loss += criterion(out_padded[j][:y_lens[j]], y_padded[j][:y_lens[j]])

            loss /= batch_size # normalize loss for each batch..
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            if batch % 10 == 0:
                print(f'Batch: {batch}, loss = {loss:.4f}')

        if SCHEDULER and epoch < 2:
            scheduler.step() # learning rate adjust
            if epoch == 1:
                lr = 5e-5
        if SCHEDULER and epoch == 7:
            lr = 1e-5

        # Test the model after each epoch
        with torch.no_grad():
            print("testing...")
            n_correct = 0
            n_samples = 0
            targets = []
            outputs = []
            for x_padded, y_padded, x_lens, y_lens in test_loader:
                y_padded = y_padded.to(device)

                # pass the data through the model
                out_padded, _ = model(x_padded.to(device), x_lens, None)

                # value, index
                for j in range(out_padded.size(0)):
                    classes = torch.argmax(out_padded[j][:y_lens[j]], dim=1)
                    n_samples += y_lens[j]
                    n_correct += torch.sum(classes == y_padded[j][:y_lens[j]]).item()

                    # average precision
                    p = softmax(out_padded[j][:y_lens[j]])
                    outputs.append(p.cpu().numpy())
                    targets.append(y_padded[j][:y_lens[j]].cpu().numpy())

            acc = 100.0 * n_correct / n_samples
            print(f"accuracy = {acc:.2f}")

            # and run the AP
            targets = np.concatenate(targets)
            outputs = np.concatenate(outputs)
            targets_oh = np.eye(3)[targets]
            out_AP = average_precision_score(targets_oh, outputs, average=None)
            mAP = average_precision_score(targets_oh, outputs, average='micro')

            print(out_AP)
            print(f"mAP: {mAP}")

        # Save the model (after each epoch just to be sure...)
        if SAVE_MODEL:

            # if necessary, create the destination path for the model...
            path_seg = MODEL_PATH.split('/')[:-1]
            if path_seg != []:
                if not os.path.exists(MODEL_PATH.rpartition('/')[0]):
                    os.makedirs('/'.join(path_seg))
            torch.save(model.state_dict(), MODEL_PATH)

