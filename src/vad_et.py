"""@package vad_et

This module implements the ET vad architecture from {paper_link}. The input for this architecture
consists of a 296-dimensional feature vector of which 40 values are our extracted logfbank
energies and the other 256 values is the target speaker embedding d-vector.

"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
import kaldiio
import argparse as ap

import numpy as np
import pickle
import os
import sys
from glob import glob

from vad import pad_collate

# model hyper parameters
num_epochs = 3
batch_size = 128
batch_size_test = 32

input_dim = 296
hidden_dim = 64
out_dim = 3
num_layers = 2
lr = 1e-3
SCHEDULER = True

DATA_TRAIN = 'data/train'
DATA_TEST = 'data/test'
MODEL_PATH = 'vad_et.pt'
SAVE_MODEL = True

USE_KALDI = False
MULTI_GPU = True
DATA_TRAIN_KALDI = 'data/train'
DATA_TEST_KALDI = 'data/test'

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class VadETDatasetArk(Dataset):
    """VadET training dataset. Uses kaldi scp and ark files."""

    def __init__(self, root_dir):
        self.root_dir = root_dir if root_dir[-1] == '/' else root_dir + '/'
        self.fbanks = kaldiio.load_scp(f'{self.root_dir}fbanks.scp')
        self.embed = kaldiio.load_scp(f'{self.root_dir}embed.scp')
        self.labels = kaldiio.load_scp(f'{self.root_dir}labels.scp')
        self.keys = np.array(list(self.fbanks)) # get all the keys

    def __len__(self):
        return self.keys.size

    def __getitem__(self, idx):
        key = self.keys[idx]
        x = self.fbanks[key]
        embed = self.embed[key]
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

# TODO: implement the WPL loss function
class VadET(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, out_dim):
        super(VadET, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.out_dim = out_dim

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, x_lens, hidden):
        x_packed = pack_padded_sequence(x, x_lens, batch_first=True, enforce_sorted=False)
        out_packed, hidden = self.lstm(x_packed, hidden)
        out_padded, _ = pad_packed_sequence(out_packed, batch_first=True)

        out_padded = self.fc(out_padded)
        return out_padded, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.num_layers, batch_size, self.hidden_dim)
        cell = weight.new(self.num_layers, batch_size, self.hidden_dim)
        return torch.stack([hidden, cell])

def wpl(output, target):
    """Weighted pairwise loss.
    """

if __name__ == '__main__':
    # default data path
    data_train = DATA_TRAIN_KALDI if USE_KALDI else DATA_TRAIN
    data_test = DATA_TEST_KALDI if USE_KALDI else DATA_TEST

    # program arguments
    parser = ap.ArgumentParser(description="Train the VAD ET model.")
    parser.add_argument('--train_dir', type=str, default=data_train)
    parser.add_argument('--test_dir', type=str, default=data_test)
    parser.add_argument('--model_path', type=str, default=MODEL_PATH)
    parser.add_argument('--use_kaldi', action='store_true')
    args = parser.parse_args()

    MODEL_PATH = args.model_path
    data_train = args.train_dir
    data_test = args.test_dir
    USE_KALDI = args.use_kaldi

    # Load the data and create DataLoader instances
    if USE_KALDI:
        train_data = VadETDatasetArk(data_train)
        test_data = VadETDatasetArk(data_test)
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
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if SCHEDULER:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

    # Train!!! hype!!!
    for epoch in range(num_epochs):
        print(f"====== Starting epoch {epoch} ======")
        for batch, (x_padded, y_padded, x_lens, y_lens) in enumerate(train_loader):
            y_padded = y_padded.to(device)

            # pass the data through the model
            out_padded, _ = model(x_padded.to(device), x_lens, None)

            # compute the loss
            loss = torch.zeros(3, device=device)
            for j in range(out_padded.size(0)):
                loss += criterion(out_padded[j][:y_lens[j]], y_padded[j][:y_lens[j]])

            loss = loss.sum() / batch_size # normalize loss for each batch..
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            if batch % 10 == 0:
                print(f'Batch: {batch}, loss = {loss:.4f}')

        if SCHEDULER:
            if epoch == 0:
                scheduler.step() # learning rate adjust
            if (epoch + 1) % 6 == 0:
                scheduler.step() # learning rate adjust

        # Test the model after each epoch
        with torch.no_grad():
            print("testing...")
            n_correct = 0
            n_samples = 0
            for x_padded, y_padded, x_lens, y_lens in test_loader:
                y_padded = y_padded.to(device)

                # pass the data through the model
                out_padded, _ = model(x_padded.to(device), x_lens, None)

                # value, index
                for j in range(out_padded.size(0)):
                    classes = torch.argmax(out_padded[j][:y_lens[j]], dim=1)
                    n_samples += y_lens[j]
                    n_correct += torch.sum(classes == y_padded[j][:y_lens[j]]).item()

            acc = 100.0 * n_correct / n_samples
            print(f"accuracy = {acc:.2f}")

        # Save the model (after each epoch just to be sure...)
        if SAVE_MODEL:

            # if necessary, create the destination path for the model...
            path_seg = MODEL_PATH.split('/')[:-1]
            if path_seg != []:
                if not os.path.exists(MODEL_PATH.rpartition('/')[0]):
                    os.makedirs('/'.join(path_seg))
            torch.save(model.state_dict(), MODEL_PATH)

