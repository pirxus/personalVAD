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

import numpy as np
import pickle
import os
import sys
from glob import glob

from vad import pad_collate

# model hyper parameters
num_epochs = 5
batch_size = 128
batch_size_test = 8

input_dim = 296
hidden_dim = 64
out_dim = 3
num_layers = 2
lr = 1e-2

DATA_TRAIN = 'data/features/train'
DATA_TEST = 'data/features/test'
MODEL_PATH = 'src/models/vad.pt'

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

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
                self.file_list.extend(glob(folder.path + '/*.fea'))
        self.n_utterances = len(self.file_list)

    def __len__(self):
        return self.n_utterances

    def __getitem__(self, index):
        with open(self.file_list[index], 'rb') as f:
            x, embed, y = pickle.load(f) # (x, y) == (logfbanks, labels)

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

    def forward(self, x):
        out_packed, _ = self.lstm(x)
        out_padded, out_lengths = pad_packed_sequence(out_packed, batch_first=True)

        out_padded = self.fc(out_padded)
        return out_padded

if __name__ == '__main__':
    # Load the data and create DataLoader instances
    train_data = VadETDataset(DATA_TRAIN)
    test_data = VadETDataset(DATA_TEST)
    train_loader = DataLoader(
            dataset=train_data, batch_size=batch_size, shuffle=True, collate_fn=pad_collate)
    test_loader = DataLoader(
            dataset=test_data, batch_size=batch_size_test, shuffle=False, collate_fn=pad_collate)

    model = VadET(input_dim, hidden_dim, num_layers, out_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

    # Train!!! hype!!!
    for epoch in range(num_epochs):
        print(f"====== Starting epoch {epoch} ======")
        for batch, (x_padded, y_padded, x_lens, y_lens) in enumerate(train_loader):

            # pad the sequences..
            x_packed = pack_padded_sequence(
                    x_padded, x_lens, batch_first=True, enforce_sorted=False).to(device)

            # zero the gradients..
            optimizer.zero_grad()

            # get the prediction
            out_padded = model(x_packed)
            
            loss = torch.zeros(3, device=device)
            y_padded = y_padded.to(device)

            for j in range(out_padded.size(0)):
                loss += criterion(out_padded[j][:y_lens[j]], y_padded[j][:y_lens[j]])

            loss = loss.sum() / batch_size # normalize loss for each batch..
            loss.backward()
            optimizer.step()
            
            if batch % 10 == 0:
                print(f'Batch: {batch}, loss = {loss:.4f}')

        scheduler.step() # learning rate adjust

        # Test the model after each epoch
        with torch.no_grad():
            print("testing...")
            n_correct = 0
            n_samples = 0
            for x_padded, y_padded, x_lens, y_lens in test_loader:
                # pad the sequences..
                x_packed = pack_padded_sequence(
                        x_padded, x_lens, batch_first=True, enforce_sorted=False).to(device)

                out_padded = model(x_packed)
                y_padded = y_padded.to(device)

                # value, index
                for j in range(out_padded.size(0)):
                    classes = torch.argmax(out_padded[j][:y_lens[j]], dim=1)
                    n_samples += y_lens[j]
                    n_correct += torch.sum(classes == y_padded[j][:y_lens[j]]).item()

            acc = 100.0 * n_correct / n_samples
            print(f"accuracy = {acc:.2f}")

    # Save the model
    torch.save(model.state_dict(), MODEL_PATH)

