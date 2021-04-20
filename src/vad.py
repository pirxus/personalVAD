"""@package vad

This module implements a basic vad model.

"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
import kaldiio
import argparse as ap

from sklearn.metrics import average_precision_score

import numpy as np
import os
import sys
from glob import glob

from personal_vad import PersonalVAD, pad_collate

# model hyper parameters
num_epochs = 6
batch_size = 128
batch_size_test = 128

input_dim = 40
hidden_dim = 64
num_layers = 2
out_dim = 2
lr = 1e-3
SCHEDULER = True

DATA_TRAIN = 'data/features/train'
DATA_TEST = 'data/features/test'
MODEL_PATH = 'vad.pt'
SAVE_MODEL = True

USE_KALDI = False
DATA_TRAIN_KALDI = 'data/train'
DATA_TEST_KALDI = 'data/test'

CONVERT_LABELS = True # uses the ns, tss, ntss labels and converts them to the base vad labels

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class VadDatasetArk(Dataset):
    """Vad training dataset. Uses kaldi scp and ark files."""

    def __init__(self, root_dir):
        self.root_dir = root_dir if root_dir[-1] == '/' else root_dir + '/'
        self.fbanks = kaldiio.load_scp(f'{self.root_dir}fbanks.scp')
        if CONVERT_LABELS:
            self.labels = kaldiio.load_scp(f'{self.root_dir}labels.scp')
        else:
            self.labels = kaldiio.load_scp(f'{self.root_dir}labels_vad.scp')
        self.keys = np.array(list(self.fbanks)) # get all the keys

    def __len__(self):
        return self.keys.size

    def __getitem__(self, idx):
        key = self.keys[idx]
        x = self.fbanks[key]
        y = self.labels[key]

        if CONVERT_LABELS:
            y = (y != 0).astype('int')

        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).long()
        return x, y

class VadDataset(Dataset):
    """Vad training dataset."""

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
                self.file_list.extend(glob(folder.path + '/*.vad.fea.npz'))
        self.n_utterances = len(self.file_list)

    def __len__(self):
        return self.n_utterances

    def __getitem__(self, index):
        with np.load(self.file_list[index]) as f:
            x = f['x']
            y = f['y']

            x = torch.from_numpy(x).float()
            y = torch.from_numpy(y).float()
            return x, y

        return None

# training process..
if __name__ == '__main__':
    # default data path
    data_train = DATA_TRAIN_KALDI if USE_KALDI else DATA_TRAIN
    data_test = DATA_TEST_KALDI if USE_KALDI else DATA_TEST

    # program arguments
    parser = ap.ArgumentParser(description="Train the base VAD model.")
    parser.add_argument('--train_dir', type=str, default=data_train)
    parser.add_argument('--test_dir', type=str, default=data_test)
    parser.add_argument('--model_path', type=str, default=MODEL_PATH)
    parser.add_argument('--use_kaldi', action='store_true')
    parser.add_argument('--nuse_fc', action='store_false')
    args = parser.parse_args()

    MODEL_PATH = args.model_path
    data_train = args.train_dir
    data_test = args.test_dir
    USE_KALDI = args.use_kaldi

    # Load the data and create DataLoader instances
    if USE_KALDI:
        train_data = VadDatasetArk(data_train)
        test_data = VadDatasetArk(data_test)
    else:
        train_data = VadDataset(data_train)
        test_data = VadDataset(data_test)

    train_loader = DataLoader(
            dataset=train_data, num_workers=4, pin_memory=True,
            batch_size=batch_size, shuffle=True, collate_fn=pad_collate)
    test_loader = DataLoader(
            dataset=test_data, num_workers=4, pin_memory=True,
            batch_size=batch_size_test, shuffle=False, collate_fn=pad_collate)

    model = PersonalVAD(input_dim, hidden_dim, num_layers, out_dim, use_fc=args.nuse_fc).to(device)

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

            loss /= batch_size # normalize for the batch
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            if batch % 10 == 0:
                print(f'Batch: {batch}, loss = {loss.item():.4f}')

        if SCHEDULER and epoch < 2:
            scheduler.step() # learning rate adjust
            if epoch == 1:
                lr = 5e-5
        if epoch == 4:
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
            targets_oh = np.eye(2)[targets]
            out_AP = average_precision_score(targets_oh, outputs, average=None)
            mAP = average_precision_score(targets_oh, outputs, average='micro')

            print(out_AP)
            print(f"mAP: {mAP}")

        # Save the model - after each epoch for ensurance...
        if SAVE_MODEL:

            # if necessary, create the destination path for the model...
            path_seg = MODEL_PATH.split('/')[:-1]
            if path_seg != []:
                if not os.path.exists(MODEL_PATH.rpartition('/')[0]):
                    os.makedirs('/'.join(path_seg))
            torch.save(model.state_dict(), MODEL_PATH)
