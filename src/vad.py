"""@package vad

Author: Simon Sedlacek
Email: xsedla1h@stud.fit.vutbr.cz

This module implements the baseline VAD model training loop.

The input for this architecture are the 40-dimensional log Mel-filterbank energies.

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

from personal_vad import PersonalVAD, pad_collate

# model hyper parameters
num_epochs = 6
batch_size = 64
batch_size_test = 64

input_dim = 40
hidden_dim = 64
num_layers = 2
out_dim = 2
lr = 1e-3
SCHEDULER = True

DATA_TRAIN = 'data/train'
DATA_TEST = 'data/test'
MODEL_PATH = 'vad.pt'
SAVE_MODEL = True

NUM_WORKERS = 2

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class VadDataset(Dataset):
    """Vad dataset class. Uses kaldi scp and ark files."""

    def __init__(self, root_dir):
        self.root_dir = root_dir

        # load the scp files...
        self.fbanks = kaldiio.load_scp(f'{self.root_dir}/fbanks.scp')
        self.labels = kaldiio.load_scp(f'{self.root_dir}/labels.scp')
        self.keys = np.array(list(self.fbanks)) # get all the keys

    def __len__(self):
        return self.keys.size

    def __getitem__(self, idx):
        key = self.keys[idx]
        x = self.fbanks[key]
        y = self.labels[key]
        
        # convert the PVAD labels to speech/non-speech only
        y = (y != 0).astype('int')

        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).long()
        return x, y


if __name__ == '__main__':
    """ Model training  """

    # program arguments
    parser = ap.ArgumentParser(description="Train the base VAD model.")
    parser.add_argument('--train_dir', type=str, default=DATA_TRAIN)
    parser.add_argument('--test_dir', type=str, default=DATA_TEST)
    parser.add_argument('--model_path', type=str, default=MODEL_PATH)
    parser.add_argument('--use_kaldi', action='store_true')
    parser.add_argument('--nuse_fc', action='store_false')
    parser.add_argument('--linear', action='store_true')
    parser.add_argument('--nsave_model', action='store_false')
    args = parser.parse_args()

    MODEL_PATH = args.model_path
    DATA_TRAIN = args.train_dir
    linear = args.linear
    DATA_TEST = args.test_dir
    SAVE_MODEL = args.nsave_model

    # Load the data and create DataLoader instances
    train_data = VadDataset(DATA_TRAIN)
    test_data = VadDataset(DATA_TEST)

    train_loader = DataLoader(
            dataset=train_data, num_workers=NUM_WORKERS, pin_memory=True,
            batch_size=batch_size, shuffle=True, collate_fn=pad_collate)
    test_loader = DataLoader(
            dataset=test_data, num_workers=NUM_WORKERS, pin_memory=True,
            batch_size=batch_size_test, shuffle=False, collate_fn=pad_collate)

    model = PersonalVAD(input_dim, hidden_dim, num_layers, out_dim, use_fc=args.nuse_fc, linear=linear).to(device)

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
                optimizer.param_groups[0]['lr'] = 5e-5
        if SCHEDULER and epoch == 4:
            optimizer.param_groups[0]['lr'] = 1e-5

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
