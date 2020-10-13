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

num_epochs = 5
batch_size = 128
batch_size_test = 8

input_dim = 40
hidden_dim = 64
num_layers = 2
lr = 1e-3

DATA_TRAIN = '../data/features/train'
DATA_TEST = '../data/features/test'

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class SpeechDataset(Dataset):
    """Concatenated LibriSpeech uterrance dataset."""

    def __init__(self, root_dir):
        """
        Args:
            root_dir (string): The path to the topmost directory of the dataset.
        """

        self.file_list = []

        # first load the paths to the feature files
        with os.scandir(root_dir) as folders:
            for folder in folders:
                for f in glob(folder.path + '/*.fea'):
                    self.file_list.append(f)
        self.n_utterances = len(self.file_list)

    def __len__(self):
        return self.n_utterances

    def __getitem__(self, index):
        with open(self.file_list[index], 'rb') as f:
            x, y = pickle.load(f) # (x, y) == (logfbanks, labels)
            x = torch.from_numpy(x).float()
            y = torch.from_numpy(y).float()
            f.close()
            return x, y
        return None

class Net(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(Net, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers  = num_layers

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out_packed, _ = self.lstm(x)
        out_padded, out_lengths = pad_packed_sequence(out_packed, batch_first=True)

        out_padded = self.fc(out_padded)
        out_padded = self.sigmoid(out_padded)
        return out_padded

def pad_collate(batch):
    (xx, yy) = zip(*batch)
    x_lens = [len(x) for x in xx]
    y_lens = [len(y) for y in yy]

    xx_pad = pad_sequence(xx, batch_first=True, padding_value=0)
    yy_pad = pad_sequence(yy, batch_first=True, padding_value=0)

    return xx_pad, yy_pad, x_lens, y_lens

# Load the data and create DataLoader instances
train_data = SpeechDataset(DATA_TRAIN)
test_data = SpeechDataset(DATA_TEST)
train_loader = DataLoader(
        dataset=train_data, batch_size=batch_size, shuffle=True, collate_fn=pad_collate)
test_loader = DataLoader(
        dataset=test_data, batch_size=batch_size_test, shuffle=False, collate_fn=pad_collate)

model = Net(input_dim, hidden_dim, num_layers).to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

# Train!!! hype!!!
for epoch in range(num_epochs):
    print(f"====== Starting epoch {epoch} ======")
    for batch, (x_padded, y_padded, x_lens, y_lens) in enumerate(train_loader):
        x_packed = pack_padded_sequence(
                x_padded, x_lens, batch_first=True, enforce_sorted=False).to(device)
        out_padded = model(x_packed)
        
        batch_loss = 0.0
        y_padded = y_padded.to(device)
        for j in range(out_padded.size(0)):
            loss = criterion(out_padded[j][:y_lens[j]], torch.unsqueeze(y_padded[j][:y_lens[j]], 1))
            batch_loss += loss
        batch_loss /= batch_size
        batch_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        if batch % 10 == 0:
            print(f'Batch: {batch}, loss = {batch_loss.item():.4f}')

    scheduler.step() # learning rate adjust

    # Test the model after each epoch
    with torch.no_grad():
        print("testing...")
        n_correct = 0
        n_samples = 0
        for x_padded, y_padded, x_lens, y_lens in test_loader:
            x_packed = pack_padded_sequence(
                    x_padded, x_lens, batch_first=True, enforce_sorted=False).to(device)
            out_padded = model(x_packed)
            y_padded = y_padded.to(device)

            # value, index
            for j in range(out_padded.size(0)):
                predictions = (out_padded[j][:y_lens[j]] > 0.5).float().cuda()
                n_samples += y_lens[j]
                n_correct += torch.sum(predictions.squeeze() == y_padded[j][:y_lens[j]]).item()

        acc = 100.0 * n_correct / n_samples
        print(f"accuracy = {acc:.2f}")
