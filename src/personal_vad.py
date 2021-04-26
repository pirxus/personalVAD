import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.nn.functional import one_hot
import torch.nn.functional as F

class PersonalVAD(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, out_dim, use_fc=True, linear=False):
        super(PersonalVAD, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.out_dim = out_dim
        self.use_fc = use_fc
        self.linear = linear

        # define the model layers...
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

        # use the original PersonalVAD configuration with one additional layer
        if use_fc:
            self.fc1 = nn.Linear(hidden_dim, hidden_dim)
            if not self.linear:
                self.tanh = nn.Tanh()
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, x_lens, hidden):
        # first pack the padded sequences
        x_packed = pack_padded_sequence(x, x_lens, batch_first=True, enforce_sorted=False)
        # lstm pass
        out_packed, hidden = self.lstm(x_packed, hidden)
        out_padded, _ = pad_packed_sequence(out_packed, batch_first=True)

        # pass them through an additional layer if specified...
        if self.use_fc:
            out_padded = self.fc1(out_padded)
            if not self.linear:
                out_padded = self.tanh(out_padded)

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

    For better understanding of the loss function, check out either the original paper,
    or my thesis...

    """

    def __init__(self, weights=torch.tensor([1.0, 0.1, 1.0])):
        super(WPL, self).__init__()
        self.weights = weights
        assert len(weights) == 3, "The wpl is defined for three classes only."

    def forward(self, output, target):
        output = torch.exp(output)
        label_mask = one_hot(target) > 0.5 # boolean mask
        label_mask_r1 = torch.roll(label_mask, 1, 1) # if ntss, then tss
        label_mask_r2 = torch.roll(label_mask, 2, 1) # if ntss, then ns

        # get the probability of the actual label and the other two into one array
        actual = torch.masked_select(output, label_mask)
        plus_one = torch.masked_select(output, label_mask_r1)
        minus_one = torch.masked_select(output, label_mask_r2)

        # arrays of the first pair weight and the second pair weight used in the equation
        w1 = torch.masked_select(self.weights, label_mask) # if ntss, w1 is <ntss, ns>
        w2 = torch.masked_select(self.weights, label_mask_r1) # if ntss, w2 is <tss, ntss>

        # first pair
        first_pair = w1 * torch.log(actual / (actual + minus_one))
        second_pair = w2 * torch.log(actual / (actual + plus_one))

        # get the negative mean value for the two pairs
        wpl = -0.5 * (first_pair + second_pair)

        # sum and average for minibatch
        return torch.mean(wpl) 

def pad_collate(batch):
    """Padding function used to deal with batches of sequences of variable lengths.

    Returns:
        tuple: A tuple containing:
            
            xx_pad (torch.tensor): Padded feature vector.
            yy_pad (torch.tensor): Padded ground truth vector.
            x_lens (torch.tensor): Lengths of the original feature vectors within the batch.
            y_lens (torch.tensor): Lengths of the original ground truth vectors within the batch.

    """

    (xx, yy) = zip(*batch)
    x_lens = [len(x) for x in xx]
    y_lens = [len(y) for y in yy]

    xx_pad = pad_sequence(xx, batch_first=True, padding_value=0)
    yy_pad = pad_sequence(yy, batch_first=True, padding_value=0)
    # TODO: yy_padding is not necessary in this case....

    return xx_pad, yy_pad, x_lens, y_lens
