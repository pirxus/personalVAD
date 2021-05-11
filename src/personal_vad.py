"""@package personal_vad

Author: Simon Sedlacek
Email: xsedla1h@stud.fit.vutbr.cz

This is the main Personal VAD module. This module contains the PersonalVAD
model class, the definition of the Weighted Pairwise Loss (WPL) and the
padding function used by the data loaders while training.

"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.nn.functional import one_hot

class PersonalVAD(nn.Module):
    """Personal VAD model class. """

    def __init__(self, input_dim, hidden_dim, num_layers, out_dim, use_fc=True, linear=False):
        """PersonalVAD class initializer.

        Args:
            input_dim (int): Input feature vector size.
            hidden_dim (int): LSTM hidden state size.
            num_layers (int): Number of LSTM layers in the model.
            out_dim (int): Number of neurons in the output layer.
            use_fc (bool, optional): Specifies, whether the model should use the
                last fully-connected hidden layer. Defaults to True.
            linear (bool, optional): Specifies the activation function used by the last
                hidden layer. If False, the tanh is used, if True, no activation is
                used. Defaults to False.
        """

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

    def forward(self, x, x_lens, hidden=None):
        """Personal VAD model forward pass method.

        Args:
            x (torch.tensor): Input feature batch. The individual feature sequences are padded.
            x_lens (list of int): A list of the original pre-padding feature sequence lengths.
            hidden (tuple of torch.tensor, optional): The hidden state value to be used by the LSTM as
                the initial hidden state. Defaults to None.

        Returns:
            tuple: tuple containing:
                out_padded (torch.tensor): Tensor of tensors containing the network predictions.
                    The dimensionality of the output prediction depends on the out_dim attribute.
                hidden (tuple of torch.tensor): Tuple containing the last hidden and cell state
                    values for each processed sequence.
        """

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

class WPL(nn.Module):
    """Weighted pairwise loss implementation for three classes.

    The weight pairs are interpreted as follows:
    [<ns,tss> ; <ntss,ns> ; <tss,ntss>]

    Target labels contain indices, the model output is a tensor of probabilites for each class.
    (ns, ntss, tss) -> {0, 1, 2} 

    For better understanding of the loss function, check out either the original Personal VAD
    paper at https://arxiv.org/abs/1908.04284, or, alternatively, my thesis :)
    """

    def __init__(self, weights=torch.tensor([1.0, 0.5, 1.0])):
        """Initialize the WPL class.

        Args:
            weights (torch.tensor, optional): The weight values for each class pair.
        """

        super(WPL, self).__init__()
        self.weights = weights
        assert len(weights) == 3, "The wpl is defined for three classes only."

    def forward(self, output, target):
        """Compute the WPL for a sequence.

        Args:
            output (torch.tensor): A tensor containing the model predictions.
            target (torch.tensor): A 1D tensor containing the indices of the target classes.

        Returns:
            torch.tensor: A tensor containing the WPL value for the processed sequence.
        """

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
    # NOTE: yy_padding is not necessary in this case....

    return xx_pad, yy_pad, x_lens, y_lens
