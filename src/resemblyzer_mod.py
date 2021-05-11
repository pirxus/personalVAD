"""@package resemblyzer_mod

Author: Simon Sedlacek
Email: xsedla1h@stud.fit.vutbr.cz

This module contains an extension of the base Resemblyzer VoiceEncoder class,
modified to support streaming, frame-level inference, returning one d-vector for
each individual frame in the input feature vector.

"""

import torch
from resemblyzer import VoiceEncoder

class VoiceEncoderMod(VoiceEncoder):
    def __init__(self):
        super().__init__()

    def forward_stream(self, x, hidden=None):
        """Modified VoiceEncoder forward method.

        Return embeddings for all input frames. 

        Args:
            x (torch.tensor): Input feature vector batch.
            hidden (torch.tensor, optional): LSTM hidden state initialization
                value. Defaults to None.

        Returns:
            tuple: A tuple containing:
                embeddings (torch.tensor): L2 normalized d-vectors for each input frame.
                hidden (torch.tensor): The last value of the LSTM hidden and cell states.
        """

        out, hidden = self.lstm(x, hidden)
        embeds_raw = self.relu(self.linear(out[:,:]))
        norm = torch.norm(embeds_raw, dim=2, keepdim=True)
        return embeds_raw / norm, hidden
