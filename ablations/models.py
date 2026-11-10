"""R1 ablation: retain CNN, trainable Morlet WNN and attention; replace only recurrent cells.

This is NOT the paper architecture and must not be presented as its exact reproduction.
"""
import torch
from torch import nn
from paper_reproduction.models import PaperCNNWNNWBiLSTMAM


class CNNWNNBiLSTMAM(PaperCNNWNNWBiLSTMAM):
    def __init__(self):
        super().__init__()
        self.wbilstm = nn.LSTM(
            input_size=127, hidden_size=94, num_layers=1,
            batch_first=True, bidirectional=True
        )

    def forward(self, x, return_attention=False):
        if x.ndim != 3 or x.shape[-1] != 4 or x.shape[1] < 4:
            raise ValueError("Expected (batch, time, 4) inputs")
        h = self.dropout(self.pool(self.activation(self.conv(x.transpose(1, 2)))))
        h = h.transpose(1, 2)
        h = self.wnn_activation(self.wnn_projection(h))
        h, _ = self.wbilstm(h)
        if return_attention:
            context, weights = self.attention(h, return_weights=True)
            return self.head(context).squeeze(-1), weights
        return self.head(self.attention(h)).squeeze(-1)
