"""CNN → trainable Morlet WNN → bidirectional wavelet LSTM → attention.

Paper constants: Automatika 66(2) (2025), Table 2 and equations (3)–(21).
The original authors' training source is unavailable. The explicit recurrent
cell, score-network topology and input resampling are reconstruction choices.
"""
from __future__ import annotations

import math
import torch
from torch import nn


def morlet(x: torch.Tensor) -> torch.Tensor:
    """Mother wavelet, Equation (6): cos(1.75*x) * exp(-x² / 2)."""
    return torch.cos(1.75 * x) * torch.exp(-0.5 * x.square())


class TrainableMorlet(nn.Module):
    """Trainable wavelet dilation and translation, equations (3)–(6)."""

    def __init__(self, units: int):
        super().__init__()
        self.log_scale = nn.Parameter(torch.zeros(units))
        self.translation = nn.Parameter(torch.zeros(units))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Positive scale from exponential; clamp prevents overflow in extreme runs.
        scale = self.log_scale.clamp(-3.0, 3.0).exp()
        return morlet((x - self.translation) / scale)


class WaveletLSTMCell(nn.Module):
    """Morlet replacing sigmoid/tanh in all four gates (paper Eqs. 13–17).

    Negative wavelet gate values are permitted, exactly as written in the
    published equations. An ordinary LSTM would not reproduce this equation.
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.gates = nn.Linear(input_size + hidden_size, 4 * hidden_size)
        # Standard LSTM-like initialization of recurrent affine transforms.
        nn.init.xavier_uniform_(self.gates.weight)
        nn.init.zeros_(self.gates.bias)

    def forward(self, x, hidden, cell):
        f, i, o, candidate = self.gates(torch.cat((x, hidden), dim=-1)).chunk(4, -1)
        f, i, o, candidate = map(morlet, (f, i, o, candidate))
        cell = f * cell + i * candidate
        hidden = o * morlet(cell)
        return hidden, cell


class BidirectionalWaveletLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.forward_cell = WaveletLSTMCell(input_size, hidden_size)
        self.backward_cell = WaveletLSTMCell(input_size, hidden_size)
        self.hidden_size = hidden_size

    def _scan(self, cell, x, reverse=False):
        n, steps, _ = x.shape
        hidden = x.new_zeros(n, self.hidden_size)
        state = x.new_zeros(n, self.hidden_size)
        outputs = []
        iterator = range(steps - 1, -1, -1) if reverse else range(steps)
        for step in iterator:
            hidden, state = cell(x[:, step, :], hidden, state)
            outputs.append(hidden)
        if reverse:
            outputs.reverse()
        return torch.stack(outputs, dim=1)

    def forward(self, x):
        forward = self._scan(self.forward_cell, x)
        backward = self._scan(self.backward_cell, x, reverse=True)
        return torch.cat((forward, backward), dim=-1)


class Attention188(nn.Module):
    """188-unit sigmoid attention scorer, then time-step softmax, Eqs. 19–21."""

    def __init__(self, channels: int = 188, attention_units: int = 188):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(channels, attention_units),
            nn.Sigmoid(),
            nn.Linear(attention_units, 1),
        )

    def forward(self, sequence, return_weights=False):
        scores = self.scorer(sequence)
        probabilities = torch.softmax(scores, dim=1)
        context = (sequence * probabilities).sum(dim=1)
        if return_weights:
            return context, probabilities.squeeze(-1)
        return context


class PaperCNNWNNWBiLSTMAM(nn.Module):
    def __init__(
        self, input_channels: int = 4, cnn_filters: int = 64,
        wnn_units: int = 127, wbilstm_units: int = 94,
        attention_units: int = 188, dropout: float = 0.3,
    ):
        super().__init__()
        self.conv = nn.Conv1d(input_channels, cnn_filters, kernel_size=3,
                              padding="same")
        self.activation = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.dropout = nn.Dropout(dropout)
        self.wnn_projection = nn.Linear(cnn_filters, wnn_units)
        self.wnn_activation = TrainableMorlet(wnn_units)
        self.wbilstm = BidirectionalWaveletLSTM(wnn_units, wbilstm_units)
        self.attention = Attention188(2 * wbilstm_units, attention_units)
        self.head = nn.Sequential(nn.Linear(2 * wbilstm_units, 1), nn.ReLU())

    def forward(self, x, return_attention=False):
        if x.ndim != 3 or x.shape[-1] != self.conv.in_channels or x.shape[1] < 4:
            raise ValueError("Expected (batch, >=4 time samples, four V/I/T/t features)")
        h = self.dropout(self.pool(self.activation(self.conv(x.transpose(1, 2)))))
        h = h.transpose(1, 2)
        h = self.wnn_activation(self.wnn_projection(h))
        h = self.wbilstm(h)
        if return_attention:
            context, weights = self.attention(h, return_weights=True)
            return self.head(context).squeeze(-1), weights
        return self.head(self.attention(h)).squeeze(-1)
