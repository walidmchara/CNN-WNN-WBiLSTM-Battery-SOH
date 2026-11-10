"""Compact regression networks for battery SOH forecasting."""
from __future__ import annotations

import torch
from torch import nn

from . import MODEL_TYPES


class MLPRegressor(nn.Module):
    def __init__(self, n_features: int, lookback: int, hidden: int, dropout: float):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_features * lookback, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.network(x).squeeze(-1)


class RecurrentRegressor(nn.Module):
    def __init__(
        self, cell: str, n_features: int, hidden: int, dropout: float
    ):
        super().__init__()
        recurrent = nn.LSTM if cell == "lstm" else nn.GRU
        self.encoder = recurrent(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=1,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        sequence, _ = self.encoder(x)
        return self.head(sequence[:, -1]).squeeze(-1)


class CNNLSTMRegressor(nn.Module):
    def __init__(self, n_features: int, hidden: int, dropout: float):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_features, hidden, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.encoder = nn.LSTM(hidden, hidden, batch_first=True)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        features = self.conv(x.transpose(1, 2)).transpose(1, 2)
        sequence, _ = self.encoder(features)
        return self.head(sequence[:, -1]).squeeze(-1)


def build_model(name: str, n_features: int, lookback: int,
                hidden: int = 32, dropout: float = 0.1) -> nn.Module:
    if name not in MODEL_TYPES:
        raise ValueError(f"Unknown model {name}; choose from {MODEL_TYPES}")
    if hidden <= 0 or not (0 <= dropout < 1):
        raise ValueError("Invalid model hyperparameters")
    if name == "mlp":
        return MLPRegressor(n_features, lookback, hidden, dropout)
    if name in ("lstm", "gru"):
        return RecurrentRegressor(name, n_features, hidden, dropout)
    return CNNLSTMRegressor(n_features, hidden, dropout)
