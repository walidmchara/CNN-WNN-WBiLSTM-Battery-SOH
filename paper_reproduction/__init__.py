"""Reconstruction of the original Automatika 2025 CNN–WNN–WBiLSTM–AM methodology.

The original executable implementation is unavailable. Exact paper settings
are distinguished from implementation choices needed to run a new experiment.
"""
from .models import PaperCNNWNNWBiLSTMAM

PAPER_DOI = "10.1080/00051144.2025.2461821"
PAPER_SETTINGS = {
    "cnn_filters": 64, "kernel_size": 3, "pool_size": 2, "dropout": 0.3,
    "wnn_units": 127, "wbilstm_units": 94, "attention_units": 188,
    "attention_activation": "sigmoid", "optimizer": "RMSprop",
    "learning_rate": 0.001, "epochs": 58, "batch_size": 32,
    "loss": "MSE", "output_activation": "ReLU",
}
