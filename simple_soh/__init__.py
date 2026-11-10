"""Battery-wise SOH prediction with compact neural networks."""

FEATURES = (
    "voltage_mean",
    "voltage_std",
    "current_mean",
    "current_std",
    "discharge_duration_s",
)

MODEL_TYPES = ("mlp", "lstm", "gru", "cnn_lstm")
