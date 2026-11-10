"""Source and architecture checks for the original-paper reconstruction."""
from __future__ import annotations

import json
import numpy as np
import pytest
import torch

from paper_reproduction import PAPER_SETTINGS, PaperCNNWNNWBiLSTMAM
from paper_reproduction.data import GROUPS, resample, make_paper_split
from paper_reproduction.models import morlet
from paper_reproduction.run import metrics, PUBLISHED


def test_model_matches_published_table2_shapes_and_gradients():
    torch.manual_seed(42)
    net = PaperCNNWNNWBiLSTMAM()
    assert net.conv.kernel_size == (3,)
    assert net.conv.out_channels == PAPER_SETTINGS["cnn_filters"] == 64
    assert net.pool.kernel_size == PAPER_SETTINGS["pool_size"] == 2
    assert net.wnn_projection.out_features == PAPER_SETTINGS["wnn_units"] == 127
    assert net.wbilstm.hidden_size == PAPER_SETTINGS["wbilstm_units"] == 94
    assert net.attention.scorer[0].out_features == PAPER_SETTINGS["attention_units"] == 188
    assert net.dropout.p == 0.3
    x = torch.rand(3, 32, 4)
    predictions, attention = net(x, return_attention=True)
    assert predictions.shape == (3,)
    assert attention.shape == (3, 16)
    torch.testing.assert_close(attention.sum(1), torch.ones(3), atol=1e-5, rtol=0)
    assert torch.isfinite(predictions).all()
    (predictions - torch.tensor([0.7, 0.9, 0.8])).square().mean().backward()
    assert net.wnn_activation.log_scale.grad is not None
    assert net.wnn_activation.translation.grad is not None
    assert net.wbilstm.forward_cell.gates.weight.grad is not None
    assert net.wbilstm.backward_cell.gates.weight.grad is not None


def test_morlet_equation6():
    t = torch.tensor([0.0, 1.0, -1.0])
    expected = torch.cos(1.75*t) * torch.exp(-0.5*t.square())
    torch.testing.assert_close(morlet(t), expected)
    assert float(morlet(torch.zeros(1)).item()) == 1.0


def test_resample_inputs_are_ordered_V_I_T_time_not_capacity():
    raw = resample(
        [0, 10, 20, 30], [4.0, 3.9, 3.8, 3.7],
        [-2.0, -2.0, -2.0, -2.0], [22.0, 23.0, 24.0, 25.0],
        points=16,
    )
    assert raw.shape == (16, 4)
    np.testing.assert_allclose(raw[0], [4.0, -2.0, 22.0, 0.0])
    np.testing.assert_allclose(raw[-1], [3.7, -2.0, 25.0, 30.0])
    assert not np.isnan(raw).any()


def test_paper_groups_have_disjoint_train_test_and_complete_table3():
    assert sorted(GROUPS) == sorted(PUBLISHED)
    for name, cfg in GROUPS.items():
        assert len(cfg["train"]) == 3
        assert len(cfg["test"]) == 1
        assert set(cfg["train"]).isdisjoint(cfg["test"])
        assert PUBLISHED[name]["mae"] > 0
        assert PUBLISHED[name]["rmse"] >= PUBLISHED[name]["mae"]
    assert PAPER_SETTINGS["epochs"] == 58
    assert PAPER_SETTINGS["learning_rate"] == .001
    assert PAPER_SETTINGS["optimizer"] == "RMSprop"


def test_reconstruction_metrics_are_observations_not_paper_results():
    y = np.asarray([.9, .8, .7], dtype=np.float64)
    observed = metrics(y, y.copy())
    assert observed["mae"] == 0
    assert observed["r2"] == 1
    assert observed["mape_percent"] == 0
    assert observed["n_cycles"] == 3
