"""Regression tests for past-only SOH forecasting on NASA and CALCE."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from simple_soh import FEATURES, MODEL_TYPES
from simple_soh.data import fit_scaler, load_cell, load_split, make_windows
from simple_soh.models import build_model
from simple_soh.run import metrics_for

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"


@pytest.mark.parametrize(
    "dataset,expected",
    [("nasa", (168, 168, 168, 132)), ("calce", (882, 973, 1038, 1028))],
)
def test_real_battery_inventory_and_windows(dataset, expected):
    cfg = json.loads(
        (ROOT / "configs" / f"simple_soh_{dataset}.json").read_text()
    )
    split = load_split(DATA, dataset, cfg["split"])
    counts = tuple(len(cell) for key in ("train", "validation", "test")
                   for cell in split[key])
    assert counts == expected
    assert len(set(sum((cfg["split"][part]
                        for part in ("train", "validation", "test")), []))) == 4

    scaler = fit_scaler(split["train"])
    raw_train = np.concatenate([cell[list(FEATURES)].to_numpy()
                                for cell in split["train"]])
    np.testing.assert_allclose(scaler.mean_, raw_train.mean(axis=0))
    x_train, y_train, meta_train = make_windows(split["train"], scaler, 12)
    x_val, y_val, meta_val = make_windows(split["validation"], scaler, 12)
    x_test, y_test, meta_test = make_windows(split["test"], scaler, 12)
    assert x_train.shape == (sum(expected[:2]) - 24, 12, len(FEATURES))
    assert len(y_val) == expected[2] - 12
    assert len(y_test) == expected[3] - 12
    assert len(meta_test) == len(y_test)
    assert (meta_test["last_observed_cycle"] < meta_test["cycle"]).all()
    assert meta_test["cycle"].iloc[0] == 13
    assert "capacity_ah" not in FEATURES
    assert "soh_first_observed" not in FEATURES
    assert "remaining_observed_cycles" not in FEATURES
    assert "target" not in FEATURES
    assert np.isfinite(x_test).all()
    assert np.isfinite(y_test).all()
    assert set(meta_train.cell_id).isdisjoint(set(meta_val.cell_id))
    assert set(meta_train.cell_id).isdisjoint(set(meta_test.cell_id))
    assert set(meta_val.cell_id).isdisjoint(set(meta_test.cell_id))


@pytest.mark.parametrize("model_type", MODEL_TYPES)
def test_model_forward_backward(model_type):
    torch.manual_seed(7)
    net = build_model(model_type, len(FEATURES), 12, hidden=16, dropout=0.1)
    x = torch.randn(4, 12, len(FEATURES))
    y = net(x)
    assert tuple(y.shape) == (4,)
    y.square().mean().backward()
    assert any(p.grad is not None for p in net.parameters())


def test_cell_source_columns_and_target_consistency():
    nasa = load_cell(DATA, "nasa", "B0005")
    calce = load_cell(DATA, "calce", "CS2_35")
    assert len(nasa) == 168 and len(calce) == 882
    assert nasa["temperature_mean"].notna().all()
    assert calce["temperature_mean"].isna().all()
    np.testing.assert_allclose(
        calce["soh_first_observed"], calce["capacity_ah"] / calce["capacity_ah"].iat[0]
    )


def test_battery_overlap_is_rejected():
    groups = {"train": ["B0005"], "validation": ["B0005"], "test": ["B0018"]}
    with pytest.raises(ValueError, match="multiple"):
        load_split(DATA, "nasa", groups)


def test_metrics_are_consistent():
    x = np.array([.85, .9, 1.0])
    y = x.copy()
    score = metrics_for(x, y)
    assert score["mae"] == 0
    assert score["rmse"] == 0
    assert score["r2"] == 1
    assert score["test_windows"] == 3
