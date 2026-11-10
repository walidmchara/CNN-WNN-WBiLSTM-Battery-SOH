"""Run a paper-parameter CNN-WNN-wavelet-BiLSTM-attention reconstruction.

Examples:
python -m paper_reproduction.run --group nasa_24c --source-root /content/drive/MyDrive/NASA_original_source/mat --epochs 58
python -m paper_reproduction.run --group calce_cs2 --source-root /content/drive/MyDrive/CALCE_original_source --epochs 58
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from training_recovery import EpochCheckpoint
from . import PAPER_DOI, PAPER_SETTINGS, PaperCNNWNNWBiLSTMAM
from .data import GROUPS, make_paper_split


PUBLISHED = {
    "nasa_24c": {"cell": "B0018", "mae": 0.0036, "rmse": 0.0041, "mape_entry": 0.0042},
    "nasa_43c": {"cell": "B0030", "mae": 0.0032, "rmse": 0.0039, "mape_entry": 0.0031},
    "nasa_4c": {"cell": "B0056", "mae": 0.0029, "rmse": 0.0046, "mape_entry": 0.0039},
    "calce_cs2": {"cell": "CS2_38", "mae": 0.0088, "rmse": 0.0097, "mape_entry": 0.0098},
}


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def predict_batches(model, features, device, batch_size=128):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(features), batch_size):
            sample = torch.from_numpy(features[i:i + batch_size]).to(device)
            out.append(model(sample).detach().cpu().numpy().reshape(-1))
    return np.concatenate(out).astype(np.float64)


def metrics(target, pred):
    true = np.asarray(target, dtype=np.float64)
    guess = np.asarray(pred, dtype=np.float64)
    if not np.isfinite(true).all() or not np.isfinite(guess).all():
        raise ValueError("Non-finite target or prediction")
    error = guess - true
    mse = float(np.mean(error**2))
    var = float(np.sum((true - true.mean())**2))
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(math.sqrt(mse)),
        "mse": mse,
        "r2": float(1 - np.sum(error**2)/var) if var > 0 else None,
        "mape_percent": float(100*np.mean(np.abs(error/true))) if np.all(true > 0) else None,
        "bias_pred_minus_actual": float(np.mean(error)),
        "n_cycles": int(len(true)),
    }


def draw_figures(output, predictions, history):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 4))
    ax.plot(predictions["cycle"], predictions["soh_actual"], label="Measured SOH")
    ax.plot(predictions["cycle"], predictions["soh_predicted"], label="Estimated SOH")
    ax.set(xlabel="Discharge cycle", ylabel="SOH: capacity / rated capacity",
           title="CNN-WNN-WBiLSTM-AM held-out battery SOH")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "soh_prediction.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 4))
    ax.plot([x["epoch"] for x in history], [x["train_mse"] for x in history],
            label="Development")
    ax.plot([x["epoch"] for x in history], [x["validation_mse"] for x in history],
            label="Validation")
    ax.set(xlabel="Epoch", ylabel="MSE",
           title="Training and validation convergence")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "convergence.png", dpi=170)
    plt.close(fig)

    actual = predictions["soh_actual"].to_numpy()
    estimated = predictions["soh_predicted"].to_numpy()
    lo, hi = float(min(actual.min(), estimated.min())), float(max(actual.max(), estimated.max()))
    fig, ax = plt.subplots(figsize=(5.4, 5.4))
    ax.scatter(actual, estimated, s=14)
    ax.plot([lo, hi], [lo, hi], color="black", linestyle="--")
    ax.set(xlabel="Measured SOH", ylabel="Estimated SOH", title="Test-battery parity")
    fig.tight_layout()
    fig.savefig(output / "parity.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 4))
    ax.plot(predictions["cycle"], estimated-actual)
    ax.axhline(0, color="black", linestyle="--")
    ax.set(xlabel="Discharge cycle", ylabel="Predicted − measured SOH",
           title="Signed residuals")
    fig.tight_layout()
    fig.savefig(output / "residuals.png", dpi=170)
    plt.close(fig)


def run(group: str, source_root: Path, cache_root: Path,
        output_root: Path, checkpoint_root: Path, *, points: int = 64,
        epochs: int = 58, batch_size: int = 32, seed: int = 42,
        resume: bool = False, device_name: str = "auto",
        validation_fraction: float = 0.15):
    if group not in GROUPS:
        raise ValueError(f"Unknown group {group}")
    if epochs < 1 or points < 16 or batch_size < 1:
        raise ValueError("Invalid training configuration")
    seed_all(seed)
    prepared = make_paper_split(group, source_root, cache_root, points,
                                validation_fraction=validation_fraction)

    (xtrain, ytrain, _), (xval, yval, _), (xtest, ytest, testmeta) = (
        prepared["train"], prepared["validation"], prepared["test"]
    )
    run_id = f"automatika2025_{group}_p{points}_e{epochs}_seed{seed}"
    output = Path(output_root) / run_id
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(checkpoint_root) / run_id
    metadata = {
        "paper_doi": PAPER_DOI,
        "reference_model": "CNN-WNN-WBiLSTM-AM",
        "group": group,
        "source_data": ("original_nasa_mat" if prepared["dataset"] == "nasa"
                        else "original_calce_xlsx"),
        "source_sha256": prepared["source_hashes"],
        "train_cells": prepared["experiment_cells"]["train"],
        "test_cells": prepared["experiment_cells"]["test"],
        "validation_fraction": validation_fraction,
        "points": points, "epochs": epochs, "seed": seed,
        "batch_size": batch_size, "architecture": PAPER_SETTINGS,
        "label": "capacity_ah / nominal_capacity_ah",
        "nominal_capacity_ah": prepared["nominal_capacity_ah"],
        "preprocessing": "Train-only per-channel MinMax; per-cycle time resampling",
    }
    recovery = EpochCheckpoint(checkpoint, metadata)
    if recovery.path.exists() and not resume:
        raise FileExistsError(
            f"Existing training checkpoint: {recovery.path}. "
            "Use --resume to continue or re-evaluate without retraining."
        )

    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device("cuda" if device_name == "auto" and torch.cuda.is_available()
                          else "cpu" if device_name == "auto" else device_name)

    model = PaperCNNWNNWBiLSTMAM(
        input_channels=4, cnn_filters=64, wnn_units=127,
        wbilstm_units=94, attention_units=188, dropout=0.3,
    ).to(device)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(xtrain), torch.from_numpy(ytrain)),
        batch_size=batch_size, shuffle=True,
    )
    val_tensors = (
        torch.from_numpy(xval).to(device),
        torch.from_numpy(yval).to(device),
    )
    history, best_mse, best_epoch, best_state, start = [], math.inf, 0, None, 0
    if resume:
        state = recovery.restore({"model": model, "optimizer": optimizer},
                                 device=device)
        start = int(state["epoch"])
        history = state["history"]
        best_mse = float(state["best_validation_mse"])
        best_epoch = int(state["best_epoch"])
        best_state = state["best_model"]
        print(f"Resuming from completed epoch {start}", flush=True)
    criterion = nn.MSELoss()
    for epoch in range(start + 1, epochs + 1):
        model.train()
        total = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            fitted = model(x)
            loss = criterion(fitted, y)
            if not torch.isfinite(loss).item():
                raise RuntimeError(f"Non-finite training MSE at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach().item()) * len(y)
        model.eval()
        with torch.no_grad():
            losses = []
            for k in range(0, len(xval), 128):
                estimate = model(val_tensors[0][k:k + 128])
                reference = val_tensors[1][k:k + 128]
                losses.append(float(((estimate-reference)**2).sum().item()))
            validation_mse = float(sum(losses) / len(xval))
        training_mse = float(total / len(ytrain))
        history.append({
            "epoch": epoch, "train_mse": training_mse,
            "validation_mse": validation_mse,
        })
        if validation_mse < best_mse:
            best_mse, best_epoch = validation_mse, epoch
            best_state = {name: weights.detach().cpu().clone()
                          for name, weights in model.state_dict().items()}
        recovery.save(epoch, {"model": model, "optimizer": optimizer}, {
            "history": history, "best_validation_mse": best_mse,
            "best_epoch": best_epoch, "best_model": best_state,
        })
        print(f"Epoch {epoch:03d}/{epochs}: train MSE={training_mse:.6f} "
              f"validation MSE={validation_mse:.6f}", flush=True)

    if best_state is None:
        raise RuntimeError("No validated model state to evaluate")
    model.load_state_dict(best_state)
    estimates = predict_batches(model, xtest, device)
    result_metrics = metrics(ytest, estimates)
    predictions = pd.DataFrame(testmeta)
    predictions["soh_actual"] = ytest
    predictions["soh_predicted"] = estimates
    predictions.to_csv(output / "test_predictions.csv", index=False)
    pd.DataFrame(history).to_csv(output / "training_history.csv", index=False)
    draw_figures(output, predictions, history)

    model_package = {
        "model": best_state,
        "paper_settings": PAPER_SETTINGS,
        "input_scaling_min": prepared["scaling_min"],
        "input_scaling_max": prepared["scaling_max"],
        "label_nominal_ah": prepared["nominal_capacity_ah"],
        "best_epoch": best_epoch, "metadata": metadata,
    }
    torch.save(model_package, output / "best_model.pt")
    original = PUBLISHED[group]
    report = {
        "status": "new_executed_reconstruction_not_exact_original_source",
        "paper_doi": PAPER_DOI,
        "experimental_group": group,
        "input_type": metadata["source_data"],
        "train_cells": prepared["experiment_cells"]["train"],
        "test_cell": original["cell"],
        "n_train": int(len(ytrain)),
        "n_validation": int(len(yval)),
        "n_test": int(len(ytest)),
        "epochs_completed": epochs,
        "best_epoch": best_epoch, "best_validation_mse": best_mse,
        "new_run_test": result_metrics,
        "paper_published_reference": original,
        "methodological_nonidentities": [
            "Original code unavailable; trainable Morlet implementation reconstructed.",
            "Original input trace sample count unspecified; linear interpolation to fixed points.",
            "Validation fraction and seed unspecified; fixed development-only holdout.",
            "Train-only scaling scope adopted to avoid evaluating with test statistics.",
            "Framework is PyTorch; original publication used Keras/TensorFlow.",
            "CALCE has no measured temperature in inspected Channel_ source; channel zero-filled."
        ],
    }
    (output / "metrics.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (output / "provenance.json").write_text(
        json.dumps({
            **metadata,
            "source_qa": prepared["source_qa"],
            "scaling_min": prepared["scaling_min"],
            "scaling_max": prepared["scaling_max"],
        }, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"group": group, "new_run_test": result_metrics,
                      "published_reference": original}, indent=2), flush=True)
    print(f"Run completed: {output}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=sorted(GROUPS), required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, default=Path("outputs/paper_cache"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/paper_results"))
    parser.add_argument("--checkpoint-root", type=Path, default=Path("outputs/paper_checkpoints"))
    parser.add_argument("--points", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=58)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--resume", action="store_true")
    a = parser.parse_args()
    run(a.group, a.source_root, a.cache_root, a.output_root, a.checkpoint_root,
        points=a.points, epochs=a.epochs, batch_size=a.batch_size, seed=a.seed,
        resume=a.resume, device_name=a.device,
        validation_fraction=a.validation_fraction)


if __name__ == "__main__":
    main()
