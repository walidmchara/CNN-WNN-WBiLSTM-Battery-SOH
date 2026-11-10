"""Train and evaluate simple one-cycle-ahead SOH models on held-out cells."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import random
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from training_recovery import EpochCheckpoint
from . import FEATURES, MODEL_TYPES
from .data import fit_scaler, load_split, make_windows
from .models import build_model

ROOT = Path(__file__).resolve().parents[1]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model, x: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    outputs = []
    with torch.no_grad():
        for offset in range(0, len(x), 512):
            batch = torch.as_tensor(
                x[offset:offset + 512], dtype=torch.float32, device=device
            )
            outputs.append(model(batch).cpu().numpy().reshape(-1))
    return np.concatenate(outputs)


def metrics_for(actual: np.ndarray, predicted: np.ndarray) -> dict:
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Non-finite test prediction")
    mse = float(mean_squared_error(actual, predicted))
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mse)),
        "mse": mse,
        "mape_percent": float(np.mean(np.abs((actual - predicted) / actual)) * 100),
        "r2": float(r2_score(actual, predicted)) if len(actual) > 1
                and float(np.var(actual)) > 0 else None,
        "mean_signed_error": float(np.mean(predicted - actual)),
        "test_windows": int(len(actual)),
    }


def figures(output: Path, history: list[dict], predictions: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for battery, frame in predictions.groupby("cell_id"):
        ax.plot(frame["cycle"], frame["actual_soh"], label=f"{battery} measured")
        ax.plot(frame["cycle"], frame["predicted_soh"], label=f"{battery} predicted", linestyle="--")
    ax.set(xlabel="Discharge cycle", ylabel="SOH (relative to first observed capacity)",
           title="Held-out-cell SOH: observed versus predicted")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(output / "soh_trajectory.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.plot([r["epoch"] for r in history], [r["train_mse"] for r in history], label="Training")
    ax.plot([r["epoch"] for r in history], [r["validation_mse"] for r in history], label="Validation")
    ax.set(xlabel="Epoch", ylabel="MSE", title="Training and validation loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "learning_curve.png", dpi=180)
    plt.close(fig)

    actual = predictions["actual_soh"].to_numpy()
    predicted = predictions["predicted_soh"].to_numpy()
    lo = float(min(actual.min(), predicted.min()))
    hi = float(max(actual.max(), predicted.max()))
    fig, ax = plt.subplots(figsize=(5.6, 5.3))
    ax.scatter(actual, predicted, s=12, alpha=0.65)
    ax.plot([lo, hi], [lo, hi], linestyle="--", color="black", linewidth=1)
    ax.set(xlabel="Measured SOH", ylabel="Predicted SOH", title="Test-cell parity")
    fig.tight_layout()
    fig.savefig(output / "parity.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    for battery, frame in predictions.groupby("cell_id"):
        ax.plot(frame["cycle"], frame["predicted_soh"] - frame["actual_soh"],
                label=battery)
    ax.axhline(0, linewidth=1, color="black")
    ax.set(xlabel="Discharge cycle", ylabel="Predicted − observed SOH",
           title="Test-cell signed residuals")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "residuals.png", dpi=180)
    plt.close(fig)


def run(config_path: Path, *, model_override: str | None, epochs_override: int | None,
        output_root: Path, checkpoint_root: Path, resume: bool, device_name: str) -> dict:
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    dataset = cfg["dataset"]
    if dataset not in {"nasa", "calce"}:
        raise ValueError("Only NASA and CALCE are supported")
    groups = cfg["split"]
    model_type = model_override or cfg["model"]["type"]
    if model_type not in MODEL_TYPES:
        raise ValueError("Unknown model")
    epochs = int(epochs_override if epochs_override is not None else cfg["training"]["epochs"])
    seed = int(cfg["training"]["seed"])
    lookback = int(cfg["forecasting"]["lookback"])
    if int(cfg["forecasting"]["horizon"]) != 1:
        raise ValueError("This benchmark evaluates one-cycle-ahead prediction only")
    if cfg["forecasting"]["target"] != "soh_first_observed":
        raise ValueError("Unexpected SOH target")
    if epochs < 1 or lookback < 2:
        raise ValueError("Invalid epochs or lookback")

    seed_everything(seed)
    data_root = ROOT / cfg["data_root"]
    frames = load_split(data_root, dataset, groups)
    scaler = fit_scaler(frames["train"])
    split = {name: make_windows(cells, scaler, lookback) for name, cells in frames.items()}
    (x_train, y_train, _), (x_val, y_val, _), (x_test, y_test, test_meta) = (
        split["train"], split["validation"], split["test"]
    )

    run_id = f"{dataset}_{model_type}_lb{lookback}_e{epochs}_seed{seed}"
    output_dir = Path(output_root) / run_id
    checkpoint_dir = Path(checkpoint_root) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    source_hashes = {}
    for name in ("train", "validation", "test"):
        for cell in groups[name]:
            path = data_root / dataset / (cell + ".csv")
            source_hashes[cell] = hashlib.sha256(path.read_bytes()).hexdigest()
    settings = {
        "dataset": dataset,
        "split": groups,
        "model": model_type,
        "model_spec": cfg["model"],
        "training": {**cfg["training"], "epochs": epochs},
        "forecasting": cfg["forecasting"],
        "features": list(FEATURES),
        "source_sha256": source_hashes,
    }

    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but no GPU is available")
    device = torch.device(
        "cuda" if device_name == "auto" and torch.cuda.is_available() else
        device_name if device_name != "auto" else "cpu"
    )
    model = build_model(model_type, len(FEATURES), lookback,
                        hidden=int(cfg["model"]["hidden"]),
                        dropout=float(cfg["model"]["dropout"])).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg["training"]["learning_rate"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )
    criterion = nn.MSELoss()
    recovery = EpochCheckpoint(checkpoint_dir, settings)
    if recovery.path.exists() and not resume:
        raise FileExistsError(
            f"Checkpoint already exists: {recovery.path}. Use --resume."
        )

    batch_size = int(cfg["training"]["batch_size"])
    training = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=batch_size,
        shuffle=True,
    )
    val_x = torch.from_numpy(x_val).to(device)
    val_y = torch.from_numpy(y_val).to(device)
    history = []
    best_loss, best_epoch, best_weights, start_epoch = float("inf"), 0, None, 0
    if resume:
        state = recovery.restore({"model": model, "optimizer": optimizer}, device=device)
        start_epoch = state["epoch"]
        history = state["history"]
        best_loss = state["best_validation_mse"]
        best_epoch = state["best_epoch"]
        best_weights = state["best_weights"]
        if start_epoch > epochs:
            raise ValueError("Checkpoint exceeds configured epoch budget")
        print(f"Resuming after epoch {start_epoch}", flush=True)

    for epoch in range(start_epoch + 1, epochs + 1):
        model.train()
        sum_loss = 0.0
        for x, y in training:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            if not bool(torch.isfinite(loss).item()):
                raise RuntimeError("Non-finite training loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            sum_loss += float(loss.detach().item()) * len(x)
        model.eval()
        with torch.no_grad():
            validation_loss = float(criterion(model(val_x), val_y).item())
        training_loss = sum_loss / len(x_train)
        history.append({
            "epoch": epoch, "train_mse": training_loss,
            "validation_mse": validation_loss
        })
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch
            best_weights = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        recovery.save(epoch, {"model": model, "optimizer": optimizer}, {
            "history": history, "best_validation_mse": best_loss,
            "best_epoch": best_epoch, "best_weights": best_weights,
        })
        print(f"Epoch {epoch:03d}/{epochs} | "
              f"train MSE={training_loss:.6f} | validation MSE={validation_loss:.6f}",
              flush=True)

    if best_weights is None:
        raise RuntimeError("No model was trained")
    model.load_state_dict(best_weights)
    predicted = evaluate(model, x_test, device)
    measured = y_test.astype(np.float64)
    test_scores = metrics_for(measured, predicted.astype(np.float64))

    predictions = test_meta.copy()
    predictions["actual_soh"] = measured
    predictions["predicted_soh"] = predicted
    predictions.to_csv(output_dir / "test_predictions.csv", index=False)
    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)
    figures(output_dir, history, predictions)
    torch.save({
        "model": best_weights,
        "settings": settings,
        "scaler_mean": scaler.mean_,
        "scaler_scale": scaler.scale_,
        "best_epoch": best_epoch,
    }, output_dir / "best_model.pt")

    report = {
        "task": "one-cycle-ahead SOH from past-cycle discharge summaries",
        "dataset": dataset,
        "model": model_type,
        "train_cells": groups["train"],
        "validation_cells": groups["validation"],
        "test_cells": groups["test"],
        "train_windows": int(len(y_train)),
        "validation_windows": int(len(y_val)),
        "test_windows": int(len(y_test)),
        "epochs_completed": int(len(history)),
        "best_epoch": best_epoch,
        "best_validation_mse": best_loss,
        "test": test_scores,
        "limitations": (
            "Evaluation is battery-disjoint and normalization is train-only. "
            "The SOH target uses each cell's first observed capacity. "
            "Only one test cell per dataset is used; this is not an RUL benchmark."
        ),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (output_dir / "provenance.json").write_text(
        json.dumps(settings, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)
    print("Outputs:", output_dir, flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", choices=MODEL_TYPES)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/simple_soh"))
    parser.add_argument("--checkpoint-root", type=Path, default=Path("outputs/simple_soh_checkpoints"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run(args.config, model_override=args.model, epochs_override=args.epochs,
        output_root=args.output_root, checkpoint_root=args.checkpoint_root,
        resume=args.resume, device_name=args.device)


if __name__ == "__main__":
    main()
