"""Validation-only R1 experiment. Test labels are never evaluated or logged.

Usage: python -m ablations.run_bilstm --group nasa_43c
       --source-root /content/drive/MyDrive/NASA_original_source/mat
       --output-root /content/drive/MyDrive/Automatika_2025_Reproduction/r1_validation
       --checkpoint-root /content/drive/MyDrive/Automatika_2025_Reproduction/r1_checkpoints
       --cache-root /content/drive/MyDrive/Automatika_2025_Reproduction/cache
       --seed 42 --epochs 58 --resume

Checkpoint metadata fingerprints architecture, seed, source hashes and split settings.
"""
import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from paper_reproduction.data import make_paper_split
from paper_reproduction.run import seed_all, metrics, predict_batches
from training_recovery import EpochCheckpoint
from .models import CNNWNNBiLSTMAM


def run(args):
    seed_all(args.seed)
    data = make_paper_split(args.group, args.source_root, args.cache_root,
                            args.points, validation_fraction=args.validation_fraction)
    x_train, y_train, _ = data["train"]
    x_val, y_val, _ = data["validation"]
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    model = CNNWNNBiLSTMAM().to(device)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001)
    run_id = f"r1_bilstm_{args.group}_p{args.points}_e{args.epochs}_seed{args.seed}"
    out = args.output_root / run_id
    out.mkdir(parents=True, exist_ok=True)
    meta = {
        "architecture": "R1_CNN_WNN_standard_BiLSTM_attention",
        "group": args.group, "seed": args.seed, "points": args.points,
        "epochs": args.epochs, "batch_size": 32,
        "validation_fraction": args.validation_fraction,
        "source_sha256": data["source_hashes"],
        "train_cells": data["experiment_cells"]["train"],
        "test_cells_not_evaluated": data["experiment_cells"]["test"],
        "optimizer": "RMSprop", "lr": 0.001,
        "objective": "development_validation_only",
    }
    recovery = EpochCheckpoint(args.checkpoint_root / run_id, meta)
    history = []
    best_val = math.inf
    best_epoch = 0
    best_state = None
    start = 0
    if args.resume and recovery.path.exists():
        state = recovery.restore({"model": model, "optimizer": optimizer}, device=device)
        history = state["history"]
        best_val = state["best_validation_mse"]
        best_epoch = state["best_epoch"]
        best_state = state["best_model"]
        start = state["epoch"]
        print(f"Resumed {run_id} after epoch {start}", flush=True)
    elif recovery.path.exists():
        raise FileExistsError("Existing checkpoint; use --resume")

    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=32, shuffle=True)
    val_x = torch.from_numpy(x_val).to(device)
    val_y = torch.from_numpy(y_val).to(device)
    criterion = nn.MSELoss()
    for epoch in range(start + 1, args.epochs + 1):
        model.train()
        weighted_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = criterion(pred, yb)
            if not torch.isfinite(loss).item():
                raise RuntimeError(f"Non-finite loss at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            weighted_loss += loss.detach().item() * len(yb)
        model.eval()
        with torch.no_grad():
            pred_val = predict_batches(model, x_val, device)
            val_mse = float(np.mean((pred_val - y_val) ** 2))
        train_mse = weighted_loss / len(y_train)
        history.append({"epoch": epoch, "train_mse": train_mse,
                        "validation_mse": val_mse})
        if val_mse < best_val:
            best_val = val_mse
            best_epoch = epoch
            best_state = copy.deepcopy({
                k: v.detach().cpu() for k, v in model.state_dict().items()})
        recovery.save(epoch, {"model": model, "optimizer": optimizer}, {
            "history": history, "best_validation_mse": best_val,
            "best_epoch": best_epoch, "best_model": best_state,
        })
        print(f"{run_id} epoch={epoch}/{args.epochs} "
              f"train_mse={train_mse:.7f} val_mse={val_mse:.7f}", flush=True)
    if best_state is None:
        raise RuntimeError("No best model recorded")
    model.load_state_dict(best_state)
    val_predictions = predict_batches(model, x_val, device)
    report = {
        "status": "R1_ablation_validation_only",
        "run_id": run_id,
        "best_epoch": best_epoch,
        "best_validation_mse": best_val,
        "train_count": len(y_train),
        "validation_count": len(y_val),
        "validation_metrics": metrics(y_val, val_predictions),
        "validation_prediction_std": float(np.std(val_predictions)),
        "validation_target_std": float(np.std(y_val)),
        "test_evaluated": False,
        "configuration": meta,
    }
    (out / "validation_metrics.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame(history).to_csv(out / "training_history.csv", index=False)
    torch.save({"model": best_state, "metadata": meta, "best_epoch": best_epoch},
               out / "best_model.pt")
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", choices=["nasa_43c", "nasa_4c", "nasa_24c"], required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=58)
    parser.add_argument("--points", type=int, default=64)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
