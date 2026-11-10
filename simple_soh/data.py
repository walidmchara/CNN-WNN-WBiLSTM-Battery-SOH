"""Validated cycle-level inputs and battery-disjoint forecasting windows."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from . import FEATURES


def _validate_cell(frame: pd.DataFrame, cell_id: str, source: Path) -> pd.DataFrame:
    required = list(FEATURES) + ["cycle", "capacity_ah", "soh_first_observed"]
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise ValueError(f"{source}: missing columns {missing}")
    if "cell_id" in frame.columns and not (frame["cell_id"].astype(str) == cell_id).all():
        raise ValueError(f"{source}: cell identifier does not match source filename")
    for column in required:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    frame = frame.sort_values("cycle", kind="stable").reset_index(drop=True)
    if frame["cycle"].isna().any() or frame["cycle"].duplicated().any():
        raise ValueError(f"{source}: invalid or repeated cycle numbers")
    if not frame["cycle"].is_monotonic_increasing:
        raise ValueError(f"{source}: cycles not ordered")
    values = frame[required].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"{source}: missing/non-finite required observations")
    if len(frame) < 15 or (frame["capacity_ah"] <= 0).any():
        raise ValueError(f"{source}: insufficient cycles or non-positive capacity")
    reference = float(frame.loc[0, "capacity_ah"])
    derived = frame["capacity_ah"].to_numpy() / reference
    target = frame["soh_first_observed"].to_numpy()
    if not np.allclose(derived, target, rtol=1e-8, atol=1e-10):
        raise ValueError(f"{source}: inconsistent SOH label definition")
    frame["cell_id"] = cell_id
    return frame


def load_cell(data_root: str | Path, dataset: str, cell_id: str) -> pd.DataFrame:
    if dataset not in ("nasa", "calce"):
        raise ValueError("Dataset must be nasa or calce")
    if "/" in cell_id or "\\" in cell_id or ".." in cell_id:
        raise ValueError("Invalid cell identifier")
    path = Path(data_root) / dataset / (cell_id + ".csv")
    if not path.is_file():
        raise FileNotFoundError(path)
    return _validate_cell(pd.read_csv(path), cell_id, path)


def load_split(data_root, dataset: str, groups: dict, *,
               source_format: str = "csv", mat_dir: str | Path | None = None) -> dict:
    keys = ("train", "validation", "test")
    if set(groups) != set(keys):
        raise ValueError(f"Required split groups: {keys}")
    ids = [cell for name in keys for cell in groups[name]]
    if len(ids) != len(set(ids)):
        raise ValueError("A battery cannot appear in multiple data splits")
    if not all(groups[name] for name in keys):
        raise ValueError("Every split requires at least one cell")
    if source_format == "original_mat":
        if dataset != "nasa" or mat_dir is None:
            raise ValueError("Original MATLAB loading requires nasa and --nasa-mat-dir")
        from .nasa_mat import load_nasa_mat_cell

        return {
            name: [
                _validate_cell(
                    load_nasa_mat_cell(mat_dir, cell), cell,
                    Path(mat_dir) / f"{cell}.mat",
                )
                for cell in groups[name]
            ]
            for name in keys
        }
    if source_format != "csv":
        raise ValueError(f"Unknown source_format: {source_format}")
    return {
        name: [load_cell(data_root, dataset, cell) for cell in groups[name]]
        for name in keys
    }


def fit_scaler(train_cells: list[pd.DataFrame]) -> StandardScaler:
    """Fit feature normalization exclusively on training batteries."""
    data = np.concatenate(
        [frame.loc[:, list(FEATURES)].to_numpy(dtype=np.float64) for frame in train_cells],
        axis=0,
    )
    scaler = StandardScaler().fit(data)
    if not np.isfinite(scaler.mean_).all() or not np.isfinite(scaler.scale_).all():
        raise ValueError("Non-finite scaler statistics")
    return scaler


def make_windows(
    cells: list[pd.DataFrame], scaler: StandardScaler, lookback: int
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Forecast SOH at cycle t from measurements at t-lookback,...,t-1."""
    if not isinstance(lookback, int) or lookback < 2:
        raise ValueError("lookback must be an integer >= 2")
    xs, ys, meta = [], [], []
    for frame in cells:
        if len(frame) <= lookback:
            raise ValueError(f"Insufficient history: {frame['cell_id'].iloc[0]}")
        transformed = scaler.transform(
            frame.loc[:, list(FEATURES)].to_numpy(dtype=np.float64)
        ).astype(np.float32)
        target = frame["soh_first_observed"].to_numpy(dtype=np.float32)
        for t in range(lookback, len(frame)):
            xs.append(transformed[t - lookback : t])
            ys.append(float(target[t]))
            meta.append({
                "cell_id": frame["cell_id"].iat[t],
                "cycle": int(frame["cycle"].iat[t]),
                "last_observed_cycle": int(frame["cycle"].iat[t - 1]),
            })
    return (
        np.stack(xs).astype(np.float32),
        np.asarray(ys, dtype=np.float32),
        pd.DataFrame(meta),
    )
