"""Cycle-level time-sampled inputs from original NASA MAT and CALCE XLSX archives.

NASA: original discharge Voltage_measured, Current_measured,
Temperature_measured, Time traces.
CALCE: original discharge Channel_ worksheet V/I/Test_Time; temperature
is not recorded in the inspected CS2 channels, so the fourth channel is
zero-filled and explicitly flagged. Raw files are not committed to Git.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import posixpath
import re
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from simple_soh.nasa_mat import inspect_file

NOMINAL_AH = {"nasa": 2.0, "calce": 1.1}
GROUPS = {
    "nasa_24c": {"dataset": "nasa", "train": ["B0005", "B0006", "B0007"], "test": ["B0018"]},
    "nasa_43c": {"dataset": "nasa", "train": ["B0029", "B0031", "B0032"], "test": ["B0030"]},
    "nasa_4c": {"dataset": "nasa", "train": ["B0053", "B0054", "B0055"], "test": ["B0056"]},
    "calce_cs2": {"dataset": "calce", "train": ["CS2_35", "CS2_36", "CS2_37"], "test": ["CS2_38"]},
}


def sha256_file(file: Path) -> str:
    h = hashlib.sha256()
    with Path(file).open("rb") as f:
        for b in iter(lambda: f.read(2 ** 20), b""):
            h.update(b)
    return h.hexdigest()


def resample(time, voltage, current, temperature, points: int) -> np.ndarray:
    if points < 16:
        raise ValueError("A cycle requires at least 16 resampling positions")
    arr = np.stack([time, voltage, current, temperature], axis=-1).astype(np.float64)
    arr = arr[np.isfinite(arr).all(axis=1)]
    if len(arr) < 3:
        raise ValueError("Discharge trace has fewer than three finite measurements")
    arr = arr[np.argsort(arr[:, 0], kind="stable")]
    t, unique_indices = np.unique(arr[:, 0], return_index=True)
    arr = arr[unique_indices]
    if len(t) < 3 or float(t[-1] - t[0]) <= 0:
        raise ValueError("Discharge trace has insufficient time support")
    query = np.linspace(float(t[0]), float(t[-1]), points)
    result = np.stack([
        np.interp(query, t, arr[:, 1]),  # V
        np.interp(query, t, arr[:, 2]),  # I
        np.interp(query, t, arr[:, 3]),  # T
        query - query[0],                # within-discharge time, seconds
    ], axis=-1).astype(np.float32)
    return result


def nasa_cell(mat_root: Path, battery: str, points: int) -> dict:
    path = Path(mat_root) / f"{battery}.mat"
    audit = inspect_file(path, battery)
    records = loadmat(path, struct_as_record=False, squeeze_me=True)
    events = np.atleast_1d(records[battery].cycle)
    signals, targets, source_indices = [], [], []
    raw_discharge_index = 0
    for event in events.flat:
        if str(getattr(event, "type", "")).lower() != "discharge":
            continue
        raw_discharge_index += 1
        data = event.data
        capacity = float(np.asarray(data.Capacity, dtype=float).reshape(-1)[0])
        if not (math.isfinite(capacity) and capacity > 0):
            continue
        tr = resample(
            np.asarray(data.Time).reshape(-1),
            np.asarray(data.Voltage_measured).reshape(-1),
            np.asarray(data.Current_measured).reshape(-1),
            np.asarray(data.Temperature_measured).reshape(-1),
            points,
        )
        signals.append(tr)
        targets.append(capacity / NOMINAL_AH["nasa"])
        source_indices.append(raw_discharge_index)
    if len(signals) != audit["usable_discharge_cycles"]:
        raise ValueError(f"{battery}: raw-trace extraction disagrees with verified MATLAB audit")
    return {
        "x": np.stack(signals),
        "y": np.asarray(targets, dtype=np.float32),
        "source_indices": np.asarray(source_indices, dtype=np.int32),
        "file_sha256": audit["mat_sha256"],
        "source_qa": audit,
        "missing_temperature": False,
    }


def _calce_workbook_cycles(workbook_bytes: bytes, workbook_name: str, points: int):
    """Stream original Arbin Channel_ worksheet XML, not capacity summary CSV."""
    from lxml import etree
    ns = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    relationships_ns = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    pkg_rel_ns = "http://schemas.openxmlformats.org/package/2006/relationships"
    tag = lambda name: f"{{{ns}}}{name}"
    with zipfile.ZipFile(io.BytesIO(workbook_bytes)) as wb:
        shared = []
        if "xl/sharedStrings.xml" in wb.namelist():
            tree = etree.fromstring(wb.read("xl/sharedStrings.xml"))
            shared = [
                "".join((v.text or "") for v in item.iter(tag("t")))
                for item in tree.findall(tag("si"))
            ]
        book = etree.fromstring(wb.read("xl/workbook.xml"))
        rels = etree.fromstring(wb.read("xl/_rels/workbook.xml.rels"))
        links = {v.get("Id"): v.get("Target")
                 for v in rels.findall(f"{{{pkg_rel_ns}}}Relationship")}
        for sh in book.iter(tag("sheet")):
            if not sh.get("name", "").startswith("Channel_"):
                continue
            target = links.get(sh.get(f"{{{relationships_ns}}}id"))
            if not target:
                continue
            location = (target.lstrip("/") if target.startswith("/")
                        else posixpath.normpath("xl/" + target))
            column_index, cycles = None, defaultdict(list)
            with wb.open(location) as xml:
                for _, row in etree.iterparse(xml, events=("end",), tag=tag("row")):
                    values = {}
                    for c in row.iterchildren(tag("c")):
                        match = re.match(r"[A-Z]+", c.get("r", ""))
                        if not match:
                            continue
                        index = 0
                        for ch in match.group():
                            index = 26 * index + ord(ch) - 64
                        index -= 1
                        if column_index is not None and index not in column_index.values():
                            continue
                        value = c.find(tag("v"))
                        if value is None or value.text is None:
                            inline = c.find(tag("is"))
                            if inline is None:
                                continue
                            v = "".join((z.text or "") for z in inline.iter(tag("t")))
                        else:
                            v = value.text
                        if c.get("t") == "s":
                            v = shared[int(v)]
                        else:
                            try:
                                v = float(v)
                            except (TypeError, ValueError):
                                pass
                        values[index] = v
                    if column_index is None:
                        column_index = {str(v).strip(): k for k, v in values.items()}
                        mandatory = (
                            "Cycle_Index", "Test_Time(s)", "Voltage(V)",
                            "Current(A)", "Discharge_Capacity(Ah)",
                        )
                        if not set(mandatory).issubset(column_index):
                            raise ValueError(f"{workbook_name}: missing required Arbin columns")
                    else:
                        def get(column):
                            return values.get(column_index.get(column, -1), math.nan)
                        try:
                            cycle_idx = int(float(get("Cycle_Index")))
                            time = float(get("Test_Time(s)"))
                            voltage = float(get("Voltage(V)"))
                            current = float(get("Current(A)"))
                            capacity = float(get("Discharge_Capacity(Ah)"))
                        except (TypeError, ValueError, OverflowError):
                            continue
                        if all(map(math.isfinite, (time, voltage, current, capacity))):
                            cycles[cycle_idx].append((time, voltage, current, capacity))
                    row.clear()
                    while row.getprevious() is not None:
                        del row.getparent()[0]

            for local_cycle in sorted(cycles):
                data = np.asarray(cycles[local_cycle], dtype=np.float64)
                if len(data) < 4:
                    continue
                cap = float(data[:, 3].max() - data[:, 3].min())
                discharge = data[data[:, 2] < -0.01]
                if cap <= 0.01 or len(discharge) < 3:
                    continue
                # Most source CS2 workbooks have no recorded temperature column.
                seq = resample(
                    discharge[:, 0], discharge[:, 1], discharge[:, 2],
                    np.zeros(len(discharge)), points,
                )
                yield local_cycle, cap, seq


def calce_cell(zip_root: Path, battery: str, points: int, cache: Path) -> dict:
    path = Path(zip_root) / f"{battery}.zip"
    if not path.is_file():
        raise FileNotFoundError(f"Original CALCE workbook archive missing: {path}")
    checksum = sha256_file(path)
    cache.mkdir(parents=True, exist_ok=True)
    cached = cache / f"{battery}_traces_{points}.npz"
    if cached.is_file():
        with np.load(cached, allow_pickle=False) as saved:
            if str(saved["source_sha256"]) == checksum:
                return {
                    "x": saved["x"].copy(), "y": saved["y"].copy(),
                    "source_indices": saved["source_indices"].copy(),
                    "file_sha256": checksum,
                    "source_qa": json.loads(str(saved["source_qa"])),
                    "missing_temperature": True,
                }
    source, capacities, inputs, source_indices = [], [], [], []
    with zipfile.ZipFile(path) as archive:
        names = sorted(n for n in archive.namelist() if n.lower().endswith(".xlsx"))
        if not names:
            raise ValueError(f"No original CALCE workbooks in {path}")
        for workbook in names:
            for local_idx, capacity, trace in _calce_workbook_cycles(
                archive.read(workbook), workbook, points
            ):
                source_indices.append(len(source_indices) + 1)
                source.append({"workbook": workbook, "source_cycle": local_idx})
                capacities.append(capacity)
                inputs.append(trace)
    if not inputs:
        raise ValueError(f"{battery}: no usable discharge traces")
    qa = {
        "source_format": "original_calce_xlsx_archive",
        "workbooks": len(names), "included_cycles": len(inputs),
        "source_workbook_cycle_keys": source,
        "missing_temperature_channel": True,
    }
    x = np.stack(inputs)
    y = np.asarray(capacities, dtype=np.float32) / NOMINAL_AH["calce"]
    idx = np.asarray(source_indices, dtype=np.int32)
    temporary = cache / (cached.name + ".partial.npz")
    try:
        np.savez_compressed(temporary, x=x, y=y, source_indices=idx,
                            source_sha256=np.asarray(checksum),
                            source_qa=np.asarray(json.dumps(qa)))
        temporary.replace(cached)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "x": x, "y": y, "source_indices": idx, "file_sha256": checksum,
        "source_qa": qa, "missing_temperature": True,
    }


def make_paper_split(group: str, source_root: Path, cache: Path,
                     points: int, validation_fraction: float = .15) -> dict:
    if group not in GROUPS:
        raise ValueError(f"Unknown experimental group {group}")
    if not (0.05 <= validation_fraction <= 0.30):
        raise ValueError("Validation fraction must be between 5% and 30%")
    assignment = GROUPS[group]
    loader = nasa_cell if assignment["dataset"] == "nasa" else None
    cells = {}
    for cell in assignment["train"] + assignment["test"]:
        cells[cell] = (
            loader(source_root, cell, points) if loader
            else calce_cell(source_root, cell, points, cache)
        )
    train, validation, test = [], [], []
    for cell in assignment["train"]:
        n = len(cells[cell]["y"])
        nv = max(1, int(round(n * validation_fraction)))
        if n - nv < 2:
            raise ValueError(f"{cell}: insufficient cycles for development split")
        train.extend((cell, i) for i in range(n - nv))
        validation.extend((cell, i) for i in range(n - nv, n))
    for cell in assignment["test"]:
        test.extend((cell, i) for i in range(len(cells[cell]["y"])))

    # Fixed feature scaler: estimate minima/maxima from TRAIN cycles only.
    raw_train = np.concatenate([cells[c]["x"][i] for c, i in train], axis=0)
    minval = raw_train.min(axis=0)
    maxval = raw_train.max(axis=0)
    denominator = np.where(maxval > minval, maxval - minval, 1.0)
    def stack(keys):
        features = np.stack(
            [(cells[cell]["x"][i] - minval) / denominator for cell, i in keys]
        ).astype(np.float32)
        targets = np.asarray([cells[cell]["y"][i] for cell, i in keys], dtype=np.float32)
        identifiers = [{"cell_id": cell,
                        "cycle": int(i + 1),
                        "source_discharge_index": int(cells[cell]["source_indices"][i])}
                       for cell, i in keys]
        return features, targets, identifiers

    return {
        "group": group, "dataset": assignment["dataset"],
        "train": stack(train), "validation": stack(validation),
        "test": stack(test),
        "scaling_min": minval.tolist(),
        "scaling_max": maxval.tolist(),
        "nominal_capacity_ah": NOMINAL_AH[assignment["dataset"]],
        "source_hashes": {k: c["file_sha256"] for k, c in cells.items()},
        "source_qa": {k: c["source_qa"] for k, c in cells.items()},
        "experiment_cells": assignment,
        "points": points,
        "validation_fraction": validation_fraction,
    }
