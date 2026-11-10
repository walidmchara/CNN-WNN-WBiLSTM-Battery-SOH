"""Original NASA PCoE MATLAB discharge data and provenance verification."""
from __future__ import annotations

import hashlib
import io
import math
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from scripts.prepare_battery_data import load_nasa_mat

NASA_ARCHIVE_URL = "https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip"
NASA_ARCHIVE_SHA256 = "82302a7db4fc1b34e0b6676326610438d43b816bdf11a69d1d012a464ef2f92e"

# Counts are raw discharge events, including a zero-capacity terminal event
# in B0053 and B0054; these are recorded but excluded from SOH targets.
RAW_DISCHARGES = {
    "B0005": 168, "B0006": 168, "B0007": 168, "B0018": 132,
    "B0029": 40, "B0030": 40, "B0031": 40, "B0032": 40,
    "B0053": 56, "B0054": 103, "B0055": 102, "B0056": 102,
}
VALID_DISCHARGES = {
    **{cell: count for cell, count in RAW_DISCHARGES.items()
       if cell not in ("B0053", "B0054")},
    "B0053": 55, "B0054": 102,
}
MAT_SHA256 = {
    "B0005": "0eae4585baf3f200c09fe24c5ab884f1889679fc75206ca1aa19da704104f0b0",
    "B0006": "fa818ab4db5db8ab21e910b6dd6c3e20d3761bb9672089e1a4de8f96074616c5",
    "B0007": "d022afa086efaf54ab5b63f05220f5be178c8027e2fa8d68589db0e85a441a3b",
    "B0018": "d1e6c923a43ea1c9666b3a90bbb521757a067fd60b4d17dbfaa49c50b179da69",
    "B0029": "da9c63ac7b8a10d2354ff53fec9be23f4194352e667de5c503b069f7d267ebb2",
    "B0030": "c29a910256812449aa6090031471a57440e2e7bc827bd2c822b0d3135485cc99",
    "B0031": "c47cb3a5ef47ef4de26fcc80c2173fec24a139f6c305852fccee603cd6c4bcec",
    "B0032": "bd146cad3a18f9e8d42d05014c90e7bcde7abc7ccc89bdd46e5655604c61bc0f",
    "B0053": "5ae945704d75f9865608c1bf585c80b62ba8e6a98c555840735c1f64fa5513f9",
    "B0054": "02ececa6edf2ab1febdfa31daccaaff59b090e627b9403a59060157c9488712a",
    "B0055": "8405c13378ced0894019a74f343057b0151f58c417fed45a5e068c4e372ea039",
    "B0056": "e894d92938c6fb09b6455bcc49330902c0e3ea8ee65129f664f1718a4843e709",
}


def _mat_events(payload: bytes, cell: str):
    record = loadmat(io.BytesIO(payload), squeeze_me=True, struct_as_record=False)
    if cell not in record:
        raise ValueError(f"{cell}: MATLAB object is missing")
    return [
        event for event in np.atleast_1d(record[cell].cycle).flat
        if str(getattr(event, "type", "")).lower() == "discharge"
    ]


def inspect_bytes(payload: bytes, cell: str) -> dict:
    if cell not in RAW_DISCHARGES:
        raise ValueError(f"Unrecognized NASA battery {cell!r}")
    sha = hashlib.sha256(payload).hexdigest()
    if sha != MAT_SHA256[cell]:
        raise ValueError(f"{cell}: original MATLAB SHA-256 mismatch")
    events = _mat_events(payload, cell)
    valid_indices = []
    excluded = []
    for source_index, event in enumerate(events, start=1):
        try:
            capacity = float(np.asarray(event.data.Capacity).reshape(-1)[0])
        except (AttributeError, ValueError, IndexError, TypeError):
            capacity = math.nan
        if not (math.isfinite(capacity) and capacity > 0):
            excluded.append({"source_discharge_index": source_index,
                             "capacity_ah": capacity if math.isfinite(capacity) else None,
                             "reason": "non-positive or invalid capacity"})
        else:
            valid_indices.append(source_index)
    if len(events) != RAW_DISCHARGES[cell] or len(valid_indices) != VALID_DISCHARGES[cell]:
        raise ValueError(
            f"{cell}: unexpected discharge count "
            f"(raw={len(events)}, valid={len(valid_indices)})"
        )
    return {
        "cell_id": cell,
        "mat_sha256": sha,
        "raw_discharge_events": len(events),
        "usable_discharge_cycles": len(valid_indices),
        "excluded_discharge_events": excluded,
        "source_indices": valid_indices,
    }


def inspect_file(path: Path, cell: str) -> dict:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    return inspect_bytes(path.read_bytes(), cell)


def load_nasa_mat_cell(mat_dir: str | Path, cell: str):
    """Load original MATLAB discharge measurements, not an intermediate CSV."""
    path = Path(mat_dir) / f"{cell}.mat"
    payload = path.read_bytes()
    provenance = inspect_bytes(payload, cell)
    frame = load_nasa_mat(payload, cell)
    if len(frame) != provenance["usable_discharge_cycles"]:
        raise ValueError(f"{cell}: MATLAB extraction disagrees with raw-event audit")
    frame["source_discharge_index"] = provenance["source_indices"]
    frame.attrs["source_format"] = "original_mat"
    frame.attrs["mat_sha256"] = provenance["mat_sha256"]
    frame.attrs["raw_discharge_events"] = provenance["raw_discharge_events"]
    frame.attrs["excluded_discharge_events"] = provenance["excluded_discharge_events"]
    return frame
