"""Source verification and MATLAB extraction tests for NASA discharge cells."""
from __future__ import annotations

import hashlib
import io
import zipfile

import numpy as np
import pytest
from scipy.io import savemat

from simple_soh import nasa_mat
from simple_soh.restore_nasa_mat import restore


def make_mat(cell: str = "B0005") -> bytes:
    cycles = np.empty((1, 3), dtype=[("type", "O"), ("data", "O")])
    for i, capacity in enumerate((1.8, 1.7, 0.0)):
        discharge = np.empty((1, 1), dtype=[
            ("Capacity", "O"), ("Voltage_measured", "O"),
            ("Current_measured", "O"), ("Temperature_measured", "O"),
            ("Time", "O"),
        ])
        signals = {
            "Capacity": np.array([[capacity]]),
            "Voltage_measured": np.array([[3.8, 3.7, 3.6]]),
            "Current_measured": np.array([[-2.0, -2.0, -2.0]]),
            "Temperature_measured": np.array([[24.0, 25.0, 26.0]]),
            "Time": np.array([[0.0, 100.0, 200.0]]),
        }
        for name, signal in signals.items():
            discharge[0, 0][name] = signal
        cycles[0, i]["type"] = np.array(["discharge"])
        cycles[0, i]["data"] = discharge
    battery = np.empty((1, 1), dtype=[("cycle", "O")])
    battery[0, 0]["cycle"] = cycles
    buf = io.BytesIO()
    savemat(buf, {cell: battery})
    return buf.getvalue()


def patch_test_source(monkeypatch, blob: bytes) -> None:
    monkeypatch.setitem(nasa_mat.MAT_SHA256, "B0005", hashlib.sha256(blob).hexdigest())
    monkeypatch.setitem(nasa_mat.RAW_DISCHARGES, "B0005", 3)
    monkeypatch.setitem(nasa_mat.VALID_DISCHARGES, "B0005", 2)


def test_original_mat_discharge_audit_and_features(tmp_path, monkeypatch):
    blob = make_mat()
    patch_test_source(monkeypatch, blob)
    mat_dir = tmp_path / "nasa"
    mat_dir.mkdir()
    (mat_dir / "B0005.mat").write_bytes(blob)

    audit = nasa_mat.inspect_bytes(blob, "B0005")
    assert audit["raw_discharge_events"] == 3
    assert audit["usable_discharge_cycles"] == 2
    assert audit["source_indices"] == [1, 2]
    assert audit["excluded_discharge_events"] == [
        {"source_discharge_index": 3, "capacity_ah": 0.0,
         "reason": "non-positive or invalid capacity"}
    ]

    frame = nasa_mat.load_nasa_mat_cell(mat_dir, "B0005")
    assert len(frame) == 2
    np.testing.assert_allclose(frame["soh_first_observed"], [1.0, 1.7 / 1.8])
    np.testing.assert_allclose(frame["voltage_mean"], [3.7, 3.7])
    np.testing.assert_allclose(frame["discharge_duration_s"], [200.0, 200.0])
    assert frame["source_discharge_index"].tolist() == [1, 2]
    assert frame.attrs["source_format"] == "original_mat"


def test_corrupted_mat_file_is_rejected(monkeypatch):
    blob = make_mat()
    patch_test_source(monkeypatch, blob)
    with pytest.raises(ValueError, match="SHA-256"):
        nasa_mat.inspect_bytes(blob + b"unauthorized appended bytes", "B0005")


def test_restore_nested_nasa_zip(tmp_path, monkeypatch):
    blob = make_mat()
    patch_test_source(monkeypatch, blob)
    nested = io.BytesIO()
    with zipfile.ZipFile(nested, mode="w") as archive:
        archive.writestr("B0005.mat", blob)
    outer = tmp_path / "5.+Battery+Data+Set.zip"
    with zipfile.ZipFile(outer, mode="w") as archive:
        archive.writestr("5. Battery Data Set/demo.zip", nested.getvalue())
    monkeypatch.setattr(
        "simple_soh.restore_nasa_mat.NASA_ARCHIVE_SHA256",
        hashlib.sha256(outer.read_bytes()).hexdigest(),
    )
    result = restore(outer, tmp_path / "raw", ["B0005"], allow_download=False)
    assert result["batteries"]["B0005"]["usable_discharge_cycles"] == 2
    assert (tmp_path / "raw" / "B0005.mat").read_bytes() == blob
    second = restore(outer, tmp_path / "raw", ["B0005"], allow_download=False)
    assert second["batteries"]["B0005"]["raw_discharge_events"] == 3


def test_all_twelve_original_sha256_definitions():
    assert len(nasa_mat.MAT_SHA256) == 12
    assert set(nasa_mat.MAT_SHA256) == set(nasa_mat.RAW_DISCHARGES)
    assert set(nasa_mat.MAT_SHA256) == set(nasa_mat.VALID_DISCHARGES)
    assert sum(nasa_mat.VALID_DISCHARGES.values()) == 1157
    assert sum(nasa_mat.RAW_DISCHARGES.values()) == 1159
    assert nasa_mat.RAW_DISCHARGES["B0053"] == 56
    assert nasa_mat.VALID_DISCHARGES["B0053"] == 55
    assert nasa_mat.RAW_DISCHARGES["B0054"] == 103
    assert nasa_mat.VALID_DISCHARGES["B0054"] == 102
