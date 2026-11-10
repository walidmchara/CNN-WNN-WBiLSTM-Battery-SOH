"""Restore original NASA PCoE MATLAB source files with SHA-256 validation.

Source: NASA PCoE nested ZIP archive. Extracted MATLAB files and the
source ZIP belong on persistent storage, not in Git history.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import urllib.request
import zipfile
from pathlib import Path

from .nasa_mat import (
    MAT_SHA256, NASA_ARCHIVE_SHA256, NASA_ARCHIVE_URL, RAW_DISCHARGES,
    inspect_bytes, inspect_file,
)


def sha256_file(path: Path) -> str:
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def download_verified_archive(archive: Path) -> None:
    archive.parent.mkdir(parents=True, exist_ok=True)
    temporary = archive.with_name(archive.name + ".download")
    try:
        req = urllib.request.Request(
            NASA_ARCHIVE_URL, headers={"User-Agent": "NASA-SOH-Research/1.0"}
        )
        print(f"Downloading original NASA archive: {NASA_ARCHIVE_URL}", flush=True)
        with urllib.request.urlopen(req, timeout=120) as remote, temporary.open("wb") as local:
            shutil.copyfileobj(remote, local, length=1024 * 1024)
        if sha256_file(temporary) != NASA_ARCHIVE_SHA256:
            raise ValueError("NASA source archive SHA-256 mismatch")
        os.replace(temporary, archive)
        print("NASA archive SHA-256 verified", flush=True)
    finally:
        temporary.unlink(missing_ok=True)


def restore(archive: Path, dest: Path, wanted: list[str], *,
            allow_download: bool = True) -> dict:
    wanted = sorted(set(wanted))
    if not wanted or any(cell not in RAW_DISCHARGES for cell in wanted):
        raise ValueError("Expected NASA IDs from B0005..B0056 collection")
    dest.mkdir(parents=True, exist_ok=True)
    audits = {}
    for cell in wanted:
        target = dest / f"{cell}.mat"
        if target.exists():
            audits[cell] = inspect_file(target, cell)
            print(f"Existing verified {cell}.mat: "
                  f"{audits[cell]['usable_discharge_cycles']} usable cycles", flush=True)

    missing = set(wanted) - set(audits)
    if missing:
        if not archive.exists():
            if not allow_download:
                raise FileNotFoundError(archive)
            try:
                download_verified_archive(archive)
            except Exception as exc:
                raise RuntimeError(
                    "NASA original archive could not be downloaded or verified. "
                    "Copy the previously uploaded 5.+Battery+Data+Set.zip into "
                    "the requested --archive path on Google Drive."
                ) from exc
        if sha256_file(archive) != NASA_ARCHIVE_SHA256:
            raise ValueError("Unexpected NASA ZIP SHA-256. Refusing extraction.")
        print(f"Extracting {len(missing)} missing .mat files from NASA nested ZIPs", flush=True)
        with zipfile.ZipFile(archive) as outer:
            for outer_item in outer.infolist():
                if outer_item.is_dir() or not outer_item.filename.lower().endswith(".zip"):
                    continue
                with zipfile.ZipFile(io.BytesIO(outer.read(outer_item))) as nested:
                    for inner_item in nested.infolist():
                        cell = Path(inner_item.filename).stem.upper()
                        if inner_item.is_dir() or cell not in missing:
                            continue
                        if Path(inner_item.filename).name.upper() != cell + ".MAT":
                            continue
                        payload = nested.read(inner_item)
                        audit = inspect_bytes(payload, cell)
                        target = dest / f"{cell}.mat"
                        temporary = dest / f".{cell}.mat.partial"
                        try:
                            temporary.write_bytes(payload)
                            os.replace(temporary, target)
                        finally:
                            temporary.unlink(missing_ok=True)
                        audits[cell] = audit
                        missing.remove(cell)
                        print(f"Verified {cell}.mat: "
                              f"{audit['raw_discharge_events']} raw / "
                              f"{audit['usable_discharge_cycles']} usable cycles",
                              flush=True)
                if not missing:
                    break
    if missing:
        raise FileNotFoundError(f"Original NASA MATLAB files unavailable: {sorted(missing)}")
    return {
        "source_url": NASA_ARCHIVE_URL,
        "source_archive_sha256": NASA_ARCHIVE_SHA256,
        "mat_directory": str(dest),
        "batteries": {cell: {k: v for k, v in audits[cell].items()
                             if k != "source_indices"} for cell in wanted},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument("--cells", nargs="+", choices=sorted(RAW_DISCHARGES),
                        default=sorted(RAW_DISCHARGES))
    parser.add_argument("--manifest-out", type=Path)
    parser.add_argument("--offline", action="store_true",
                        help="Never download; require an existing verified ZIP or files")
    args = parser.parse_args()
    info = restore(args.archive, args.dest, args.cells,
                   allow_download=not args.offline)
    if args.manifest_out:
        args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
        args.manifest_out.write_text(
            json.dumps(info, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
    print("Verified original NASA .mat files:", len(info["batteries"]), flush=True)


if __name__ == "__main__":
    main()
