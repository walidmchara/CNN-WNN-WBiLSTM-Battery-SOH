"""Publish a completed simple-SOH run without Git history operations.

The source data and PyTorch checkpoints are retained in persistent storage.
Only numeric reports and prediction figures are accepted for upload.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen

PUBLISH_FILES = (
    "metrics.json",
    "provenance.json",
    "training_history.csv",
    "test_predictions.csv",
    "soh_trajectory.png",
    "learning_curve.png",
    "parity.png",
    "residuals.png",
)


def request(url: str, token: str, method: str = "GET",
            body: bytes | None = None) -> dict:
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if body is not None:
        headers["Content-Type"] = "application/json"
    req = Request(url, data=body, headers=headers, method=method)
    try:
        with urlopen(req, timeout=60) as response:
            return json.load(response)
    except HTTPError as exc:
        raise RuntimeError(
            f"GitHub API returned HTTP {exc.code} for {method} "
            f"{url.split('api.github.com')[-1]}"
        ) from None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--repository", default="walidmchara/CNN-WNN-WBiLSTM-Battery-SOH"
    )
    parser.add_argument("--branch", default="main")
    parser.add_argument("--min-epochs", type=int, default=30)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        raise RuntimeError("GITHUB_TOKEN environment variable is required")
    if len(args.repository.split("/")) != 2:
        raise ValueError("repository must be OWNER/NAME")
    root = args.run_dir.expanduser().resolve()
    report = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
    if report["epochs_completed"] < args.min_epochs:
        raise ValueError("Refusing to publish an incomplete training run")
    if not (root / "provenance.json").is_file():
        raise FileNotFoundError(root / "provenance.json")
    missing = [name for name in PUBLISH_FILES if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing run files: {missing}")
    provenance = json.loads((root / "provenance.json").read_text())
    run_slug = provenance.get("run_slug", report["dataset"])
    if root.name != (f"{run_slug}_{report['model']}_"
                     f"lb{provenance['forecasting']['lookback']}_"
                     f"e{report['epochs_completed']}_"
                     f"seed{provenance['training']['seed']}"):
        raise ValueError("Run folder does not match recorded configuration")

    api = f"https://api.github.com/repos/{args.repository}"
    metadata = request(api, token)
    if not metadata.get("private", False):
        raise PermissionError("Publication refused: target repository is not private")
    prefix = f"results/simple_soh/{root.name}"
    manifest = {
        "run_id": root.name,
        "files": [
            {
                "name": name,
                "size_bytes": (root/name).stat().st_size,
                "sha256": hashlib.sha256((root/name).read_bytes()).hexdigest(),
            }
            for name in PUBLISH_FILES
        ],
        "source_weights": "Retained outside the GitHub repository",
    }

    for name in PUBLISH_FILES + ("manifest.json",):
        local = (
            json.dumps(manifest, indent=2).encode("utf-8") + b"\n"
            if name == "manifest.json" else (root/name).read_bytes()
        )
        url = api + "/contents/" + quote(prefix + "/" + name, safe="/")
        try:
            old = request(url + f"?ref={quote(args.branch)}", token)
        except RuntimeError as exc:
            if "HTTP 404" not in str(exc):
                raise
            old = None
        existing = old.get("content") if old else None
        if existing:
            uploaded = base64.b64decode(existing)
            if uploaded != local:
                raise FileExistsError(
                    f"Remote file differs: {prefix}/{name}; "
                    "choose a new experiment identifier before republishing"
                )
            print(f"Already present: {name}", flush=True)
            continue
        if args.dry_run:
            print(f"Would publish: {prefix}/{name}", flush=True)
            continue
        data = json.dumps({
            "message": f"Publish measured {root.name} SOH report: {name}",
            "content": base64.b64encode(local).decode("ascii"),
            "branch": args.branch,
        }).encode("utf-8")
        request(url, token, method="PUT", body=data)
        print(f"Published: {prefix}/{name}", flush=True)

    print(f"GitHub results: https://github.com/{args.repository}/tree/{args.branch}/{prefix}")


if __name__ == "__main__":
    main()
