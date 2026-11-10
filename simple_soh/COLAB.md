# Running the NASA and CALCE SOH baselines in Google Colab

> **NASA original MATLAB source (12 batteries):** use [the new MATLAB/Colab launcher](COLAB_NASA_MAT12.md). The older sample below is retained only for the prior CSV experiment and must not be used for the current 12-battery MATLAB benchmark. CALCE remains on its verified CSV pipeline.

The following cell is intended for a **new Google Colab notebook**. The repository is private. Add a fine-grained GitHub token to Colab Secrets under the name `GithubToken` and allow notebook access to this secret. Select a GPU runtime if available; CPU is supported.

## 1. Train one dataset/model pair

Set `DATASET` to `nasa` or `calce`; set `MODEL` to `mlp`, `lstm`, `gru`, or `cnn_lstm`. By default, use NASA/GRU first and CALCE/LSTM second.

```python
import os
import sys
import json
import base64
import tempfile
import subprocess
from pathlib import Path
from google.colab import drive, userdata

DATASET = "nasa"              # Or "calce"
MODEL = "gru"                 # Or "lstm", "mlp", "cnn_lstm"
EPOCHS = 30
SEED = 42
LOOKBACK = 12

OWNER = "walidmchara"
NAME = "CNN-WNN-WBiLSTM-Battery-SOH"
drive.mount("/content/drive")
base = Path("/content/drive/MyDrive/Battery_SOH_Baselines")
outputs = base / "results"
checkpoints = base / "checkpoints"
outputs.mkdir(parents=True, exist_ok=True)
checkpoints.mkdir(parents=True, exist_ok=True)

token = userdata.get("GithubToken")
if not token:
    raise RuntimeError("Missing GithubToken secret in Colab")

encoded = base64.b64encode(
    f"x-access-token:{token}".encode()
).decode()
git_env = os.environ.copy()
git_env["GIT_CONFIG_COUNT"] = "1"
git_env["GIT_CONFIG_KEY_0"] = "http.https://github.com/.extraheader"
git_env["GIT_CONFIG_VALUE_0"] = f"AUTHORIZATION: basic {encoded}"
repo = Path(tempfile.mkdtemp(prefix="simple_soh_repo_", dir="/content")) / "repository"
cloned = subprocess.run(
    ["git", "-c", "credential.helper=", "clone", "--depth", "1",
     f"https://github.com/{OWNER}/{NAME}.git", str(repo)],
    env=git_env, capture_output=True, text=True
)
if cloned.returncode:
    raise RuntimeError("GitHub clone failed. Check repository access and token permissions.")
del token, encoded, git_env
print("Repository cloned from GitHub")

subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "-r", str(repo / "requirements.txt")],
    check=True,
)
import torch
print("GPU:", torch.cuda.is_available())

run_id = f"{DATASET}_{MODEL}_lb{LOOKBACK}_e{EPOCHS}_seed{SEED}"
checkpoint = checkpoints / run_id / "latest.pt"
command = [
    sys.executable, "-u", "-m", "simple_soh.run",
    "--config", f"configs/simple_soh_{DATASET}.json",
    "--model", MODEL,
    "--output-root", str(outputs),
    "--checkpoint-root", str(checkpoints),
]
if checkpoint.exists():
    command.append("--resume")
    print("Resuming a verified checkpoint")
else:
    print("Starting a new training run")

process = subprocess.Popen(
    command, cwd=repo,
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    text=True, bufsize=1
)
for line in process.stdout:
    print(line, end="", flush=True)
if process.wait() != 0:
    raise RuntimeError("Training failed; inspect the preceding log")

report = json.loads((outputs/run_id/"metrics.json").read_text())
assert report["epochs_completed"] == EPOCHS
print("Completed run:", run_id)
print(json.dumps(report["test"], indent=2))
print("Results:", outputs/run_id)
```

The training command works directly on the **CSV data already versioned in the repository**; no NASA MATLAB or CALCE Excel download is required for this baseline.

After a Colab disconnect, rerun the same cell. The script makes a fresh read-only clone, finds the persistent checkpoint on Drive and continues from the last completed epoch. A changed dataset, model or training protocol will be rejected by the recovery fingerprint.

## 2. Publish a completed report (optional)

Execute this second cell only after the full 30 epochs have succeeded. It sends metrics, predictions, provenance and plots to the **private** GitHub repository. It does not upload source data, neural-network weights or checkpoints. Publishing is an explicit action, not part of training.

```python
import os
import sys
import subprocess
from google.colab import userdata

secret = userdata.get("GithubToken")
assert secret, "Missing GithubToken"
publish_env = os.environ.copy()
publish_env["GITHUB_TOKEN"] = secret
subprocess.run(
    [sys.executable, "-m", "simple_soh.publish",
     "--run-dir", str(outputs / run_id)],
    cwd=repo, env=publish_env, check=True,
)
del secret, publish_env
print("Scientific run report published")
```

The publication API is idempotent: a retry skips files whose bytes already match the GitHub version and rejects conflicting overwrites. The token must have **Contents: Read and write** permission for the selected private repository.

## Data and scientific interpretation

NASA and CALCE are run as separate within-dataset benchmarks. The test battery is never used for feature normalization or model selection. The five input features come from earlier cycles only; the label is the next cycle's SOH relative to first observed capacity.

The resulting figures depict actually measured held-out-cell observations and model estimates. The existing CALCE column `remaining_observed_cycles` is not a validated RUL target and is not used. Full experiment definitions and methodological limitations are documented in [README.md](README.md).
