# NASA original MATLAB data — 12-cell Colab experiment

This experiment directly loads twelve SHA-256-verified NASA PCoE `.mat` files, extracted from the original source archive. NASA CSV summaries are **not** used for training. CALCE keeps its existing independent CSV workflow.

## Training and checkpoint recovery

Create a new Colab notebook, activate the private GitHub secret `GithubToken`, and paste this complete cell:

```python
import os, sys, json, tempfile, subprocess
from pathlib import Path
from google.colab import drive, userdata
from IPython.display import display, Image

drive.mount("/content/drive")
REPO_NAME = "CNN-WNN-WBiLSTM-Battery-SOH"
RUN_ID = "nasa_mat12_gru_lb12_e30_seed42"
BASE = Path("/content/drive/MyDrive/Battery_SOH_Baselines")
NASA_SOURCE = Path("/content/drive/MyDrive/NASA_original_source")
MAT_DIR = NASA_SOURCE / "mat"
ARCHIVE = NASA_SOURCE / "5.+Battery+Data+Set.zip"
OUTPUTS, CHECKPOINTS = BASE / "results", BASE / "checkpoints"
for directory in (NASA_SOURCE, MAT_DIR, OUTPUTS, CHECKPOINTS):
    directory.mkdir(parents=True, exist_ok=True)

token = userdata.get("GithubToken")
assert token, "GithubToken secret missing from Colab"
REPO = Path(tempfile.mkdtemp(prefix="nasa_mat12_", dir="/content")) / "repository"

with tempfile.TemporaryDirectory() as auth_dir:
    askpass = Path(auth_dir) / "askpass.py"
    askpass.write_text(
        "#!/usr/bin/env python3\n"
        "import os,sys\n"
        "print('x-access-token' if 'username' in sys.argv[1].lower() "
        "else os.environ['GITHUB_TOKEN'])\n"
    )
    askpass.chmod(0o700)
    auth = os.environ.copy()
    auth.update(GITHUB_TOKEN=token, GIT_ASKPASS=str(askpass),
                GIT_TERMINAL_PROMPT="0")
    check = subprocess.run(
        ["git", "-c", "credential.helper=", "clone", "--depth", "1",
         f"https://github.com/walidmchara/{REPO_NAME}.git", str(REPO)],
        env=auth, text=True, capture_output=True
    )
    if check.returncode:
        raise RuntimeError("Private clone failed: check GithubToken access")
del token, auth
print("Private GitHub repository cloned")

subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q",
     "-r", str(REPO / "requirements.txt")],
    check=True
)

# If absent, download the 210 MB official NASA ZIP once to Drive,
# then verify the exact ZIP and all twelve individual MATLAB checksums.
subprocess.run([
    sys.executable, "-u", "-m", "simple_soh.restore_nasa_mat",
    "--archive", str(ARCHIVE), "--dest", str(MAT_DIR),
    "--manifest-out", str(NASA_SOURCE / "mat12_source_audit.json"),
], cwd=REPO, check=True)

command = [
    sys.executable, "-u", "-m", "simple_soh.run",
    "--config", "configs/simple_soh_nasa_mat12.json",
    "--model", "gru", "--nasa-mat-dir", str(MAT_DIR),
    "--device", "auto",
    "--output-root", str(OUTPUTS),
    "--checkpoint-root", str(CHECKPOINTS),
]

if (CHECKPOINTS / RUN_ID / "latest.pt").is_file():
    command.append("--resume")
    print("Resuming the last verified NASA .mat checkpoint")
else:
    print("New NASA MAT12 training, 30 epochs")

env = os.environ.copy()
env["PYTHONUNBUFFERED"] = "1"
proc = subprocess.Popen(command, cwd=REPO, env=env,
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        text=True, bufsize=1)
for line in proc.stdout:
    print(line, end="", flush=True)
if proc.wait():
    raise RuntimeError("MAT12 training interrupted; review preceding logs")

result_dir = OUTPUTS / RUN_ID
report = json.loads((result_dir / "metrics.json").read_text())
assert report["epochs_completed"] == 30
assert report["source_format"] == "original_mat"
assert report["test_cells"] == ["B0018", "B0030", "B0056"]
assert report["test_windows"] == 238
print("Overall test metrics:", json.dumps(report["test"], indent=2))
print("Per-battery metrics:", json.dumps(report["test_by_cell"], indent=2))
for name in ("soh_trajectory.png", "learning_curve.png", "parity.png", "residuals.png"):
    display(Image(filename=str(result_dir / name)))
print("Results saved to:", result_dir)
```

The old NASA CSV run (`nasa_gru_lb12_e30_seed42`) is unchanged. The new experiment uses `nasa_mat12_gru_lb12_e30_seed42` and has separate model checkpoints and results.

The NASA source archive was previously uploaded to the ChatGPT research workspace. It is **not automatically available in Colab**. The restoration command uses an existing ZIP on Google Drive or downloads the same source from the official NASA archive and verifies its SHA-256. Original binaries are not uploaded to GitHub.

## Publish only after reviewing the full run

```python
import os, sys, subprocess
from google.colab import userdata

token = userdata.get("GithubToken")
assert token, "GithubToken unavailable"
publish_env = os.environ.copy()
publish_env["GITHUB_TOKEN"] = token
subprocess.run([
    sys.executable, "-m", "simple_soh.publish",
    "--run-dir", str(OUTPUTS / RUN_ID)
], cwd=REPO, env=publish_env, check=True)
del token, publish_env
print("Full experiment report published to private GitHub")
```

The uploader requires repository Contents: Read/write, only publishes reports/plots, never checkpoints or .mat source files, and refuses to overwrite a non-matching experiment.

The full research protocol, included battery IDs, excluded zero-capacity discharge events, and interpretation limits are documented in [MAT12 protocol](NASA_MAT12_PROTOCOL.md).
