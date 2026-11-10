# R1 ablation: standard BiLSTM vs wavelet BiLSTM

This experiment retains the R0 CNN, Morlet WNN, attention, head, dropout, RMSprop
(0.001), 58 epochs, 32 samples/batch, and train-only MinMax preprocessing.
**Only** the recurrent block changes from custom Morlet-gated bidirectional cells
to a standard bidirectional PyTorch LSTM (94 hidden units each direction).

R0 stays unchanged in `paper_reproduction/`. The R1 model is a methodological
ablation, not the reported Automatika architecture. Outcomes must not be described
as reproduction of the original model. Neither test scores nor test predictions
are computed here, to prevent repeated test-driven hyperparameter selection.

## Colab invocation

On a Colab GPU runtime with Drive mounted, and this branch cloned into `/content/r1repo`:

```python
import subprocess, sys
from pathlib import Path
repo = Path("/content/r1repo")
base = Path("/content/drive/MyDrive/Automatika_2025_Reproduction")
mat = Path("/content/drive/MyDrive/NASA_original_source/mat")
for group in ["nasa_43c", "nasa_4c"]:
    for seed in [42, 123, 2026]:
        cmd = [
            sys.executable, "-u", "-m", "ablations.run_bilstm",
            "--group", group, "--source-root", str(mat),
            "--cache-root", str(base / "cache"),
            "--output-root", str(base / "r1_validation"),
            "--checkpoint-root", str(base / "r1_checkpoints"),
            "--epochs", "58", "--seed", str(seed), "--device", "auto",
            "--resume",
        ]
        subprocess.run(cmd, cwd=repo, check=True)
```

Cloning a private repository requires GitHub authentication; do not embed
a personal access token in a notebook or Git URL. Use Colab's GithubToken secret
and the authenticated clone mechanism described in
`paper_reproduction/COLAB.md`; switch its clone command to
`git clone -b experiment/bilstm-ablation-r1 ...`.

Check `r1_validation/<run_id>/validation_metrics.json`. Compare validation
RMSE, R², and standard deviation of SOH predictions across all three seeds.
Only after the architecture/hyperparameters are frozen from development results,
run a final held-out battery evaluation in a separately recorded step.

Important: the complete data loader reads test source files to form a split,
but R1 never accesses test labels or computes predictions on test.
