# Training recovery (PyTorch)

This repository saves an atomic `latest.pt` after **each completed epoch**.
The snapshot includes the model, optimizer, best validation weights/metric,
training history and Python/NumPy/PyTorch random-number generator states.
A SHA-256 configuration and input-data fingerprint prevents silent mixing of
unrelated experiments. It does **not** create or alter scientific results.

Use a **persistent checkpoint location**. The default output folder can be
ephemeral in Google Colab; mount Google Drive or a RunPod persistent volume.

Example measured-data run (replace paths and parameters with verified inputs):

```bash
python train.py --csv data/my_experiment.csv --epochs 50 \
  --search none --out outputs/experiment_01 \
  --checkpoint-dir /content/drive/MyDrive/model_recovery/experiment_01
```

After interruption, rerun the **same configuration**, adding `--resume`:

```bash
python train.py --csv data/my_experiment.csv --epochs 50 \
  --search none --out outputs/experiment_01 \
  --checkpoint-dir /content/drive/MyDrive/model_recovery/experiment_01 --resume
```

The last incomplete epoch is repeated. Never change the total number of
epochs, split, seed or data while resuming. Automatic resume of hyperparameter
search (WOA, CBO or Optuna) is **not** enabled; with `--resume`, select
`--search none`.

### What is *not* claimed

- Saved optimizer state is for continuing a run, not for publishing a validated
  checkpoint in GitHub.
- A snapshot must be copied to a persistent drive: an unmounted Colab runtime
  deletes its local files when it resets.
- Exact bitwise reproducibility across different CUDA/PyTorch versions or GPU
  architectures is not guaranteed.
- This change does not modify model equations, existing test metrics or
  manuscript values. Source dataset validation remains separate.
- Only restore checkpoints you created and trust, because PyTorch checkpoint
  files may contain Python pickle content.

Run the recovery contract test with `python -m pytest tests/test_recovery.py -q`.
