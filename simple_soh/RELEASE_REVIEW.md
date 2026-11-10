# Scientific validation and release checklist — simple SOH baselines

The `simple_soh/` package is an independent, lightweight-model benchmark. It is not a reconstruction of the root-level wavelet/BiLSTM publication. The repository remains private until visibility is explicitly changed by its owner.

## Implemented and inspected

| Item | Validation |
|---|---|
| Data | NASA PCoE and CALCE CS2 source-derived cycle-level CSVs are versioned; checksums and counts are recorded in `data/processed/manifest.json` |
| Selected cells | 4 NASA and 4 CALCE batteries; all five selected input features finite in the inspected records |
| SOH target | `capacity_ah / first observed capacity` verified against `soh_first_observed` for each battery |
| Input leakage | Same-cycle capacity, SOH, dataset-end cycle count, and future-cycle observations excluded |
| Temporal causality | Previous 12 discharge cycles predict the subsequent cycle's SOH |
| Train/validation/test | No cell shared across partitions; splits fixed before test evaluation |
| Normalization | StandardScaler fitted exclusively on training batteries |
| Models | MLP, GRU, LSTM, CNN-LSTM with consistent input features and budget |
| Selection | Lowest validation MSE; held-out test batteries never used to select checkpoints |
| Storage | Atomic epoch-level PyTorch checkpoints with model, optimizer, RNG and source hashes |
| Outputs | Recorded metrics, predictions, source provenance and four PNG diagnostics |
| Publishing | Explicit opt-in GitHub upload; refuses public destinations, incomplete runs and conflicting replacements |
| Automation | GitHub Actions checks model gradients, dataset protocol and real-data short training runs |

Passing unit tests and one-epoch CI training checks **do not** demonstrate that full scientific training has been completed or that the model generalizes to unseen domains. The one-epoch CI reports are ephemeral technical checks and are not versioned experimental results.

## Required before comparative results are made public

- [ ] Execute all 30 epochs for NASA and CALCE using persistent checkpoints.
- [ ] Publish only the resulting complete-run metrics, predictions, training history, figures and source hashes.
- [ ] Check degradation trajectories, low-SOH outliers, and residual patterns, particularly in long CALCE sequences.
- [ ] Repeat each architecture with multiple random seeds and report mean ± standard deviation on identical splits.
- [ ] Run additional held-out batteries or leave-one-cell-out evaluations to avoid relying on one test battery per dataset.
- [ ] Include simple non-neural baselines and compare on consistent prediction horizons and metrics.
- [ ] Freeze the exact software environment (Python, PyTorch, NumPy, scikit-learn) and hardware for the final research report.
- [ ] Review licenses and redistribution permissions for datasets, pretrained files, and inherited code before changing repository visibility.
- [ ] Check commit history for tokens or personal material: editing the current branch does not erase past commits.

## Interpretation of RUL

`remaining_observed_cycles` measures the distance to the last record in a finite archive. It is **not** true remaining useful life. RUL experiments require an explicit physical or operational end-of-life criterion and careful censoring rules. This baseline suite consequently reports SOH only.

## Code provenance and presentation

The baseline package uses ordinary technical comments, documented data contracts and explicit experimental boundaries. Presentation does not claim nonexistent training runs or imply a paper-level reproduction. Any disclosure rules required by authors' institutions, conferences or journals must still be followed.

See [experiment instructions](README.md), [source data checks](DATA_QA.md) and the [Colab protocol](COLAB.md).
