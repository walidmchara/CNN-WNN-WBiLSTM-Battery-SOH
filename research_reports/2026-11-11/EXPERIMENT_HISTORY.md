# Experiment chronology and interpretation (R0–R5)

| Stage | Design | Summary | Scientific status |
|---|---|---|---|
| Published 2025 | CNN 64 / WNN 127 / Morlet WBiLSTM 94 / attention 188 | B0018 RMSE 0.0041; B0030 0.0039; B0056 0.0046 | Values transcribed from the published article |
| R0 | Independent PyTorch reconstruction with 64-point resampling | B0018 RMSE 0.023332; B0030 0.036580; B0056 0.082826 | Subsequent experiments, not exact original source |
| R1 | Standard BiLSTM recurrent ablation | R0–R2 notebook contains 3-seed comparisons; R1 validation RMSE mean 43°C 0.024554; 4°C 0.016037 | Development validation only |
| R2 | Variance-regularized MSE alternative | Validation RMSE mean 43°C 0.010792; 4°C 0.039642 | Development validation only |
| R3 | 60 Optuna trials per group + LOBO and multi-seed confirmation | Selected scores 43°C 0.020104 ± 0.000166; 4°C 0.032498 ± 0.000319 | LOBO objective, not independent test |
| R4 | 3 preprocessing variants × 3 folds × 3 seeds × 2 groups | C is clearly better on 43°C LOBO but essentially tied with A at 4°C | Development validation |
| R5 | 3 seeds × 2 groups, frozen R4-derived choice C | B0030 RMSE 0.039584 ± 0.003283; B0056 0.104856 ± 0.041909 | Retrospective post-hoc test, worse than R0 |

All values above were read from user-executed notebook outputs and earlier companion audit; they are not newly measured by the archival step.

**Data integrity:** The original authors' published figures are the established reference. Modern reconstruction / optimization runs are later measurements and **do not replace paper-reported values**, including where they are less accurate.

**Research-property boundary:** Favorable but unpublished research results and internal training assets remain private laboratory research material pending applicable IP / authorship arrangements. The article's published findings are separately citable public reference values, not a claim that all later numbers appear in the paper.

**Reproducibility limitation:** Original notebook graphs and R3–R5 checkpoints are on Drive; their availability in this GitHub archive is not implied by these summaries.
