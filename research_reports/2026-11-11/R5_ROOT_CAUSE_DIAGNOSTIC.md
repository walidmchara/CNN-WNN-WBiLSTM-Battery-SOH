# R5 — root-cause audit of frozen model outcomes

**Basis:** Executed `Automatika_2025_R5_Complete_Root_Cause_Diagnostic(1).ipynb` (output reports `20261111T043944Z`), reading original NASA discharge traces and the already-computed R5 predictions. No additional fitting in the diagnostic.

## Integrity, provenance and limits

- Six R5 final models reported loadable with finite weights; all six recovery snapshots located, and all six development dataset hash checks passed in the Colab diagnostic outputs.
- The frozen R5 source commit reported by the notebook: `86b6428a91fb407aef01661827fffe86686d9738`.
- Model training counts: 43°C 12 epochs × 3 seeds; 4°C 7 epochs × 3 seeds.
- Both groups were evaluated with `C_ROBUST5_DURATION`: robust-scaled measured V/I/T/t plus discharge duration, using a standard BiLSTM.
- B0030 and B0056 had been inspected in earlier studies; results here are **post-hoc, not untouched test validation**. We do not claim these runs reproduce the published paper.

## Retrospective R5 outcomes

| Group | Target | Mean RMSE | SD RMSE | Mean MAE | Mean R² | Median pred/actual SD |
|---|---|---:|---:|---:|---:|---:|
| NASA 43°C | B0030 | 0.039584 | 0.003283 | 0.038689 | -0.738907 | 0.959 |
| NASA 4°C | B0056 | 0.104856 | 0.041909 | 0.099397 | -10.191600 | 0.254 |

See [per-seed data](statistics/r5_per_seed_posthoc.csv). Against the author's published CNN–WNN–WBiLSTM–AM RMSE of 0.0039 (B0030) and 0.0046 (B0056), these **modified-model** outcomes are substantially worse and not protocol-identical.

## Observed failure modes

- **B0030 / 43°C:** prediction variability approximately follows observed SOH variability (seed SD ratios 0.938–1.016), but predicted SOH is systematically too high. Mean signed error +0.038689. Per-seed RMSE 0.036620, 0.043113 and 0.039019.
- **B0056 / 4°C:** mean signed error is negative (approximately -0.0959), predicted SOH variation only ~24–27% of actual variation; severe and seed-dependent degradation. RMSE 0.056652, 0.125269 and 0.132648. In a small development probe, the six model diagnostic reported the three 4°C model output SDs as 0.000136, 0.000179 and 0.000120; this probe is not equivalent to full train-set performance.
- **Distribution shift:** development LOBO B0029 current channel outside train-fold range 95.96%; post-hoc B0030 current outside development range 0.27%. Development 4°C B0054 temperature outside train-fold range 8.12%; post-hoc B0056 temperature 0.08%.
- **Physical associations:** discharge-duration/SOH correlations are very high for four 43°C batteries (0.965–0.994), weaker and variable for 4°C (0.542–0.825). These are descriptive within-cell correlations, not causal explanations.
- **Duration extraction:** all four 43°C batteries share reported duration min/max/mean (1537/1723/1616.30 sec rounded). Cross-check raw event time stamps before using duration as a claimed distinct physical signature. Same summary duration statistics alone do not prove extraction error.
- **Capacity trajectories:** non-monotone SOH measurements and heterogeneous protocols should be examined for comparability; do not silently smooth or alter ground-truth labels.

## Proposed next controlled checks

1. Confirm source MATLAB fields, cycle alignment, nominal-capacity SOH denominator, duration computation and discharge current units/sign.
2. Compare capacity proxies derived from physical discharge current integration with raw reported capacities, documenting whether protocol and current measurements permit meaningful integration.
3. Benchmark duration/physics feature baselines and simple regularized regression **using development-only LOBO**.
4. Inspect sensitivity to fixed epochs (12, 7), initializer and head gradient norms on development data only.
5. Use new never-tuned battery data for independent evidence. Do **not** calibrate a bias from B0030/B0056 labels and report the same batteries as untouched test.

Original notebook-produced detailed exports and plots were saved to Google Drive at:
`MyDrive/Automatika_2025_Reproduction/R5_Diagnostic_Audit/20261111T043944Z/`.
Those binary and CSV outputs were not accessible for direct copying into this GitHub archive; linked SVGs are reproduced from tabulated values.
