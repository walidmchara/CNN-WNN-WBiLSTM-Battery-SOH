# Experimental audits and research archive — 11 November 2026

This private repository stores a traceable summary of R0–R5 reconstruction, ablations, Optuna/LOBO, preprocessing refinement, final development training and the post-hoc R5 diagnostic. **Values from the published 2025 article must not be confused with newly measured, unpublished values.**

## Research confidentiality and publication boundary

**Laboratory research-property notice:** The optimized configurations, unpublished experiment outputs, analysis, figures and any stronger future results are treated as **private research material of the research laboratory**, subject to applicable institutional agreements, coauthor rights, and research data policies. Repository access is restricted; do not redistribute unpublished results or checkpoints without appropriate authorization. This notice does not independently establish legal ownership.

**Published numerical results:** The author-reported reference measurements are in the [2025 *Automatika* article](https://doi.org/10.1080/00051144.2025.2461821), Table 3, transcribed separately in [the published reference CSV](../../results/automatika_2025_table3_author_reported.csv). The **latest R0–R5 experimental measurements documented here are subsequent research observations and are not presented as measurements published in that paper.** No claim of performance matching the paper is made.

## Reports

- [R3 — Optuna/LOBO audit](R3_OPTUNA_LOBO_AUDIT.md)
- [R4 — preprocessing comparison](R4_PREPROCESSING_SUMMARY.md)
- [R5 — detailed scientific diagnostic](R5_ROOT_CAUSE_DIAGNOSTIC.md)
- [R0–R5 — evidence and decisions](EXPERIMENT_HISTORY.md)

## Statistics

- [Published versus new results](statistics/paper_vs_experiments.csv)
- [R5 per-seed test outcomes (post-hoc)](statistics/r5_per_seed_posthoc.csv)
- [R4 LOBO method comparison](statistics/r4_lobo_summary.csv)
- [R5 per-battery physical indicators](statistics/r5_battery_physics.csv)

## Figures

- [R5 RMSE versus published reference (SVG)](figures/r5_rmse_comparison.svg)
- [R4 LOBO preprocessing comparison (SVG)](figures/r4_lobo_comparison.svg)

These SVG figures are reconstructed from the tabulated notebook outputs, **not copies of the original Matplotlib figures on Google Drive**. Original cycle-by-cycle figures, full CSV exports, SQLite studies and binary PyTorch checkpoints remain on Drive until separately provided/uploaded. Source: user-executed R3, R4, R5 and diagnostic notebooks supplied in this conversation, not independently re-executed here.

## Interpretation

- R3 demonstrated LOBO-based selection but revealed an extensive out-of-range current shift on the B0029 fold.
- R4's 5-channel robust-plus-duration variant improved the 43°C development LOBO score; its 4°C advantage was negligible.
- R5's frozen six models were loadable according to notebook execution logs, but their **post-hoc B0030/B0056 test scores were poor**, and B0056 varied substantially across seeds.
- The current evidence **does not demonstrate test-battery generalization matching the published study**. New independent batteries and controlled protocol checks are required for a stronger claim.

This archive documents negative results as faithfully as promising ones; the paper's reference table is preserved unchanged.
