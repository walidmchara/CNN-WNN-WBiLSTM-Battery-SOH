# R4 — preprocessing and temporal representation

**Basis:** Executed `Automatika_2025_R4_Generalization_Refinement_LOBO (1).ipynb`, development-only LOBO, 3 seeds per variant per group.

| Variant | Meaning | NASA 43°C LOBO objective | NASA 4°C LOBO objective |
|---|---|---:|---:|
| A | Train-fold MinMax, 4 features | 0.023062 | 0.035249 |
| B | Train-fold median/IQR robust scaling, 4 features | 0.029959 | 0.035535 |
| C | Train-fold robust scaling plus discharge duration, 5 features | **0.016182** | **0.035106** |

Method C improved the 43°C LOBO objective by about 29.8% versus A; at 4°C the 0.4% difference from A is small and must not be overstated. These scores are composite fold objectives **not** RMSE on the held-out test batteries.

**Audit note:** A previous diagnostic highlighted out-of-range current values for the 43°C B0029 LOBO fold. A robust transformation does not by itself eliminate protocol/domain mismatch. Duration may provide capacity-proxy information from the same discharge; this is same-cycle SOH estimation, not advance cycle forecasting.

**Checkpoints:** Notebook output reported 54 finished runs and successful reloads. Direct repository access to the original Drive checkpoint binaries and original generated figures was not available when this summary was archived.

**Caution:** R5 subsequently used C for both groups, despite R4's negligible 4°C gain. The test results below show why development LOBO ranking alone did not guarantee transfer.
