# Reproduction of the associated Automatika (2025) study

**Article:** Walid Mchara, Mohamed Abdellatif Khalfa and Lazhar Manai, *Advanced health state intelligent diagnosis of lithium-ion batteries based on CNN-WNN-WBiLSTM model with attention mechanism*, *Automatika* 66(2), 154–173 (2025), DOI: [10.1080/00051144.2025.2461821](https://doi.org/10.1080/00051144.2025.2461821).

**Research status:** The original NASA and CALCE source archives have been located; original NASA MATLAB files have been verified. **Exact reproduction of the article’s reported metrics is not yet established.** The independently built `simple_soh/` models and the current root `train.py` prototype are not faithful reproductions of the article.

## Author-reported numerical reference

| Test cell | Ambient temperature | CNN–WNN–WBiLSTM–AM MAE | RMSE | MAPE (reported) |
|---|---:|---:|---:|---:|
| NASA B0018 | 24 °C | **0.0036** | **0.0041** | 0.0042 |
| NASA B0030 | 43 °C | **0.0032** | **0.0039** | 0.0031 |
| NASA B0056 | 4 °C | **0.0029** | **0.0046** | 0.0039 |
| CALCE CS2-38 | — | **0.0088** | Not yet verified from full Table 3 | Not yet verified from full Table 3 |

Numbers above are reported in §4.5 / Table 3 and the article narrative, **not scores recomputed by this repository**. Do not silently treat 0.0042 MAPE as either 0.0042% or 0.42% until the tabulated convention is reconciled against Equation (26). The searchable text expresses the metric formula as a percentage and lists the paper’s numeric table entry as 0.0042. The reference record is machine-readable at [`results/published_automatika_2025_reference.json`](results/published_automatika_2025_reference.json).

## What must be reconstructed faithfully

| Protocol element | Published description | Current independent experiments | Reproduction action |
|---|---|---|---|
| Model | CNN → WNN → wavelet BiLSTM → Attention | GRU/LSTM/MLP or PyTorch component-family prototype | Recover original TensorFlow/Keras architecture, wavelet equations and learned parameters |
| NASA batteries | Separate groups 24 °C, 43 °C and 4 °C | One combined 12-cell split | Recreate separate group-specific train/test experiments |
| Input | Per-cycle matrix of voltage V, current I, temperature T and sample time t (Eq. 22) | Five cycle-level aggregate features | Process time-resolved original `.mat` traces, not only means/std/duration |
| Scaling | Min–Max to [0,1] (Eq. 23) | Training-cell-only StandardScaler | Establish paper’s scaler fitting scope and reproduce its transformation without introducing unseen deviations |
| Optimizer | RMSprop, LR 0.001 | Adam | Match RMSprop and published training settings |
| Training | Paper reports 58 epochs as sufficient for minimal loss | 30 epochs | Retrieve full Table 2 settings; do not force 30 epochs or infer early stopping |
| SOH | Capacity-based target | First-observed-capacity normalized target | Verify exact paper target denominator and treatment of first/outlying discharges |
| Evaluation | Per-battery MAE/RMSE/MAPE, Figure 8 and Table 3 | Pooled/grouped and independent task metrics | Replicate target horizon, sample aggregation, evaluated cycles and units before numeric comparison |

The exact allocation of B0053/B0054/B0055 and CALCE CS2-35/36/37, and the full parameter list in Table 2, still require a complete source-code/manuscript-level audit. B0056 and CS2-38 are the plotted test batteries; **do not infer their precise validation folds or normalize against test batteries merely because those details are absent**.

## Controls before a replication claim

1. Freeze the extracted raw MATLAB/NASA and original CALCE workbook checksums and inspect any excluded or partially recorded cycles.
2. Obtain the original executable Keras/TensorFlow code, or reconstruct each layer and hyperparameter directly from the manuscript, labeling any unresolved choice.
3. Use the same paper temperature groups and training/test batteries separately; preserve test sets during model selection.
4. Verify source traces, SOH definition, normalization fit scope and predicted-cycle alignment, including potential label leakage.
5. Execute the published training recipe with recorded seeds and checkpoint provenance, preferably repeated trials; generate comparable per-battery predictions and plots.
6. Compare reproduced values to the **published** row only after the implementation and protocol checks pass. Otherwise report them as *paper-inspired experiments*, never an exact replication.

## Existing repository files and limitations

- [`models.py`](models.py) implements a PyTorch hybrid model containing fixed Morlet filters; this is explicitly a prototype and does **not** prove fidelity to the article’s trainable WNN.
- [`train.py`](train.py) uses Adam, a 24-cycle lookback and a chronological within-file split, unlike the paper’s stated RMSprop and group-specific evaluations. Its `core.prepare()` also automatically adds the target column to its input features. That behavior needs a dedicated methodological audit and **must not** be used to validate leakage-free superiority.
- [`simple_soh/`](simple_soh/README.md) is a separate next-cycle forecasting benchmark. The 30-epoch NASA MAT12 GRU result cannot be equated with the published CNN–WNN–WBiLSTM–AM figures.

**Publication requirement:** Keep the paper reproduction and any new leakage-controlled benchmark separate, with different run identifiers, reports and clearly differentiated claims.
