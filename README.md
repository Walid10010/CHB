# CHB — Clustering Hardness Benchmark

Computes the CHB hardness fingerprint **h(D) = (S; C; T)** for a labeled
dataset, plus the separability gate, the blob-calibrated topology evidence
T_evid, and the deterministic regime assignment (**A / B / C**), as described
in the paper *"CHB: A Diagnostic Toolkit for Hardness-Aware Clustering
Evaluation"*.

## Install

```bash
pip install -r requirements.txt
```

`ripser` is required for the topology descriptors T2/T3 (PH1/PH2). Without it
they become NaN and the B/C regime decision is undetermined (the separability
gate and Regime A still work).

## Usage

```bash
# Full CHB run on a single dataset (writes one combined JSON incl. regime)
python chb_metrics.py both --input your.csv --label-col target
python chb_metrics.py both --input your.npz          # expects arrays X and y/labels

# Batch over kdd_data/ (*.npz) and kdd_data_org/ (data_*/label_* pairs)
# Base dir via env var CLUSTERING_BASE_DIR (default: cwd); output: combined_results/
python chb_metrics.py batch

# Add/refresh the CHB block (fingerprint, gate, T_evid, regime) on existing
# combined reports — understands legacy key names from earlier code versions
python chb_metrics.py chb --report combined_results/

# Individual blocks
python chb_metrics.py cohesion   --input your.csv
python chb_metrics.py separation --input your.csv
```

Every combined report contains a top-level `"chb"` block:

```json
"chb": {
  "fingerprint": {"S1": ..., "S2": ..., "S3": ..., "C1": ..., "C2": ...,
                   "T1": ..., "T2": ..., "T3": ...},
  "separability_gate": {"SEPF": ..., "gate_fails": ...},
  "topology_evidence": {"T_evid": ..., "tau_top": 15.0},
  "regime": "A" | "B" | "C"
}
```

## Primary (CHB) descriptors

| Paper | JSON key (`dataset_summary`)            | Block      | Orientation |
|-------|-----------------------------------------|------------|-------------|
| S1    | `S1_overlap`                            | separation | ↑ harder    |
| S2    | `S2_hubness`                            | separation | ↑ harder    |
| S3    | `S3_margin`                             | separation | ↓ harder    |
| C1    | `C1_density_complexity`                 | density    | ↑ harder    |
| C2    | `C2_elongation`                         | cohesion   | ↑ harder    |
| T1    | `T1_ph0_persistence`                    | cohesion   | ↑ harder    |
| T2    | `T2_ph1_persistence`                    | cohesion   | ↑ harder    |
| T3    | `T3_ph2_persistence`                    | cohesion   | ↑ harder    |

Regime rule: **A** if the separability gate fails
(2-of-3 strict failures, equivalently `SEPF = median(S1−0.5, S2−0.33, 1.0−S3) > 0`);
**B** if the gate passes and `T_evid = Σ log(1+Tᵢ) > 15.0`; **C** otherwise.

## Secondary diagnostics

All non-CHB metrics are kept for exploration and are clearly marked with a
`sec_` prefix (e.g. `sec_margin_svm`, `sec_geodesic_tightness`,
`sec_density_composite_badness`) or live in their own blocks
(`directional` A1–A5, `baseline` meta-features, hyperparameter stability).
They are not part of the CHB fingerprint or the regime rule.

## Validation

`smoke_test.py` reproduces the paper's Appendix E.1 synthetic sanity checks
(separated blobs → C, overlapping blobs → A, elongated clusters → C with high
C2). On MNIST (n=50000, seed 42) the pipeline reproduces the paper's Table 15
fingerprint to table precision and assigns Regime B.

```bash
python smoke_test.py            # all cases
python smoke_test.py legacy     # fast case only
```

## Note on legacy reports

Reports produced by earlier code versions use old key names; the `chb`
annotate mode maps them automatically. Their density block
(`DensityComplexity`) is treated as outdated and is recomputed on the next
`both`/`batch` run.
