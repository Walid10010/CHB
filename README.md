# CHB — Clustering Hardness Benchmark

[![ICML 2026](https://img.shields.io/badge/ICML-2026-blue.svg)](https://openreview.net/pdf?id=9zXUOLxbcL)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Walid10010/CHB/blob/main/examples/quickstart.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
<!-- TODO (DOI): uncomment after the Zenodo release; use the all-versions concept DOI:
     [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX) -->

Official implementation of the ICML 2026 paper
**"CHB: A Diagnostic Toolkit for Hardness-Aware Clustering Evaluation"**
(Walid Durani, Philipp Jahn, Collin Leiber, David B. Hoffmann, Thomas Seidl,
Claudia Plant, Christian Böhm).

Computes the CHB hardness fingerprint **h(D) = (S; C; T)** for a labeled
dataset, plus the separability gate, the blob-calibrated topology evidence
T_evid, and the deterministic regime assignment (**A / B / C**).

<p align="center">
  <img src="https://raw.githubusercontent.com/Walid10010/CHB/main/assets/chb_framework.png" width="780"
       alt="CHB framework overview: hardness fingerprint h(D) = (S; C; T), separability gate, topology evidence T_evid, and deterministic regime assignment (A/B/C), with key empirical findings Q1-Q3">
</p>

## Citation

If you use CHB, please cite:

```bibtex
@inproceedings{durani2026chb,
  title     = {{CHB}: A Diagnostic Toolkit for Hardness-Aware Clustering Evaluation},
  author    = {Durani, Walid and Jahn, Philipp and Leiber, Collin and
               Hoffmann, David B. and Seidl, Thomas and Plant, Claudia and
               B{\"o}hm, Christian},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning (ICML)},
  series    = {Proceedings of Machine Learning Research},
  volume    = {306},
  publisher = {PMLR},
  year      = {2026}
}
```

## Install

```bash
pip install chb-clustering
```

or from source:

```bash
git clone https://github.com/Walid10010/CHB.git && cd CHB
pip install -e .
```

All dependencies (including `ripser`, which powers the topology descriptors
T2/T3) are installed automatically. The distribution is named
`chb-clustering`, the import name is `chb` — if you happen to have the
unrelated PyPI package `chb` installed, uninstall it first to avoid a file
collision.

## Usage

### Python API

```python
from chb import compute_fingerprint

fingerprint, regime = compute_fingerprint(X, y)
# fingerprint: {"S1": ..., "S2": ..., "S3": ..., "C1": ..., "C2": ..., "T1": ..., "T2": ..., "T3": ...}
# regime:      "A" (separability collapse) | "B" (topology mismatch) | "C" (scale heterogeneity)

res = compute_fingerprint(X, y)     # rich result object
res.gate, res.t_evid, res.report    # gate details, topology evidence, full combined report
```

Labels are used for diagnosis only — never to fit clustering.

### Command line

```bash
# Full CHB run on a single dataset (writes one combined JSON incl. regime)
chb both --input your.csv --label-col target
chb both --input your.npz            # expects arrays X and y/labels

# Batch over kdd_data/ (*.npz) and kdd_data_org/ (data_*/label_* pairs)
# Base dir via env var CLUSTERING_BASE_DIR (default: cwd); output: combined_results/
chb batch

# Add/refresh the CHB block (fingerprint, gate, T_evid, regime) on existing
# combined reports — understands legacy key names from earlier code versions
chb chb --report combined_results/

# Individual blocks
chb cohesion   --input your.csv
chb separation --input your.csv
```

Also available as `python -m chb ...`; the old `python chb_metrics.py ...`
entry point keeps working via a deprecation shim.

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
python tests/smoke_test.py            # all cases
python tests/smoke_test.py legacy     # fast case only
```

## Note on legacy reports

Reports produced by earlier code versions use old key names; the `chb`
annotate mode maps them automatically. Their density block
(`DensityComplexity`) is treated as outdated and is recomputed on the next
`both`/`batch` run.

## License

Released under the [MIT License](LICENSE).
