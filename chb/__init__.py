"""CHB — Clustering Hardness Benchmark.

Minimal API:

    from chb import compute_fingerprint
    fingerprint, regime = compute_fingerprint(X, y)

`fingerprint` is a dict with the eight CHB descriptors
(S1, S2, S3, C1, C2, T1, T2, T3); `regime` is "A", "B" or "C".
For everything else (full combined report, gate details, T_evid) use the
returned `CHBResult` object or the lower-level functions in `chb.metrics`.
"""
from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, Optional

from . import metrics as metrics  # re-exported submodule (CLI + all blocks)

__version__ = "1.0.0"  # keep in sync with pyproject.toml
__all__ = ["compute_fingerprint", "assign_regime", "CHBResult",
           "metrics", "__version__"]


@dataclass
class CHBResult:
    """Result of :func:`compute_fingerprint`.

    Supports tuple unpacking: ``fingerprint, regime = compute_fingerprint(X, y)``.
    """
    fingerprint: Dict[str, Optional[float]]
    regime: Optional[str]
    gate: Dict[str, Any] = field(repr=False)
    t_evid: Optional[float] = None
    report: Dict[str, Any] = field(default_factory=dict, repr=False)

    def __iter__(self) -> Iterator[Any]:
        yield self.fingerprint
        yield self.regime

    def __repr__(self) -> str:  # compact, notebook-friendly
        def _f(v):
            return f"{v:.4g}" if isinstance(v, (int, float)) else "n/a"
        fp = " ".join(f"{k}={_f(self.fingerprint.get(k))}"
                      for k in ("S1", "S2", "S3", "C1", "C2", "T1", "T2", "T3"))
        return f"CHBResult(regime={self.regime!r}, {fp}, T_evid={_f(self.t_evid)})"


def compute_fingerprint(
    X,
    y,
    *,
    standardize: bool = True,
    include_kmeans_reference: bool = False,
    verbose: bool = False,
) -> CHBResult:
    """Compute the CHB hardness fingerprint h(D) and regime for (X, y).

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Point-cloud representation of the dataset.
    y : array-like of shape (n_samples,)
        Reference labels. CHB is an external (label-conditional) diagnostic;
        labels are used for diagnosis only, never to fit clustering.
    standardize : z-score features before computing descriptors (default True,
        matching the paper pipeline).
    include_kmeans_reference : also compute the KMeans ARI/NMI convenience
        scores stored in the report (not part of the fingerprint; slower).
    verbose : if False (default), suppress console output of the block runners.

    Returns
    -------
    CHBResult with:
      * ``fingerprint`` — dict with S1, S2, S3 (separation), C1, C2 (cohesion),
        T1, T2, T3 (topology); orientations as in the paper.
      * ``regime`` — "A" (separability collapse), "B" (topology mismatch),
        "C" (scale heterogeneity), or None if undetermined.
      * ``gate`` — separability-gate details (SEPF, failure margins, thresholds).
      * ``t_evid`` — blob-calibrated topology evidence.
      * ``report`` — the full combined report (all blocks + "chb").
    """
    import numpy as np

    if y is None:
        raise ValueError(
            "CHB is an external (label-conditional) diagnostic: reference "
            "labels y are required. Pass any reference partition of X.")
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    if X.ndim != 2:
        raise ValueError("X must be a 2D array of shape (n_samples, n_features).")
    if y.shape[0] != X.shape[0]:
        raise ValueError(f"len(y)={y.shape[0]} does not match n_samples={X.shape[0]}.")

    sink = io.StringIO()
    ctx = contextlib.nullcontext() if verbose else contextlib.redirect_stdout(sink)
    combined: Dict[str, Any] = {"input": {"dataset_name": "in-memory",
                                          "n_samples": int(X.shape[0]),
                                          "n_features": int(X.shape[1])}}
    with ctx:
        _, combined["cohesion"] = metrics.run_cohesion_on_arrays(
            X, y, standardize=standardize,
            compute_kmeans_scores=include_kmeans_reference)
        _, combined["separation"] = metrics.run_separation_on_arrays(
            X, y, standardize=standardize,
            enable_sec_density_connectivity=False,
            compute_kmeans_scores=include_kmeans_reference)
        _, combined["density"] = metrics.run_density_on_arrays(
            X, y, standardize=standardize)
    combined["chb"] = metrics.compute_chb_block(combined)

    block = combined["chb"]
    return CHBResult(
        fingerprint=block["fingerprint"],
        regime=block.get("regime"),
        gate=block.get("separability_gate", {}),
        t_evid=(block.get("topology_evidence") or {}).get("T_evid"),
        report=combined,
    )


def assign_regime(fingerprint: Dict[str, Optional[float]]) -> Optional[str]:
    """Regime ("A"/"B"/"C") for a fingerprint dict with keys S1..S3, T1..T3.

    Useful for re-deriving regimes from stored fingerprints without recomputing
    descriptors.
    """
    gate = metrics.chb_separability_gate(
        fingerprint.get("S1"), fingerprint.get("S2"), fingerprint.get("S3"))
    tev = metrics.chb_topology_evidence(
        fingerprint.get("T1"), fingerprint.get("T2"), fingerprint.get("T3"))
    return metrics.chb_assign_regime(gate, tev).get("regime")
