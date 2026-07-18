#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CHB: Clustering Hardness Benchmark - hardness fingerprint computation.

Computes the CHB hardness fingerprint h(D) = (S; C; T) for a labeled dataset,
plus the separability gate, blob-calibrated topology evidence T_evid, and the
deterministic regime assignment (A / B / C). See the CHB paper for definitions.

PRIMARY (CHB) descriptors and their dataset_summary keys:
  S1  neighbor overlap              separation.S1_overlap
  S2  hubness infiltration          separation.S2_hubness
  S3  normalized margin thickness   separation.S3_margin
  C1  multi-scale density complex.  density.C1_density_complexity
  C2  linear elongation             cohesion.C2_elongation
  T1  PH0 component persistence     cohesion.T1_ph0_persistence
  T2  PH1 loop persistence          cohesion.T2_ph1_persistence
  T3  PH2 void persistence          cohesion.T3_ph2_persistence

SECONDARY diagnostics (kept for exploration; not part of the CHB fingerprint)
are prefixed with `sec_` (e.g. sec_margin_svm, sec_geodesic_tightness,
sec_density_composite_badness) or live in their own blocks (directional A1-A5,
baseline meta-features, hyperparameter stability).

Combined reports contain a top-level "chb" block with the fingerprint, the
separability gate (2-of-3 via median of failure margins), T_evid, and the
regime label.

Usage:
  - Single dataset:  python chb_metrics.py both --input your.csv [--label-col target]
  - Batch (dirs):    python chb_metrics.py batch
  - Annotate JSONs:  python chb_metrics.py chb --report combined_results/
  - Cohesion only:   python chb_metrics.py cohesion --input your.csv
  - Separation only: python chb_metrics.py separation --input your.csv
"""
import argparse
import json
import math
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

import os
import re

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import (
    pairwise_distances,
    silhouette_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)
from sklearn.cluster import KMeans
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree, connected_components, shortest_path

# Optional PH via ripser (PH0, PH1, PH2)
try:
    from ripser import ripser  # type: ignore
    _HAS_RIPSER = True
except Exception:
    _HAS_RIPSER = False

from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA



# =====================================================================
# Hyperparameter Stability (dataset-level) for cohesion + separation + density + baseline
# =====================================================================

def _stability_summary(values):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"median": None, "iqr": None, "std": None, "cv": None}
    med = float(np.median(v))
    iqr = float(np.percentile(v, 75) - np.percentile(v, 25))
    std = float(np.std(v))
    mean = float(np.mean(v))
    cv = float(std / (mean + 1e-12))
    return {"median": med, "iqr": iqr, "std": std, "cv": cv}


def _stable_metric_dict(list_of_dicts, keys):
    """
    list_of_dicts: list of dicts with numeric values (or None)
    keys: list of keys to aggregate
    """
    out = {}
    for k in keys:
        vals = []
        for d in list_of_dicts:
            v = d.get(k, None) if isinstance(d, dict) else None
            if v is None:
                vals.append(np.nan)
            else:
                try:
                    vals.append(float(v))
                except Exception:
                    vals.append(np.nan)
        out[k] = _stability_summary(vals)
    return out


def cohesion_hyperparam_stability(
    X,
    y=None,
    n_clusters=None,
    standardize=True,
    grid=None,
):
    """
    Returns:
      {
        "runs": [{"params":..., "wtm": {...}}, ...],
        "stability": {metric: {median,iqr,std,cv}, ...},
        "notes": {...}
      }
    Uses dataset_summary["weighted_trimmed_mean"] from run_cohesion_on_arrays.
    """
    if grid is None:
        # small but informative grid (3 runs)
        grid = [
	        dict(
		        k_fraction=0.10,
		        mst_approx_threshold=600,
		        geo_max_pairs=10_000,

		        # PH fixed
		        t1_subsample_size=800,
		        t2_subsample_size=400,
		        t3_subsample_size=384,
		        t_n_subsamples=5,
		        t_max_edge_mult=1.5,
		        t3_landmarks=128,
		        t3_vr_exact_cap=400,

		        # ONLY lifetime varies
		        t_lifetime_thresh=0.005,

		        disable_t3=False,
		        run_t3_on_low_dim=False,
	        ),
	        dict(
		        k_fraction=0.10,
		        mst_approx_threshold=600,
		        geo_max_pairs=10_000,

		        t1_subsample_size=800,
		        t2_subsample_size=400,
		        t3_subsample_size=384,
		        t_n_subsamples=5,
		        t_max_edge_mult=1.5,
		        t3_landmarks=128,
		        t3_vr_exact_cap=400,

		        t_lifetime_thresh=0.01,  # baseline

		        disable_t3=False,
		        run_t3_on_low_dim=False,
	        ),
	        dict(
		        k_fraction=0.10,
		        mst_approx_threshold=600,
		        geo_max_pairs=10_000,

		        t1_subsample_size=800,
		        t2_subsample_size=400,
		        t3_subsample_size=384,
		        t_n_subsamples=5,
		        t_max_edge_mult=1.5,
		        t3_landmarks=128,
		        t3_vr_exact_cap=400,

		        t_lifetime_thresh=0.02,

		        disable_t3=False,
		        run_t3_on_low_dim=False,
	        ),
        ]

    wtm_keys = [
        "sec_density_uniformity",
        "sec_mst_ph0_norm",
        "T1_ph0_persistence",
        "sec_geodesic_tightness",
        "sec_ph1_loopiness_norm",
        "T2_ph1_persistence",
        "sec_spectral_anisotropy",
        "C2_elongation",
        "sec_ph2_voidiness_norm",
        "T3_ph2_persistence",
    ]

    runs = []
    wtm_list = []

    for cfg in grid:
        _, rep = run_cohesion_on_arrays(
            X,
            y=y,
            n_clusters=n_clusters,
            standardize=standardize,
            k_fraction=cfg["k_fraction"],
            mst_approx_threshold=cfg["mst_approx_threshold"],
            geo_max_pairs=cfg["geo_max_pairs"],
            output_prefix=None,
            write_files=False,
            t1_subsample_size=cfg["t1_subsample_size"],
            t2_subsample_size=cfg["t2_subsample_size"],
            t3_subsample_size=cfg["t3_subsample_size"],
            t_n_subsamples=cfg["t_n_subsamples"],
            t_lifetime_thresh=cfg["t_lifetime_thresh"],
            t_max_edge_mult=cfg["t_max_edge_mult"],
            t3_landmarks=cfg["t3_landmarks"],
            t3_vr_exact_cap=cfg["t3_vr_exact_cap"],
            disable_t3=cfg["disable_t3"],
            run_t3_on_low_dim=cfg["run_t3_on_low_dim"],
        )
        wtm = rep["dataset_summary"]["weighted_trimmed_mean"]
        runs.append({"params": cfg, "wtm": _json_safe(wtm)})
        wtm_list.append(wtm)

    return {
        "runs": _json_safe(runs),
        "stability": _json_safe(_stable_metric_dict(wtm_list, wtm_keys)),
        "notes": {
            "level": "dataset_summary.weighted_trimmed_mean",
            "grid_size": int(len(grid)),
            "standardized": bool(standardize),
        },
    }


def separation_hyperparam_stability(
    X,
    y=None,
    n_clusters=None,
    standardize=True,
    metric="euclidean",
    grid=None,
):
    """
    Uses dataset_summary from run_separation_on_arrays.
    Varies k_density, k_graph, q_margin/p_margin lightly.
    """
    if grid is None:
	    grid = [
		    dict(
			    s12_k_base=10,  # S1/S2: lokaler
			    enable_sec_density_connectivity=True,  # S3 an
			    k_density=10, k_graph=8,
			    q_margin=0.10, p_margin=1  # S3 margin: lighter / more sensitive
		    ),
		    dict(
			    s12_k_base=20,  # default
			    enable_sec_density_connectivity=True,
			    k_density=15, k_graph=10,
			    q_margin=0.25, p_margin=3
		    ),
		    dict(
			    s12_k_base=40,  # S1/S2: globaler
			    enable_sec_density_connectivity=True,
			    k_density=20, k_graph=12,
			    q_margin=0.50, p_margin=5  # S3 margin: more conservative
		    ),
	    ]

    keys = [
        "S1_overlap",
        "S2_hubness",
        "sec_density_connectivity_auc",
        "S3_margin",
        "sec_margin_svm",
        "sec_margin_robust",
    ]

    runs = []
    ds_list = []

    for cfg in grid:
        _, rep = run_separation_on_arrays(
            X,
            y=y,
            n_clusters=n_clusters,
            standardize=standardize,
            metric=metric,
            noise_label=-1,
            R_k=6,
            betas=(0.001, 0.01, 0.05),
            p_margin=int(cfg["p_margin"]),
            q_margin=float(cfg["q_margin"]),
            k_density=int(cfg["k_density"]),
            k_graph=int(cfg["k_graph"]),
            density_q_grid=None,
            random_state=0,
            output_prefix=None,
            write_files=False,
            enable_S1=True,
            enable_S2=True,
            enable_sec_density_connectivity=bool(cfg.get("enable_sec_density_connectivity", False)),
	        enable_S3=True,
	        s12_k_base=int(cfg.get("s12_k_base", 20)),

        )
        ds = rep["dataset_summary"]
        runs.append({"params": cfg, "dataset_summary": _json_safe({k: ds.get(k) for k in keys})})
        ds_list.append(ds)

    return {
        "runs": _json_safe(runs),
        "stability": _json_safe(_stable_metric_dict(ds_list, keys)),
        "notes": {
            "level": "dataset_summary",
            "grid_size": int(len(grid)),
            "standardized": bool(standardize),
            "metric": str(metric),
            "sec_density_connectivity_note": "S3 is disabled by default in your combined pipeline; if you enable it in the grid it will appear here.",
        },
    }


def density_hyperparam_stability(
    X,
    y=None,
    n_clusters=None,
    standardize=True,
    metric="euclidean",
    grid=None,
):
    """
    Uses dataset_summary from run_density_on_arrays.
    Varies k_density only (the biggest driver).
    """
    if grid is None:
        grid = [dict(k_density=10), dict(k_density=25), dict(k_density=50)]

    keys = ["sec_density_composite_badness", "sec_density_shape_composite", "C1_density_complexity"]

    runs = []
    ds_list = []

    for cfg in grid:
        _, rep = run_density_on_arrays(
            X,
            y=y,
            n_clusters=n_clusters,
            standardize=standardize,
            k_density=int(cfg["k_density"]),
            metric=metric,
        )
        ds = rep["dataset_summary"]
        runs.append({"params": cfg, "dataset_summary": _json_safe({k: ds.get(k) for k in keys})})
        ds_list.append(ds)

    return {
        "runs": _json_safe(runs),
        "stability": _json_safe(_stable_metric_dict(ds_list, keys)),
        "notes": {
            "level": "dataset_summary",
            "grid_size": int(len(grid)),
            "standardized": bool(standardize),
            "metric": str(metric),
        },
    }


def baseline_hyperparam_stability(
    X,
    y=None,
    n_clusters_hint=None,
    standardize=True,
    grid=None,
):
    """
    Baseline stability: varies only the 'indices_labels' logic by changing n_clusters_hint.
    If y exists, indices use truth anyway -> stability mostly trivial (still fine).
    """
    if grid is None:
        # if hint is None, baseline uses sqrt(n) heuristic; we also try a couple fixed hints
        grid = [
            dict(n_clusters_hint=None),
            dict(n_clusters_hint=5),
            dict(n_clusters_hint=10),
        ]

    keys = [
        "silhouette",
        "davies_bouldin",
        "calinski_harabasz",
        "intrinsic_dim",
        "hopkins",
        "distance_concentration",
        "pairwise_dist_mean",
        "pairwise_dist_std",
        "pairwise_dist_median",
        "pca_ev_ratio_1",
        "pca_effective_rank",
        "n_clusters",
    ]

    # apply standardize consistently
    X_proc = StandardScaler().fit_transform(X) if standardize else np.asarray(X, dtype=float)

    runs = []
    ds_list = []

    for cfg in grid:
        ds = compute_baseline_metafeatures(
            X_proc,
            y=y,
            n_clusters_hint=cfg["n_clusters_hint"],
            rng_seed=2025,
        )
        runs.append({"params": cfg, "dataset_summary": _json_safe({k: ds.get(k) for k in keys})})
        ds_list.append(ds)

    return {
        "runs": _json_safe(runs),
        "stability": _json_safe(_stable_metric_dict(ds_list, keys)),
        "notes": {
            "level": "baseline.dataset_summary",
            "grid_size": int(len(grid)),
            "standardized": bool(standardize),
            "what_varies": "n_clusters_hint (only affects indices if y is None)",
        },
    }


def compute_hyperparam_stability_block(
    X,
    y=None,
    *,
    n_clusters=None,
    standardize=True,
    metric="euclidean",
):
    """
    Single entry point: returns block for baseline + cohesion + separation + density.
    """
    out = {
        "baseline": baseline_hyperparam_stability(
            X, y=y, n_clusters_hint=n_clusters, standardize=standardize
        ),
        "cohesion": cohesion_hyperparam_stability(
            X, y=y, n_clusters=n_clusters, standardize=standardize
        ),
        "separation": separation_hyperparam_stability(
            X, y=y, n_clusters=n_clusters, standardize=standardize, metric=metric
        ),
        "density": density_hyperparam_stability(
            X, y=y, n_clusters=n_clusters, standardize=standardize, metric=metric
        ),
        "notes": {
            "meaning": "Hyperparameter stability of dataset-level summaries across a small deterministic grid. Lower IQR/CV => more stable.",
            "scope": "dataset-level only (not per-cluster) to keep reports small and batch-friendly.",
        },
    }
    return _json_safe(out)


EPS = 1e-12

def make_k_grid(
    n: int,
    base: int,
    *,
    multipliers: tuple = (0.5, 1.0, 2.0),
    extra: tuple = (1, 2, 5, 10, 15, 20, 30),
    k_min: int = 1,
    k_max: int = 50,
) -> List[int]:
    """
    Build a deterministic grid of k values (for kNN counts excluding self),
    clipped to [k_min, min(k_max, n-1)]. Returns sorted unique list.

    Used to vary k in S1/S2/S3 without adding CLI options.
    """
    if n <= 1:
        return []
    cap = min(int(k_max), max(1, n - 1))
    ks: set[int] = set()

    b = int(base)
    if b > 0:
        for m in multipliers:
            ks.add(int(round(b * float(m))))

    for k in extra:
        ks.add(int(k))

    ks = {k for k in ks if k_min <= k <= cap}
    if not ks:
        return [cap] if cap >= k_min else []
    return sorted(ks)


# ----------------------- Core helpers -----------------------

def compute_k_for_cluster(n: int, frac: float = 0.10) -> int:
    """Adaptive k(C) = max(5, min(floor(frac*|C|), |C|-1)), clipped to [1, |C|-1]."""
    if n <= 1:
        return 0
    k = int(math.floor(frac * n))
    k = max(5, k)
    k = min(k, n - 1)
    k = max(1, k)
    return k


def _sample_unique_ints_without_replacement(high: int, m: int, rng: np.random.Generator) -> np.ndarray:
    """Floyd's algorithm: sample m unique integers from range(high) without materializing the range."""
    if m <= 0:
        return np.empty(0, dtype=np.int64)
    if m > high:
        raise ValueError(f"Cannot sample {m} unique integers from range({high}).")
    selected: set[int] = set()
    for j in range(high - m, high):
        t = int(rng.integers(0, j + 1))
        if t in selected:
            selected.add(j)
        else:
            selected.add(t)
    out = np.fromiter(selected, dtype=np.int64, count=m)
    return out


def _unrank_unordered_pairs(t: np.ndarray, n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Map ranks t in [0, C(n,2)-1] to unordered pairs (i<j) uniformly."""
    t = t.astype(np.int64, copy=False)
    a = (2 * n - 1)
    i = np.floor((a - np.sqrt(a * a - 8.0 * t)) / 2.0).astype(np.int64)
    Ti = (i * (2 * n - i - 1)) // 2
    j = (t - Ti + i + 1).astype(np.int64)
    return i, j


# =======================
# Baseline meta-features
# =======================

def _sample_rows(X: np.ndarray, max_rows: int, rng: np.random.Generator) -> np.ndarray:
    n = X.shape[0]
    if n <= max_rows:
        return X
    idx = rng.choice(n, size=max_rows, replace=False)
    return X[idx]

def sample_pairwise_distances(
    X: np.ndarray,
    max_pairs: int = 20_000,
    rng_seed: int = 42,
) -> np.ndarray:
    """Uniform sample of pairwise distances without O(n^2) memory."""
    X = np.asarray(X, float)
    n = X.shape[0]
    if n <= 1:
        return np.empty(0, dtype=float)
    num_pairs = n * (n - 1) // 2
    m = int(min(max_pairs, num_pairs))
    rng = np.random.default_rng(rng_seed)
    ranks = _sample_unique_ints_without_replacement(num_pairs, m, rng)
    i, j = _unrank_unordered_pairs(ranks, n)
    return np.linalg.norm(X[i] - X[j], axis=1)

def intrinsic_dim_levina_bickel(
    X: np.ndarray,
    k: int = 20,
    max_rows: int = 5000,
    rng_seed: int = 42,
) -> float:
    """
    Levina-Bickel MLE intrinsic dimension estimate (median over points).
    Uses kNN distances; sampled rows for speed.
    """
    X = np.asarray(X, float)
    n = X.shape[0]
    if n <= k + 1 or n <= 2:
        return float("nan")

    rng = np.random.default_rng(rng_seed)
    Xs = _sample_rows(X, max_rows=max_rows, rng=rng)
    ns = Xs.shape[0]
    k_eff = int(min(max(5, k), ns - 1))

    nn = NearestNeighbors(n_neighbors=k_eff + 1, metric="euclidean", n_jobs=-1)
    nn.fit(Xs)
    dists, _ = nn.kneighbors(Xs, return_distance=True)
    dists = np.maximum(dists[:, 1:], EPS)  # exclude self

    rk = dists[:, -1]
    logs = np.log(rk[:, None] / np.maximum(dists[:, :-1], EPS))
    denom = np.sum(logs, axis=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        ids = (k_eff - 1) / np.maximum(denom, EPS)

    ids = ids[np.isfinite(ids)]
    if ids.size == 0:
        return float("nan")
    return float(np.median(ids))

def hopkins_statistic(
    X: np.ndarray,
    m: int = 200,
    max_rows: int = 20_000,
    rng_seed: int = 42,
) -> float:
    """
    Hopkins statistic in [0,1]:
      ~0.5 random, closer to 1 indicates cluster tendency.
    Uses uniform points in feature-wise bounding box; sampling for speed.
    """
    X = np.asarray(X, float)
    n, d = X.shape
    if n <= 2:
        return float("nan")

    rng = np.random.default_rng(rng_seed)

    # Optionally downsample X to keep NN queries reasonable
    Xref = _sample_rows(X, max_rows=max_rows, rng=rng)
    nref = Xref.shape[0]
    if nref <= 2:
        return float("nan")

    m_eff = int(min(max(10, m), nref - 1))

    # Nearest neighbor distances for real points (excluding self)
    nn2 = NearestNeighbors(n_neighbors=2, metric="euclidean", n_jobs=-1).fit(Xref)
    real_idx = rng.choice(nref, size=m_eff, replace=False)
    d_real, _ = nn2.kneighbors(Xref[real_idx], return_distance=True)
    w = np.maximum(d_real[:, 1], EPS)

    # Uniform random points in bounding box
    lo = np.min(Xref, axis=0)
    hi = np.max(Xref, axis=0)
    U = rng.uniform(lo, hi, size=(m_eff, d))
    nn1 = NearestNeighbors(n_neighbors=1, metric="euclidean", n_jobs=-1).fit(Xref)
    d_u, _ = nn1.kneighbors(U, return_distance=True)
    u = np.maximum(d_u[:, 0], EPS)

    H = float(np.sum(u) / (np.sum(u) + np.sum(w) + EPS))
    return H

def pca_features(
    X: np.ndarray,
    max_rows: int = 5000,
    n_components_cap: int = 50,
    rng_seed: int = 42,
) -> Dict[str, float]:
    """
    PCA features on a sampled subset for speed.
    Returns:
      - pca_ev_ratio_1
      - pca_effective_rank (exp entropy of explained variance ratios over computed comps)
    """
    X = np.asarray(X, float)
    n, d = X.shape
    if n <= 2 or d <= 1:
        return {"pca_ev_ratio_1": float("nan"), "pca_effective_rank": float("nan")}

    rng = np.random.default_rng(rng_seed)
    Xs = _sample_rows(X, max_rows=max_rows, rng=rng)
    ns = Xs.shape[0]
    n_comp = int(min(n_components_cap, d, ns - 1))
    if n_comp < 1:
        return {"pca_ev_ratio_1": float("nan"), "pca_effective_rank": float("nan")}

    try:
        pca = PCA(n_components=n_comp, svd_solver="randomized", random_state=rng_seed)
        pca.fit(Xs)
        evr = np.asarray(pca.explained_variance_ratio_, float)
        evr = evr[np.isfinite(evr)]
        if evr.size == 0:
            return {"pca_ev_ratio_1": float("nan"), "pca_effective_rank": float("nan")}
        ev1 = float(evr[0])
        p = evr / (np.sum(evr) + EPS)
        eff_rank = float(np.exp(-np.sum(p * np.log(p + EPS))))
        return {"pca_ev_ratio_1": ev1, "pca_effective_rank": eff_rank}
    except Exception:
        return {"pca_ev_ratio_1": float("nan"), "pca_effective_rank": float("nan")}

def compute_baseline_metafeatures(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    n_clusters_hint: Optional[int] = None,
    rng_seed: int = 42,
) -> Dict[str, Any]:
    """
    Baseline features to compare against CHB in predictive probes.

    Returns keys intended to be easy to pick up in your predictor CSV:
      n_samples, dimensions, n_clusters,
      silhouette, davies_bouldin, calinski_harabasz,
      intrinsic_dim, hopkins, distance_concentration,
      pairwise_dist_mean, pairwise_dist_std, pairwise_dist_median,
      pca_ev_ratio_1, pca_effective_rank
    """
    X = np.asarray(X, float)
    n, d = X.shape
    out: Dict[str, Any] = {
        "n_samples": int(n),
        "dimensions": int(d),
        "n_clusters": None,
        "silhouette": None,
        "davies_bouldin": None,
        "calinski_harabasz": None,
        "intrinsic_dim": None,
        "hopkins": None,
        "distance_concentration": None,
        "pairwise_dist_mean": None,
        "pairwise_dist_std": None,
        "pairwise_dist_median": None,
        "pca_ev_ratio_1": None,
        "pca_effective_rank": None,
    }

    if n <= 2:
        return out

    # Pairwise distance stats (sampled)
    pdists = sample_pairwise_distances(X, max_pairs=20_000, rng_seed=rng_seed)
    if pdists.size > 0:
        mu = float(np.mean(pdists))
        sd = float(np.std(pdists))
        med = float(np.median(pdists))
        out["pairwise_dist_mean"] = mu
        out["pairwise_dist_std"] = sd
        out["pairwise_dist_median"] = med
        out["distance_concentration"] = float(sd / (mu + EPS))

    # Intrinsic dimension + Hopkins (sampled)
    out["intrinsic_dim"] = intrinsic_dim_levina_bickel(X, k=20, max_rows=5000, rng_seed=rng_seed)
    out["hopkins"] = hopkins_statistic(X, m=200, max_rows=20_000, rng_seed=rng_seed)

    # PCA features (sampled)
    out.update(pca_features(X, max_rows=5000, n_components_cap=50, rng_seed=rng_seed))

    # Cluster validity indices:
    # Use ground-truth labels if available; else use KMeans with a simple K heuristic.
    labels_for_indices: Optional[np.ndarray] = None
    if y is not None:
        y_arr = np.asarray(y).reshape(-1)
        if y_arr.shape[0] == n:
            _, y_int = np.unique(y_arr, return_inverse=True)
            labels_for_indices = y_int.astype(int)
    if labels_for_indices is None:
        K = int(n_clusters_hint) if (n_clusters_hint is not None and n_clusters_hint >= 2) else int(min(10, max(2, round(np.sqrt(n)))))
        try:
            km = KMeans(n_clusters=K, n_init=10, random_state=rng_seed)
            labels_for_indices = km.fit_predict(X).astype(int)
        except Exception:
            labels_for_indices = None

    if labels_for_indices is not None:
        K_eff = int(np.unique(labels_for_indices).size)
        out["n_clusters"] = K_eff

        # Use sampling for silhouette (O(n^2)). DB/CH are cheaper but still use sampling for huge n,d.
        rng = np.random.default_rng(rng_seed)
        Xv = X
        lv = labels_for_indices
        if n > 5000:
            idx = rng.choice(n, size=5000, replace=False)
            Xv = X[idx]
            lv = lv[idx]

        try:
            if np.unique(lv).size >= 2:
                out["silhouette"] = float(silhouette_score(Xv, lv, metric="euclidean"))
        except Exception:
            out["silhouette"] = None

        try:
            if np.unique(lv).size >= 2:
                out["davies_bouldin"] = float(davies_bouldin_score(Xv, lv))
        except Exception:
            out["davies_bouldin"] = None

        try:
            if np.unique(lv).size >= 2:
                out["calinski_harabasz"] = float(calinski_harabasz_score(Xv, lv))
        except Exception:
            out["calinski_harabasz"] = None

    return _json_safe(out)


def median_pairwise_distance(X: np.ndarray,
                             max_full_pairs: int = 100_000,
                             sample_pairs: int = 20_000) -> float:
    """Median of all pairwise distances in the cluster (exact if small, else uniform sampling)."""
    n = X.shape[0]
    if n <= 1:
        return 0.0
    num_pairs = n * (n - 1) // 2
    if num_pairs <= max_full_pairs:
        D = pairwise_distances(X, metric="euclidean", n_jobs=-1)
        iu = np.triu_indices(n, k=1)
        vals = D[iu]
        return float(np.median(vals))
    rng = np.random.default_rng(42)
    m = min(sample_pairs, num_pairs)
    ranks = _sample_unique_ints_without_replacement(num_pairs, m, rng)
    i, j = _unrank_unordered_pairs(ranks, n)
    d = np.linalg.norm(X[i] - X[j], axis=1)
    return float(np.median(d))


def build_knn_graph(X: np.ndarray, n_neighbors: int) -> csr_matrix:
    """Weighted symmetric kNN graph (Euclidean)."""
    n = X.shape[0]
    n_neighbors = int(min(max(1, n_neighbors), max(1, n - 1)))
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1, metric="euclidean", n_jobs=-1)
    nbrs.fit(X)
    dists, inds = nbrs.kneighbors(X, return_distance=True)
    dists = dists[:, 1:]
    inds = inds[:, 1:]
    rows = np.repeat(np.arange(n), n_neighbors)
    cols = inds.ravel()
    data = np.maximum(dists.ravel(), EPS)
    G = csr_matrix((data, (rows, cols)), shape=(n, n))
    G = G.maximum(G.transpose())
    return G


def _connected_components_unweighted(G: csr_matrix) -> Tuple[int, np.ndarray]:
    """Connected components ignoring edge weights."""
    A = G.copy()
    if A.nnz:
        A.data = np.ones_like(A.data)
    return connected_components(A, directed=False, return_labels=True)


def ensure_connected_graph(X: np.ndarray, k_start: int, k_max: Optional[int] = None) -> Tuple[csr_matrix, int]:
    """Increase k until kNN graph is connected or until k_max is reached (used for MST approximation)."""
    n = X.shape[0]
    if n <= 1:
        return csr_matrix((n, n)), 1
    if k_max is None:
        k_max = min(n - 1, max(k_start, 128))
    k = max(1, min(k_start, n - 1))
    while True:
        G = build_knn_graph(X, n_neighbors=k)
        num_comp, _ = _connected_components_unweighted(G)
        if num_comp == 1:
            return G, k
        k_next = min(n - 1, max(k + 5, int(k * 1.5)))
        if k_next <= k or k >= k_max:
            return G, k
        k = k_next


def _landmark_min_intercomp_dists(X: np.ndarray, labels: np.ndarray,
                                  landmarks_per_comp: int = 64, rng_seed: int = 12345) -> np.ndarray:
    """Approximate inter-component distances using random landmarks per component."""
    comps = np.unique(labels)
    C = comps.size
    rng = np.random.default_rng(rng_seed)
    anchor_sets: List[np.ndarray] = []
    for c in comps:
        idx = np.where(labels == c)[0]
        if idx.size == 0:
            anchor_sets.append(np.empty((0, X.shape[1]), dtype=float))
            continue
        take = min(landmarks_per_comp, idx.size)
        anchors_idx = rng.choice(idx, size=take, replace=False)
        anchor_sets.append(X[anchors_idx])
    Dcc = np.full((C, C), np.inf, dtype=float)
    np.fill_diagonal(Dcc, 0.0)
    for i in range(C):
        Ai = anchor_sets[i]
        if Ai.shape[0] == 0:
            continue
        for j in range(i + 1, C):
            Aj = anchor_sets[j]
            if Aj.shape[0] == 0:
                continue
            d_ij = pairwise_distances(Ai, Aj, metric="euclidean", n_jobs=-1).min()
            Dcc[i, j] = d_ij
            Dcc[j, i] = d_ij
    return Dcc


def mst_total_length_normalized(X: np.ndarray, m_C: float, approx_threshold: int = 600) -> Tuple[float, int, bool]:
    """C2: MST length normalized by (n-1)*m_C (exact if small; else MST on kNN graph + landmarks for bridges)."""
    n = X.shape[0]
    if n <= 1 or m_C <= 0:
        return float("nan"), 0, False

    if n <= approx_threshold:
        D = pairwise_distances(X, metric="euclidean", n_jobs=-1)
        np.fill_diagonal(D, 0.0)
        offdiag_zero = (D == 0.0)
        np.fill_diagonal(offdiag_zero, False)
        if np.any(offdiag_zero):
            D[offdiag_zero] = EPS
        T = minimum_spanning_tree(csr_matrix(D))
        total = float(T.sum())
        norm = total / ((n - 1) * m_C)
        return norm, 0, False

    G, k_used = ensure_connected_graph(X, k_start=min(30, n - 1), k_max=min(n - 1, 256))
    num_comp, labels = _connected_components_unweighted(G)

    if num_comp == 1:
        T = minimum_spanning_tree(G)
        total = float(T.sum())
        norm = total / ((n - 1) * m_C)
        return norm, k_used, True

    totals = 0.0
    for comp_id in range(num_comp):
        idx = np.where(labels == comp_id)[0]
        if idx.size <= 1:
            continue
        subG = G[idx][:, idx]
        T = minimum_spanning_tree(subG)
        totals += float(T.sum())

    Dcc = _landmark_min_intercomp_dists(X, labels, landmarks_per_comp=64, rng_seed=2025)
    Tcent = minimum_spanning_tree(csr_matrix(Dcc))
    bridge_sum = float(Tcent.sum())

    norm = (totals + bridge_sum) / ((n - 1) * m_C)
    return norm, k_used, True


def _weighted_median(vals: np.ndarray, weights: np.ndarray) -> float:
    """Weighted median (positive weights)."""
    if vals.size == 0 or weights.size == 0:
        return float("nan")
    w = np.asarray(weights, dtype=float)
    v = np.asarray(vals, dtype=float)
    mask = (~np.isnan(v)) & (w > 0)
    if not np.any(mask):
        return float("nan")
    v = v[mask]
    w = w[mask]
    order = np.argsort(v)
    v = v[order]
    w = w[order]
    cw = np.cumsum(w)
    half = cw[-1] * 0.5
    idx = np.searchsorted(cw, half, side="left")
    idx = min(idx, len(v) - 1)
    return float(v[idx])


def geodesic_tightness_exact_k(X: np.ndarray,
                               k_for_graph: int,
                               max_pairs: int = 10_000,
                               anchor_cap: int = 64) -> float:
    """C3: geodesic tightness using exact k(C)-NN graph with anchor-based sampling."""
    n = X.shape[0]
    if n <= 1:
        return float("nan")
    k = int(min(max(1, k_for_graph), max(1, n - 1)))
    G = build_knn_graph(X, n_neighbors=k)
    num_comp, labels = _connected_components_unweighted(G)
    rng = np.random.default_rng(123)
    comp_indices = [np.where(labels == c)[0] for c in range(num_comp)]
    comp_sizes = np.array([idx.size for idx in comp_indices], dtype=np.int64)
    comp_pairs = np.maximum(comp_sizes * (comp_sizes - 1) // 2, 0)
    total_pairs = int(comp_pairs.sum())
    if total_pairs == 0:
        return float("nan")

    medians: List[float] = []
    weights: List[float] = []

    for idx, m, npairs in zip(comp_indices, comp_sizes, comp_pairs):
        if m <= 1 or npairs == 0:
            continue
        budget = max(1, int(round(max_pairs * (npairs / total_pairs))))
        anc_count = min(anchor_cap, m)
        anchors = rng.choice(idx, size=anc_count, replace=False)
        per_anchor = max(1, budget // anc_count)

        subG = G[idx][:, idx]
        X_sub = X[idx]

        ratios: List[float] = []
        for a_global in anchors:
            a_local = int(np.where(idx == a_global)[0][0])
            dist_graph = shortest_path(subG, directed=False, indices=a_local, return_predecessors=False)
            d_metric = np.linalg.norm(X_sub - X_sub[a_local], axis=1)
            mask = np.isfinite(dist_graph) & (d_metric > 0)
            cand = np.where(mask)[0]
            if cand.size == 0:
                continue
            if cand.size > per_anchor:
                cand = rng.choice(cand, size=per_anchor, replace=False)
            ratios.extend((dist_graph[cand] / d_metric[cand]).tolist())

        if len(ratios) > 0:
            r = np.asarray(ratios, dtype=float)
            r = np.maximum(r, 1.0 - 1e-12)
            medians.append(float(np.median(r)))
            weights.append(float(m))

    if not medians:
        return float("nan")
    return _weighted_median(np.array(medians, dtype=float), np.array(weights, dtype=float))


# ----------------------- PH totals (normalized) -----------------------

def _total_persistence_from_dgm(dgm: Any) -> float:
    """Sum of lifetimes in a persistence diagram."""
    if dgm is None or len(dgm) == 0:
        return 0.0
    bars = np.asarray(dgm, dtype=float)
    if bars.size == 0:
        return 0.0
    finite = np.isfinite(bars[:, 1])
    bars = bars[finite]
    if bars.size == 0:
        return 0.0
    return float(np.sum(bars[:, 1] - bars[:, 0]))


def ph_total_persistence_normalized(X: np.ndarray,
                                    m_C: float,
                                    maxdim: int = 2,
                                    max_points_vr: int = 400,
                                    landmark_count: int = 200,
                                    seed: int = 2025) -> Tuple[float, float]:
    """
    PH1 and PH2 total persistence normalized by m_C using ripser.
    - Returns (PH1, PH2) normalized totals.
    - If ripser missing, returns (NaN, NaN).
    """
    n = X.shape[0]
    if m_C <= 0 or n <= 2:
        return 0.0, 0.0
    if not _HAS_RIPSER:
        warnings.warn("ripser not installed; PH1/PH2 (C4/C7) will be NaN.")
        return float("nan"), float("nan")

    try:
        if n <= max_points_vr:
            res = ripser(X, maxdim=maxdim, metric="euclidean")
        else:
            L = min(landmark_count, n)
            rng = np.random.default_rng(seed)
            X_perm = X.copy(); rng.shuffle(X_perm, axis=0)
            res = ripser(X_perm, maxdim=maxdim, metric="euclidean", n_perm=L)
    except Exception as e:
        warnings.warn(f"ripser failed with landmark mode ({e}); falling back to smaller exact run.")
        rng = np.random.default_rng(seed)
        take = min(max_points_vr, n)
        idx = rng.choice(n, size=take, replace=False)
        res = ripser(X[idx], maxdim=maxdim, metric="euclidean")

    dgms = res.get("dgms", [])
    dgm1 = dgms[1] if len(dgms) > 1 else []
    dgm2 = dgms[2] if (maxdim >= 2 and len(dgms) > 2) else []
    total1 = _total_persistence_from_dgm(dgm1)
    total2 = _total_persistence_from_dgm(dgm2)
    return total1 / m_C, total2 / m_C


def ph1_total_persistence_normalized(X: np.ndarray, m_C: float,
                                     max_points_vr: int = 400, landmark_count: int = 200, seed: int = 2025) -> float:
    ph1, _ = ph_total_persistence_normalized(X, m_C=m_C, maxdim=1,
                                             max_points_vr=max_points_vr, landmark_count=landmark_count, seed=seed)
    return ph1


def ph2_total_persistence_normalized(X: np.ndarray, m_C: float,
                                     max_points_vr: int = 400, landmark_count: int = 200, seed: int = 2025) -> float:
    _, ph2 = ph_total_persistence_normalized(X, m_C=m_C, maxdim=2,
                                             max_points_vr=max_points_vr, landmark_count=landmark_count, seed=seed)
    return ph2


# ----------------------- PH1-b (loops resampled) -----------------------

def compute_loop_persistence(
    X,
    maxdim: int = 1,
    lifetime_thresh: float = 0.01,
    max_edge_length: Optional[float] = None,
    distance_matrix: bool = False
) -> float:
    """Sum of lifetimes in H1 with lifetime >= threshold (infinite deaths ignored)."""
    if not _HAS_RIPSER:
        warnings.warn("ripser not installed; loop persistence (T2) will be NaN.")
        return float("nan")
    ripser_kwargs: Dict[str, Any] = {"maxdim": maxdim, "distance_matrix": distance_matrix}
    if max_edge_length is not None:
        ripser_kwargs["thresh"] = max_edge_length
    result = ripser(X, **ripser_kwargs)
    diagrams = result["dgms"]
    if len(diagrams) < 2:
        return 0.0
    loop_sum = 0.0
    for birth, death in diagrams[1]:
        if not np.isfinite(death):
            continue
        lifetime = float(death - birth)
        if lifetime >= lifetime_thresh:
            loop_sum += lifetime
    return float(loop_sum)


def compute_loop_persistence_resampled(
    X,
    n_subsamples: int = 1,
    subsample_size: int = 400,         # improved default (was 2000)
    lifetime_thresh: float = 0.01,
    max_edge_length: Optional[float] = None,
    random_state: int = 42,
    normalize_by_n: bool = False
) -> float:
    """Average H1 raw persistence over subsamples; optional normalization by subset size."""
    if not _HAS_RIPSER:
        warnings.warn("ripser not installed; loop persistence (T2) will be NaN.")
        return float("nan")
    X = np.asarray(X); N = X.shape[0]
    if N == 0:
        return float("nan")
    rng = np.random.default_rng(random_state)
    loop_values: List[float] = []
    for _ in range(n_subsamples):
        X_sub = X if N <= subsample_size else X[rng.choice(N, size=subsample_size, replace=False)]
        loop_sum = compute_loop_persistence(
            X_sub, maxdim=1, lifetime_thresh=lifetime_thresh,
            max_edge_length=max_edge_length, distance_matrix=False
        )
        if normalize_by_n and X_sub.shape[0] > 0:
            loop_sum /= float(X_sub.shape[0])
        loop_values.append(loop_sum)
    clean = [v for v in loop_values if math.isfinite(v)]
    return float(np.mean(clean)) if clean else float("nan")


# ----------------------- PH0-b (components, MST-based) -----------------------

def compute_component_persistence_resampled_mst(
    X,
    n_subsamples: int = 1,
    subsample_size: int = 800,      # improved default (was 2000)
    lifetime_thresh: float = 0.01,
    approx_threshold: int = 600,
    random_state: int = 42,
) -> float:
    """
    Resampled PH0 raw persistence computed EXACTLY from the MST (VR/Euclidean equivalence).
    For large subsets, uses MST over a connected kNN graph (same approximation policy as C2).
    """
    X = np.asarray(X); N = X.shape[0]
    if N == 0:
        return float("nan")
    rng = np.random.default_rng(random_state)
    vals: List[float] = []

    for _ in range(n_subsamples):
        X_sub = X if N <= subsample_size else X[rng.choice(N, size=subsample_size, replace=False)]
        n = X_sub.shape[0]
        if n <= 1:
            vals.append(0.0); continue

        if n <= approx_threshold:
            D = pairwise_distances(X_sub, metric="euclidean", n_jobs=-1)
            np.fill_diagonal(D, 0.0)
            offdiag_zero = (D == 0.0)
            np.fill_diagonal(offdiag_zero, False)
            if np.any(offdiag_zero):
                D[offdiag_zero] = EPS
            T = minimum_spanning_tree(csr_matrix(D))
        else:
            G, _ = ensure_connected_graph(X_sub, k_start=min(30, n - 1), k_max=min(n - 1, 256))
            T = minimum_spanning_tree(G)

        w = np.asarray(T.data, dtype=float)
        vals.append(float(np.sum(w[w >= lifetime_thresh])))

    clean = [v for v in vals if math.isfinite(v)]
    return float(np.mean(clean)) if clean else float("nan")


# ----------------------- PH2-b (voids resampled, landmarked & capped) -----------------------

def compute_void_persistence_resampled(
    X,
    n_subsamples: int = 1,
    subsample_size: int = 384,          # improved default (was 2000)
    lifetime_thresh: float = 0.01,
    max_edge_length: Optional[float] = None,  # recommend ~1.5–2.0 * mC
    random_state: int = 42,
    landmark_count: int = 128,          # enable VR landmark mode when large
    vr_exact_cap: int = 400             # exact up to this size
) -> float:
    """
    Resampled PH2 raw persistence (H2 voids): uses smaller subsamples, optional landmarking, and a radius cap.
    """
    if not _HAS_RIPSER:
        warnings.warn("ripser not installed; PH2 resampled persistence will be NaN.")
        return float("nan")

    X = np.asarray(X); N = X.shape[0]
    if N == 0:
        return float("nan")

    rng = np.random.default_rng(random_state)
    values: List[float] = []

    for _ in range(n_subsamples):
        X_sub = X if N <= subsample_size else X[rng.choice(N, size=subsample_size, replace=False)]
        n = X_sub.shape[0]
        if n == 0:
            values.append(float("nan")); continue

        if n <= vr_exact_cap:
            ripser_kwargs = {"maxdim": 2, "metric": "euclidean"}
            if max_edge_length is not None:
                ripser_kwargs["thresh"] = max_edge_length
            res = ripser(X_sub, **ripser_kwargs)
        else:
            ripser_kwargs = {"maxdim": 2, "metric": "euclidean", "n_perm": min(landmark_count, n)}
            if max_edge_length is not None:
                ripser_kwargs["thresh"] = max_edge_length
            X_perm = X_sub.copy(); rng.shuffle(X_perm, axis=0)
            res = ripser(X_perm, **ripser_kwargs)

        dgms = res.get("dgms", [])
        if len(dgms) < 3:
            values.append(0.0); continue

        void_sum = 0.0
        for birth, death in dgms[2]:
            if not np.isfinite(death):
                continue
            lifetime = float(death - birth)
            if lifetime >= lifetime_thresh:
                void_sum += lifetime
        values.append(void_sum)

    clean = [v for v in values if math.isfinite(v)]
    return float(np.mean(clean)) if clean else float("nan")


# ----------------------- Spectral metrics -----------------------

def spectral_eigs(X: np.ndarray) -> np.ndarray:
    """Eigenvalues (descending) of scatter matrix via SVD."""
    if X.shape[0] <= 1:
        return np.array([])
    Xc = X - np.mean(X, axis=0, keepdims=True)
    s = np.linalg.svd(Xc, full_matrices=False, compute_uv=False)
    lambdas = s ** 2
    lambdas = lambdas[lambdas > EPS]
    if lambdas.size == 0:
        return np.array([])
    return np.sort(lambdas)[::-1]


def spectral_anisotropy(lambdas: np.ndarray) -> float:
    """C5: ||p - u||_2 / sqrt(1 - 1/r), p_i = lambda_i / sum(lambda)."""
    r = lambdas.size
    if r == 0 or r == 1:
        return 0.0
    p = lambdas / np.sum(lambdas)
    u = np.full(r, 1.0 / r, dtype=float)
    num = np.linalg.norm(p - u)
    den = math.sqrt(1.0 - 1.0 / r)
    return float(num / den)


def linear_elongation(lambdas: np.ndarray) -> float:
    """C6: lambda1 / mean(lambda2..lambda_r); 0 if r==1."""
    r = lambdas.size
    if r == 0:
        return float("nan")
    if r == 1:
        return 0.0
    denom = np.mean(lambdas[1:]) + EPS
    return float(lambdas[0] / denom)


def _std_log_radii(log_rk: np.ndarray) -> float:
    """Spec-accurate dispersion for C1: population std of log r_k."""
    return float(np.std(log_rk, ddof=0))


def weighted_trimmed_mean(values: List[float], weights: List[float], trim: float = 0.10) -> float:
    """Weighted 10% trimmed mean (by cumulative weight)."""
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    mask = (~np.isnan(v)) & (w > 0)
    if mask.sum() == 0:
        return float("nan")
    v = v[mask]; w = w[mask]
    order = np.argsort(v); v = v[order]; w = w[order]
    cw = np.cumsum(w); total_w = cw[-1]
    if total_w <= 0:
        return float("nan")
    lower = trim * total_w; upper = (1.0 - trim) * total_w
    keep = (cw > lower) & (cw < upper)
    if keep.sum() == 0:
        return float(np.sum(v * w) / total_w)
    v_keep = v[keep]; w_keep = w[keep]
    return float(np.sum(v_keep * w_keep) / np.sum(w_keep))


# ----------------------- Results container -----------------------

@dataclass
class CohesionResult:
    cluster_id: int
    size: int
    kC: int
    mC: float
    sec_density_uniformity: float
    sec_mst_ph0_norm: float
    T1_ph0_persistence: float
    sec_geodesic_tightness: float
    sec_ph1_loopiness_norm: float
    T2_ph1_persistence: float
    sec_spectral_anisotropy: float
    C2_elongation: float
    sec_ph2_voidiness_norm: float
    T3_ph2_persistence: float
    mst_used_knn_k: int
    mst_used_approx: bool


# ----------------------- Per-cluster computation -----------------------

def compute_cohesion_for_cluster(
    XC: np.ndarray,
    k_frac: float = 0.10,
    mst_approx_threshold: int = 600,
    geo_max_pairs: int = 10_000,
    *,
    t1_subsample_size: int = 800,
    t2_subsample_size: int = 400,
    t3_subsample_size: int = 384,
    t_n_subsamples: int = 1,
    t_lifetime_thresh: float = 0.01,
    t_max_edge_mult: float = 1.5,
    t3_landmarks: int = 128,
    t3_vr_exact_cap: int = 400,
    disable_t3: bool = False,
    run_t3_on_low_dim: bool = False,
) -> Tuple[float, float, float, float, float, float, float, float, float, float, int, bool, int, float]:
    """
    Returns (CHB primaries T1/T2/T3 and C2; sec_* = secondary diagnostics):
      (sec_density_unif, sec_mst_ph0, T1, sec_geo_tight, sec_ph1_norm, T2,
       sec_spec_aniso, C2_elong, sec_ph2_norm, T3, mst_k_used, used_approx, kC, mC)
    """
    n = XC.shape[0]
    if n <= 1:
        return (float("nan"),) * 10 + (0, False, 0, 0.0)

    kC = compute_k_for_cluster(n, frac=k_frac)
    mC = median_pairwise_distance(XC)

    # secondary: density uniformity (std of log kNN radii)
    nn = NearestNeighbors(n_neighbors=min(n, kC + 1), metric="euclidean", n_jobs=-1)
    nn.fit(XC)
    dists, _ = nn.kneighbors(XC, return_distance=True)
    dists = dists[:, 1:]
    idx_col = min(kC, dists.shape[1]) - 1 if dists.shape[1] > 0 else 0
    rk = dists[:, idx_col] if dists.shape[1] > 0 else np.full(n, EPS, dtype=float)
    sec_density_unif = _std_log_radii(np.log(np.maximum(rk, EPS)))

    # secondary: normalized total MST length
    sec_mst_ph0, mst_k_used, used_approx = mst_total_length_normalized(XC, m_C=mC, approx_threshold=mst_approx_threshold)

    # CHB T1: PH0 component persistence (MST-based, resampled)
    T1 = compute_component_persistence_resampled_mst(
        XC,
        n_subsamples=t_n_subsamples,
        subsample_size=t1_subsample_size,
        lifetime_thresh=t_lifetime_thresh,
        approx_threshold=mst_approx_threshold,
        random_state=2025
    )

    # secondary: geodesic tightness
    sec_geo_tight = geodesic_tightness_exact_k(XC, k_for_graph=kC, max_pairs=geo_max_pairs)

    # secondary: normalized PH1/PH2 totals
    sec_ph1_norm, sec_ph2_norm = ph_total_persistence_normalized(XC, m_C=mC)

    # CHB T2: PH1 loop persistence (resampled)
    T2 = compute_loop_persistence_resampled(
        XC,
        n_subsamples=t_n_subsamples,
        subsample_size=t2_subsample_size,
        lifetime_thresh=t_lifetime_thresh,
        max_edge_length=(t_max_edge_mult * mC if mC > 0 else None),
        random_state=2025,
        normalize_by_n=False
    )

    # spectral: secondary anisotropy + CHB C2 (linear elongation)
    lambdas = spectral_eigs(XC)
    sec_spec_aniso = spectral_anisotropy(lambdas)
    C2_elong = linear_elongation(lambdas)

    # CHB T3: PH2 void persistence (resampled)
    T3 = 0.0
    if not disable_t3 and (run_t3_on_low_dim or XC.shape[1] >= 3):
        T3 = compute_void_persistence_resampled(
            XC,
            n_subsamples=t_n_subsamples,
            subsample_size=t3_subsample_size,
            lifetime_thresh=t_lifetime_thresh,
            max_edge_length=(t_max_edge_mult * mC if mC > 0 else None),
            random_state=2025,
            landmark_count=t3_landmarks,
            vr_exact_cap=t3_vr_exact_cap
        )

    return (sec_density_unif, sec_mst_ph0, T1, sec_geo_tight, sec_ph1_norm, T2,
            sec_spec_aniso, C2_elong, sec_ph2_norm, T3, mst_k_used, used_approx, kC, mC)


# ----------------------- IO helpers -----------------------

def load_npz(npz_path: str, max_n: Optional[int] = None, seed: int = 42, tab: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    npz_path_obj = npz_path
    if hasattr(npz_path_obj, "input"):
        path = npz_path_obj.input
        label_path = getattr(npz_path_obj, "label_path", None)
    else:
        path = npz_path
        label_path = None

    if tab:
        X = np.loadtxt(path)
        labels = np.loadtxt(label_path) if label_path is not None else None
        if max_n is not None and X.shape[0] > max_n:
            rng = np.random.default_rng(seed)
            idx = rng.choice(X.shape[0], size=max_n, replace=False)
            X = X[idx]; labels = labels[idx] if labels is not None else None
        return X, labels

    data = np.load(path)
    if "X" in data:
      X = np.asarray(data["X"])
    elif "org" in data:
      X = np.asarray(data["org"])

    elif "X_umap4" in data:
      X = np.asarray(data["X_umap4"])
    else:
      raise ValueError(f"NPZ at {path} must contain array 'X'. Keys: {list(data.keys())}")

    if X.ndim == 1:
        X = X[:, None]
    X = np.ascontiguousarray(X, dtype=np.float32)
    labels = None
    for key in ["labels", "label", "y", "Y"]:
        if key in data:
            v = np.asarray(data[key]).reshape(-1)
            if v.shape[0] != X.shape[0]:
                raise ValueError(f"Label vector '{key}' length {v.shape[0]} != n={X.shape[0]}")
            try:
                v = v.astype(int)
            except Exception:
                pass
            labels = v
            break
    if max_n is not None and X.shape[0] > max_n:
        rng = np.random.default_rng(seed)
        idx = rng.choice(X.shape[0], size=max_n, replace=False)
        X = X[idx]; labels = labels[idx] if labels is not None else None
    return X, labels


def load_data(args) -> Tuple[np.ndarray, Optional[np.ndarray], List[str]]:
    if args.input is None:
        from sklearn import datasets
        iris = datasets.load_iris()
        X = iris.data.astype(float)
        y = iris.target.astype(int)
        names = [f"f{i}" for i in range(X.shape[1])]
        return X, y, names

    df = pd.read_csv(args.input)
    if args.label_col is not None:
        if args.label_col not in df.columns:
            raise ValueError(f"--label-col '{args.label_col}' not found in CSV columns.")
        y = df[args.label_col].to_numpy()
        features_df = df.drop(columns=[args.label_col])
    else:
        y = None
        features_df = df
    features_df = features_df.select_dtypes(include=[np.number]).copy()
    if features_df.shape[1] == 0:
        raise ValueError("No numeric feature columns found.")
    X = features_df.to_numpy(dtype=float)
    names = list(features_df.columns)
    return X, y, names


def get_or_make_labels(X: np.ndarray, y: Optional[np.ndarray], n_clusters: Optional[int]) -> np.ndarray:
    if y is not None:
        _, y_int = np.unique(y, return_inverse=True)
        return y_int.astype(int)

    if n_clusters is None or n_clusters <= 1:
        max_k = min(10, max(2, X.shape[0] - 1))
        best_k, best_score, best_labels = None, -np.inf, None
        for k in range(2, max_k + 1):
            km = KMeans(n_clusters=k, n_init=10, random_state=42)
            labels = km.fit_predict(X)
            try:
                score = silhouette_score(X, labels)
            except Exception:
                score = -np.inf
            if score > best_score:
                best_k, best_score, best_labels = k, score, labels
        if best_labels is None:
            raise RuntimeError("Failed to auto-select number of clusters.")
        return best_labels.astype(int)
    else:
        km = KMeans(n_clusters=int(n_clusters), n_init=10, random_state=42)
        labels = km.fit_predict(X)
        return labels.astype(int)


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, (np.floating,)):
        val = float(obj); return val if math.isfinite(val) else None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_json_safe(v) for v in obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.ndarray,)):
        return _json_safe(obj.tolist())
    return obj


def compute_kmeans_ref_scores(X: np.ndarray, y: Optional[np.ndarray], random_state: int = 42) -> Dict[str, Any]:
    scores: Dict[str, Any] = {"has_ground_truth": False, "true_num_clusters": None, "ari": None, "nmi": None}
    if y is None:
        return scores
    y_arr = np.asarray(y)
    if y_arr.ndim > 1:
        y_arr = y_arr.reshape(-1)
    if y_arr.shape[0] != X.shape[0]:
        warnings.warn(f"compute_kmeans_ref_scores: len(y)={y_arr.shape[0]} != n_samples={X.shape[0]}; skipping ARI/NMI.")
        return scores
    uniq, inv = np.unique(y_arr, return_inverse=True)
    k_true = int(len(uniq))
    if k_true < 2 or X.shape[0] < 2:
        return scores
    km = KMeans(n_clusters=k_true, n_init=10, random_state=random_state)
    km_labels = km.fit_predict(X)
    scores.update({
        "has_ground_truth": True,
        "true_num_clusters": k_true,
        "ari": float(adjusted_rand_score(inv, km_labels)),
        "nmi": float(normalized_mutual_info_score(inv, km_labels)),
    })
    return scores


# ----------------------- In-memory cohesion pipeline -----------------------

def run_cohesion_on_arrays(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    n_clusters: Optional[int] = None,
    standardize: bool = True,
    k_fraction: float = 0.10,
    mst_approx_threshold: int = 600,
    geo_max_pairs: int = 10_000,
    output_prefix: Optional[str] = None,
    write_files: bool = False,
    *,
    t1_subsample_size: int = 800,
    t2_subsample_size: int = 400,
    t3_subsample_size: int = 384,
    t_n_subsamples: int = 1,
    t_lifetime_thresh: float = 0.01,
    t_max_edge_mult: float = 1.5,
    t3_landmarks: int = 128,
    t3_vr_exact_cap: int = 400,
    disable_t3: bool = False,
    run_t3_on_low_dim: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be 2D array.")
    n = X.shape[0]

    X_proc = StandardScaler().fit_transform(X) if standardize else X
    kmeans_scores = compute_kmeans_ref_scores(X_proc, y)
    labels = get_or_make_labels(X_proc, y, n_clusters)

    clusters: Dict[int, np.ndarray] = {int(cid): np.where(labels == cid)[0] for cid in np.unique(labels)}

    results: List[CohesionResult] = []
    weights: List[float] = []
    sec_du_list: List[float] = []; sec_mst_list: List[float] = []; T1_list: List[float] = []
    sec_geo_list: List[float] = []; sec_ph1_list: List[float] = []; T2_list: List[float] = []
    sec_sa_list: List[float] = []; C2_list: List[float] = []; sec_ph2_list: List[float] = []; T3_list: List[float] = []

    for cid, idx in clusters.items():
        XC = X_proc[idx, :]
        (sec_du, sec_mst, T1, sec_geo, sec_ph1, T2,
         sec_sa, C2_el, sec_ph2, T3, mst_k_used, used_approx, kC, mC) = compute_cohesion_for_cluster(
            XC,
            k_frac=k_fraction,
            mst_approx_threshold=mst_approx_threshold,
            geo_max_pairs=geo_max_pairs,
            t1_subsample_size=t1_subsample_size,
            t2_subsample_size=t2_subsample_size,
            t3_subsample_size=t3_subsample_size,
            t_n_subsamples=t_n_subsamples,
            t_lifetime_thresh=t_lifetime_thresh,
            t_max_edge_mult=t_max_edge_mult,
            t3_landmarks=t3_landmarks,
            t3_vr_exact_cap=t3_vr_exact_cap,
            disable_t3=disable_t3,
            run_t3_on_low_dim=run_t3_on_low_dim,
        )
        res = CohesionResult(
            cluster_id=int(cid), size=int(len(idx)), kC=int(kC), mC=float(mC),
            sec_density_uniformity=float(sec_du),
            sec_mst_ph0_norm=float(sec_mst),
            T1_ph0_persistence=float(T1),
            sec_geodesic_tightness=float(sec_geo),
            sec_ph1_loopiness_norm=float(sec_ph1),
            T2_ph1_persistence=float(T2),
            sec_spectral_anisotropy=float(sec_sa),
            C2_elongation=float(C2_el),
            sec_ph2_voidiness_norm=float(sec_ph2),
            T3_ph2_persistence=float(T3),
            mst_used_knn_k=int(mst_k_used),
            mst_used_approx=bool(used_approx)
        )
        results.append(res)

        w = len(idx) / n; weights.append(w)
        sec_du_list.append(res.sec_density_uniformity); sec_mst_list.append(res.sec_mst_ph0_norm); T1_list.append(res.T1_ph0_persistence)
        sec_geo_list.append(res.sec_geodesic_tightness); sec_ph1_list.append(res.sec_ph1_loopiness_norm); T2_list.append(res.T2_ph1_persistence)
        sec_sa_list.append(res.sec_spectral_anisotropy); C2_list.append(res.C2_elongation)
        sec_ph2_list.append(res.sec_ph2_voidiness_norm); T3_list.append(res.T3_ph2_persistence)

    # dataset-level
    sec_du_ds = weighted_trimmed_mean(sec_du_list, weights);  sec_mst_ds = weighted_trimmed_mean(sec_mst_list, weights)
    T1_ds = weighted_trimmed_mean(T1_list, weights);          sec_geo_ds = weighted_trimmed_mean(sec_geo_list, weights)
    sec_ph1_ds = weighted_trimmed_mean(sec_ph1_list, weights); T2_ds = weighted_trimmed_mean(T2_list, weights)
    sec_sa_ds = weighted_trimmed_mean(sec_sa_list, weights);  C2_ds = weighted_trimmed_mean(C2_list, weights)
    sec_ph2_ds = weighted_trimmed_mean(sec_ph2_list, weights); T3_ds = weighted_trimmed_mean(T3_list, weights)

    approx_count = sum(1 for r in results if r.mst_used_approx)
    approx_fraction = (approx_count / len(results)) if results else 0.0
    df_out = pd.DataFrame([r.__dict__ for r in results])

    report = {
        "dataset_summary": {
            "num_samples": int(n),
            "num_clusters": int(len(clusters)),
            "weighted_trimmed_mean": {
                "T1_ph0_persistence": T1_ds,
                "T2_ph1_persistence": T2_ds,
                "T3_ph2_persistence": T3_ds,
                "C2_elongation": C2_ds,
                "sec_density_uniformity": sec_du_ds,
                "sec_mst_ph0_norm": sec_mst_ds,
                "sec_geodesic_tightness": sec_geo_ds,
                "sec_ph1_loopiness_norm": sec_ph1_ds,
                "sec_spectral_anisotropy": sec_sa_ds,
                "sec_ph2_voidiness_norm": sec_ph2_ds,
            },
            "mst_approximation": {
                "clusters_using_knn_or_landmark": int(approx_count),
                "fraction": float(approx_fraction)
            },
            "kmeans_trueK_external_scores": kmeans_scores,
        },
        "per_cluster": [r.__dict__ for r in results],
        "notes": {
            "standardized": standardize,
            "k_fraction": k_fraction,
            "mst_approx_threshold": mst_approx_threshold,
            "geo_max_pairs": geo_max_pairs,
            "ph1_ph2_uses_ripser": _HAS_RIPSER,
            "mC_sampler": "uniform unordered pairs via Floyd + exact unranking",
            "topology_params": {
                "t1_subsample_size": t1_subsample_size,
                "t2_subsample_size": t2_subsample_size,
                "t3_subsample_size": t3_subsample_size,
                "t_n_subsamples": t_n_subsamples,
                "t_lifetime_thresh": t_lifetime_thresh,
                "t_max_edge_mult": t_max_edge_mult,
                "t3_landmarks": t3_landmarks,
                "t3_vr_exact_cap": t3_vr_exact_cap,
                "disable_t3": disable_t3,
                "run_t3_on_low_dim": run_t3_on_low_dim
            }
        }
    }
    report_safe = _json_safe(report)
    if write_files and output_prefix:
        json_path = f"{output_prefix}_cohesion_report.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report_safe, f, indent=2, allow_nan=False)

    return df_out, report_safe


def _print_console_summary(report: Dict[str, Any], results: List[CohesionResult], title: str) -> None:
    ds = report["dataset_summary"]; wtm = ds["weighted_trimmed_mean"]; approx = ds["mst_approximation"]
    km_scores = ds.get("kmeans_trueK_external_scores")
    print(f"\n=== {title} ===")
    print(f"Samples: {ds['num_samples']} | Clusters: {ds['num_clusters']} | PH1/PH2 via ripser: {_HAS_RIPSER}")
    print("Dataset-level (weighted 10% trimmed mean):")
    print(f"  T1 PH0 component persistence (CHB)        : {wtm['T1_ph0_persistence']:.6f}")
    print(f"  T2 PH1 loop persistence (CHB)             : {wtm['T2_ph1_persistence']:.6f}")
    print(f"  T3 PH2 void persistence (CHB)             : {wtm['T3_ph2_persistence']:.6f}")
    print(f"  C2 linear elongation (CHB)                : {wtm['C2_elongation']:.6f}")
    print(f"  sec: density uniformity                   : {wtm['sec_density_uniformity']:.6f}")
    print(f"  sec: MST total length (normalized)        : {wtm['sec_mst_ph0_norm']:.6f}")
    print(f"  sec: geodesic tightness                   : {wtm['sec_geodesic_tightness']:.6f}")
    print(f"  sec: PH1 loopiness (normalized)           : {wtm['sec_ph1_loopiness_norm']:.6f}")
    print(f"  sec: spectral anisotropy                  : {wtm['sec_spectral_anisotropy']:.6f}")
    print(f"  sec: PH2 voidiness (normalized)           : {wtm['sec_ph2_voidiness_norm']:.6f}")
    print(f"\nMST approximation: used kNN/landmark MST for {approx['clusters_using_knn_or_landmark']}/{ds['num_clusters']} clusters "
          f"({approx['fraction']:.1%}).")

    if km_scores and km_scores.get("has_ground_truth"):
        print("\nKMeans reference (K = #true labels):")
        print(f"  true #clusters (K)                 : {km_scores['true_num_clusters']}")
        print(f"  ARI (KMeans vs. truth)            : {km_scores['ari']:.6f}")
        print(f"  NMI (KMeans vs. truth)            : {km_scores['nmi']:.6f}")

    print("Per-cluster (JSON report only):")
    for r in results:
        print(
            f"  Cluster {r.cluster_id:>3} | n={r.size:>4} | kC={r.kC:>3} | "
            f"T1={r.T1_ph0_persistence:.5f}  "
            f"T2={r.T2_ph1_persistence:.5f}  "
            f"T3={r.T3_ph2_persistence:.5f}  "
            f"C2={r.C2_elongation:.5f}  "
            f"sec_du={r.sec_density_uniformity:.5f}  "
            f"sec_mst={r.sec_mst_ph0_norm:.5f}  "
            f"sec_geo={r.sec_geodesic_tightness:.5f}  "
            f"sec_ph1={r.sec_ph1_loopiness_norm:.5f}  "
            f"sec_sa={r.sec_spectral_anisotropy:.5f}  "
            f"sec_ph2={r.sec_ph2_voidiness_norm:.5f}  "
            f"MST_k={r.mst_used_knn_k:>3} approx={str(r.mst_used_approx):>5}"
        )


# ----------------------- Cohesion CLI main (CSV) -----------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compute CHB cohesion/topology profile (C2 elongation, T1/T2/T3 persistence; + secondary metrics)."
    )
    parser.add_argument("--input", type=str, default=None, help="CSV file path (defaults to Iris if omitted).")
    parser.add_argument("--label-col", type=str, default=None, help="Column name containing labels (if present).")
    parser.add_argument("--n-clusters", type=int, default=None, help="If labels absent, use this K for KMeans (auto by silhouette if omitted).")
    parser.add_argument("--no-standardize", action="store_true", help="Disable StandardScaler on features.")
    parser.add_argument("--k-fraction", type=float, default=0.10, help="Fraction for k(C).")
    parser.add_argument("--mst-approx-threshold", type=int, default=600, help="Exact MST if |C|<=this; else approximate via kNN graph.")
    parser.add_argument("--geo-max-pairs", type=int, default=10_000, help="Max sampled pairs for geodesic tightness.")
    parser.add_argument("--output-prefix", type=str, default="cohesion", help="Prefix for output files.")

    # Performance knobs
    parser.add_argument("--t1-subsample-size", type=int, default=800)
    parser.add_argument("--t2-subsample-size", type=int, default=400)
    parser.add_argument("--t3-subsample-size", type=int, default=384)
    parser.add_argument("--t-n-subsamples", type=int, default=1)
    parser.add_argument("--t-lifetime-thresh", type=float, default=0.01)
    parser.add_argument("--t-max-edge-mult", type=float, default=1.5)
    parser.add_argument("--t3-landmarks", type=int, default=128)
    parser.add_argument("--t3-vr-exact-cap", type=int, default=400)
    parser.add_argument("--disable-t3", action="store_true")
    parser.add_argument("--t3-run-on-low-dim", action="store_true")

    args = parser.parse_args()

    X, y, _ = load_data(args)
    df, report = run_cohesion_on_arrays(
        X, y=y, n_clusters=args.n_clusters,
        standardize=not args.no_standardize,
        k_fraction=args.k_fraction,
        mst_approx_threshold=args.mst_approx_threshold,
        geo_max_pairs=args.geo_max_pairs,
        output_prefix=args.output_prefix,
        write_files=True,
        t1_subsample_size=args.t1_subsample_size,
        t2_subsample_size=args.t2_subsample_size,
        t3_subsample_size=args.t3_subsample_size,
        t_n_subsamples=args.t_n_subsamples,
        t_lifetime_thresh=args.t_lifetime_thresh,
        t_max_edge_mult=args.t_max_edge_mult,
        t3_landmarks=args.t3_landmarks,
        t3_vr_exact_cap=args.t3_vr_exact_cap,
        disable_t3=args.disable_t3,
        run_t3_on_low_dim=args.t3_run_on_low_dim
    )
    results = [CohesionResult(**row) for row in df.to_dict(orient="records")]
    _print_console_summary(report, results, "CSV-based run (Cohesion)")


# =====================================================================
# Separation code
# =====================================================================

from typing import Dict as _DictT, Any as _AnyT, List as _ListT, Tuple as _TupleT, Optional as _OptionalT
import math as _math
import warnings as _warnings
import numpy as _np
import pandas as _pd
from sklearn.neighbors import NearestNeighbors as _NearestNeighbors, kneighbors_graph as _kneighbors_graph
from sklearn.preprocessing import MinMaxScaler as _MinMaxScaler
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.stats import skew, kurtosis
from sklearn.linear_model import SGDClassifier as _SGDClassifier
from sklearn.kernel_approximation import RBFSampler as _RBFSampler
from sklearn.cluster import KMeans as _KMeans
from sklearn.metrics import silhouette_score as _silhouette_score, adjusted_rand_score as _adjusted_rand_score, normalized_mutual_info_score as _normalized_mutual_info_score
from scipy.sparse import csr_matrix as _csr_matrix
from scipy.sparse.csgraph import connected_components as _connected_components
from scipy.stats import skew as _skew, kurtosis as _kurtosis

def _unique_sorted(arr) -> _np.ndarray:
    return _np.array(sorted(set(arr.tolist() if isinstance(arr, _np.ndarray) else list(arr))))

def _build_knn_indices(X: _np.ndarray, n_neighbors: int, metric: str = 'euclidean'):
    n_neighbors_eff = min(n_neighbors + 1, X.shape[0])
    try:
        nn = _NearestNeighbors(n_neighbors=n_neighbors_eff, metric=metric)
    except TypeError:
        nn = _NearestNeighbors(n_neighbors=n_neighbors_eff, metric=metric, n_jobs=None)
    nn.fit(X)
    dists, idx = nn.kneighbors(X, return_distance=True)
    if n_neighbors_eff > n_neighbors:
        dists = dists[:, 1:]
        idx = idx[:, 1:]
    return dists, idx

def _within_cluster_scale(XC: _np.ndarray, k0: int, metric='euclidean') -> float:
    if XC.shape[0] <= 1:
        return 1.0
    k_eff = min(k0 + 1, XC.shape[0])
    nn = _NearestNeighbors(n_neighbors=k_eff, metric=metric)
    nn.fit(XC)
    dists, _ = nn.kneighbors(XC, return_distance=True)
    if k_eff > 1:
        sigma = dists[:, k_eff - 1]
    else:
        sigma = _np.zeros(XC.shape[0])
    return float(_np.median(sigma))

def _point_overlap(X: _np.ndarray, labels: _np.ndarray, k: int = 20, metric: str = 'euclidean') -> _np.ndarray:
    X = _np.asarray(X, dtype=float)
    labels = _np.asarray(labels)
    n = X.shape[0]
    if n <= 1:
        return _np.zeros(n, dtype=float)
    k_eff = min(max(k, 2), n)
    nn = _NearestNeighbors(n_neighbors=k_eff, metric=metric)
    nn.fit(X)
    _, indices = nn.kneighbors(X, return_distance=True)
    overlap_scores = _np.zeros(n, dtype=float)
    for i in range(n):
        neigh_idxs = indices[i, 1:]
        if neigh_idxs.size == 0:
            overlap_scores[i] = 0.0
            continue
        my_label = labels[i]
        out_of_cluster = _np.sum(labels[neigh_idxs] != my_label)
        overlap_scores[i] = out_of_cluster / float(neigh_idxs.size)
    return overlap_scores

def _overlap_cluster_stats(labels: _np.ndarray,
                           normalized_overlap: _np.ndarray,
                           noise_label: int = -1) -> _pd.DataFrame:
    labels = _np.asarray(labels)
    df = _pd.DataFrame({'Cluster': labels, 'NormalizedOverlap': normalized_overlap})
    if noise_label is not None:
        df = df[df['Cluster'] != noise_label].copy()
    if df.empty:
        return _pd.DataFrame(columns=[
            'Cluster', 'Mean_Overlap', 'Median_Overlap', 'Std_Overlap',
            'Min_Overlap', 'Max_Overlap', 'IQR_Overlap',
            'Skewness', 'Kurtosis', 'Mean_Median_Ratio', 'Hardness_Score'
        ])

    stats = df.groupby('Cluster').agg(
        Mean_Overlap=('NormalizedOverlap', 'mean'),
        Median_Overlap=('NormalizedOverlap', 'median'),
        Std_Overlap=('NormalizedOverlap', 'std'),
        Min_Overlap=('NormalizedOverlap', 'min'),
        Max_Overlap=('NormalizedOverlap', 'max'),
        IQR_Overlap=('NormalizedOverlap',
                     lambda x: _np.percentile(x, 75) - _np.percentile(x, 25)),
        Skewness=('NormalizedOverlap', _skew),
        Kurtosis=('NormalizedOverlap', _kurtosis),
    ).reset_index()

    epsilon = 1e-10
    stats['Mean_Median_Ratio'] = stats['Mean_Overlap'] / (stats['Median_Overlap'] + epsilon)

    std = stats['Std_Overlap'].fillna(0.0)
    iqr = stats['IQR_Overlap'].fillna(0.0)
    skew_abs = stats['Skewness'].abs().fillna(0.0)
    kurt_vals = stats['Kurtosis'].fillna(1.0)
    mm_ratio = stats['Mean_Median_Ratio'].fillna(0.0)

    hardness = 0.5 * (std + iqr) + 0.3 * skew_abs + 0.2 * (1.0 - kurt_vals) + 0.3 * mm_ratio
    hardness = hardness.replace([_np.inf, -_np.inf], _np.nan).fillna(0.0)

    scaler = _MinMaxScaler()
    stats['Hardness_Score'] = scaler.fit_transform(hardness.values.reshape(-1, 1)).ravel()
    return stats

def _hubness_cluster_stats(labels: _np.ndarray,
                           infiltration: _np.ndarray,
                           noise_label: int = -1) -> _pd.DataFrame:
    labels = _np.asarray(labels)
    df = _pd.DataFrame({'Cluster': labels, 'Infiltration': infiltration})
    if noise_label is not None:
        df = df[df['Cluster'] != noise_label].copy()
    if df.empty:
        return _pd.DataFrame(columns=[
            'Cluster', 'Mean_Infiltration', 'Median_Infiltration',
            'Std_Infiltration', 'Min_Infiltration', 'Max_Infiltration',
            'IQR_Infiltration', 'Skewness', 'Kurtosis',
            'Mean_Median_Ratio', 'Hardness_Score'
        ])

    stats = df.groupby('Cluster').agg(
        Mean_Infiltration=('Infiltration', 'mean'),
        Median_Infiltration=('Infiltration', 'median'),
        Std_Infiltration=('Infiltration', 'std'),
        Min_Infiltration=('Infiltration', 'min'),
        Max_Infiltration=('Infiltration', 'max'),
        IQR_Infiltration=('Infiltration',
                          lambda x: _np.percentile(x, 75) - _np.percentile(x, 25)),
        Skewness=('Infiltration', _skew),
        Kurtosis=('Infiltration', _kurtosis),
    ).reset_index()

    epsilon = 1e-10
    stats['Mean_Median_Ratio'] = stats['Mean_Infiltration'] / (stats['Median_Infiltration'] + epsilon)

    std = stats['Std_Infiltration'].fillna(0.0)
    iqr = stats['IQR_Infiltration'].fillna(0.0)
    skew_abs = stats['Skewness'].abs().fillna(0.0)
    kurt_vals = stats['Kurtosis'].fillna(1.0)
    mm_ratio = stats['Mean_Median_Ratio'].fillna(0.0)

    hardness = 0.5 * (std + iqr) + 0.3 * skew_abs + 0.2 * (1.0 - kurt_vals) + 0.3 * mm_ratio
    hardness = hardness.replace([_np.inf, -_np.inf], _np.nan).fillna(0.0)

    scaler = _MinMaxScaler()
    stats['Hardness_Score'] = scaler.fit_transform(hardness.values.reshape(-1, 1)).ravel()
    return stats

def _density_scores(X: _np.ndarray, k_density: int, metric='euclidean') -> _np.ndarray:
    k_eff = min(k_density + 1, X.shape[0])
    nn = _NearestNeighbors(n_neighbors=k_eff, metric=metric)
    nn.fit(X)
    dists, _ = nn.kneighbors(X, return_distance=True)
    r_k = dists[:, -1] if k_eff > 1 else _np.zeros(X.shape[0])
    scores = 1.0 / (r_k + 1e-12)
    return scores

def _prepare_rdc(X: _np.ndarray,
                 density_scores: _np.ndarray,
                 k_graph: int,
                 q_grid: _ListT[float],
                 metric: str = 'euclidean') -> _DictT[str, _AnyT]:
    n = X.shape[0]
    q_grid_sorted = sorted({q for q in q_grid if 0.0 <= q <= 1.0}) or [0.5, 0.75, 0.9, 0.95]
    thr_values = [float(_np.quantile(density_scores, q)) for q in q_grid_sorted]

    k_graph_eff = max(k_graph, 2)
    n_neighbors_eff = max(1, min(k_graph_eff, n - 1))
    if n_neighbors_eff < 1:
        A = _csr_matrix((n, n))
    else:
        A = _kneighbors_graph(X, n_neighbors=n_neighbors_eff,
                              mode='connectivity', include_self=False,
                              metric=metric)
        A = A.maximum(A.T)

    keep_nodes_list: _ListT[_np.ndarray] = []
    labels_cc_list: _ListT[_OptionalT[_np.ndarray]] = []
    n_comp_list: _ListT[int] = []

    for thr in thr_values:
        keep_nodes = density_scores >= thr
        if not _np.any(keep_nodes):
            keep_nodes_list.append(keep_nodes)
            labels_cc_list.append(None)
            n_comp_list.append(0)
            continue

        Asub = A[keep_nodes][:, keep_nodes]
        n_comp, labels_cc = _connected_components(Asub, directed=False, return_labels=True)

        keep_nodes_list.append(keep_nodes)
        labels_cc_list.append(labels_cc)
        n_comp_list.append(int(n_comp))

    return {
        'q_grid': q_grid_sorted,
        'thr_values': thr_values,
        'keep_nodes': keep_nodes_list,
        'labels_cc': labels_cc_list,
        'n_comp': n_comp_list,
    }

def _rdc_for_cluster(precomp: _DictT[str, _AnyT], mask_C: _np.ndarray) -> _DictT[str, _AnyT]:
    q_grid_sorted: _ListT[float] = precomp['q_grid']
    keep_nodes_list: _ListT[_np.ndarray] = precomp['keep_nodes']
    labels_cc_list: _ListT[_OptionalT[_np.ndarray]] = precomp['labels_cc']
    n_comp_list: _ListT[int] = precomp['n_comp']

    no_cross: _ListT[bool] = []
    valid: _ListT[bool] = []
    frac_mixed_C: _ListT[float] = []

    for keep_nodes, labels_cc, n_comp in zip(keep_nodes_list,
                                             labels_cc_list,
                                             n_comp_list):
        if not _np.any(keep_nodes):
            no_cross.append(False)
            valid.append(False)
            frac_mixed_C.append(_math.nan)
            continue

        mask_C_sub = mask_C[keep_nodes]
        has_C = _np.any(mask_C_sub)
        has_notC = _np.any(~mask_C_sub)

        if not (has_C and has_notC):
            no_cross.append(False)
            valid.append(False)
            frac_mixed_C.append(_math.nan if not has_C else 0.0)
            continue

        if labels_cc is None or n_comp == 0:
            no_cross.append(False)
            valid.append(False)
            frac_mixed_C.append(_math.nan)
            continue

        comp_has_C = _np.bincount(labels_cc[mask_C_sub], minlength=n_comp) > 0
        comp_has_notC = _np.bincount(labels_cc[~mask_C_sub], minlength=n_comp) > 0
        mixed_comp = comp_has_C & comp_has_notC

        no_cross.append(not mixed_comp.any())
        valid.append(True)

        if has_C:
            mixed_labels = _np.where(mixed_comp)[0]
            if mixed_labels.size == 0:
                frac_mixed_C.append(0.0)
            else:
                is_mixed_C_node = _np.isin(labels_cc, mixed_labels) & mask_C_sub
                frac_mixed_C.append(
                    float(is_mixed_C_node.sum()) / float(mask_C_sub.sum())
                )
        else:
            frac_mixed_C.append(_math.nan)

    RDC_quantile = 0.0
    for q, nc, ok in zip(q_grid_sorted, no_cross, valid):
        if ok and nc and (q > RDC_quantile):
            RDC_quantile = q

    frac_values = _np.array(
        [f for f, ok in zip(frac_mixed_C, valid)
         if ok and (not _np.isnan(f))], dtype=float
    )
    if frac_values.size > 0:
        RDC_auc_val = float(1.0 - _np.mean(frac_values))
    else:
        RDC_auc_val = _math.nan

    return {
        'q_grid': list(q_grid_sorted),
        'no_cross_flags': list(no_cross),
        'valid_flags': list(valid),
        'frac_mixed_C': [None if _np.isnan(f) else float(f)
                         for f in frac_mixed_C],
        'RDC_quantile': float(RDC_quantile),
        'RDC_auc': RDC_auc_val,
    }

def _knn_quantile_margin(X: _np.ndarray,
                         C_idx: _np.ndarray,
                         p: int = 3,
                         q: float = 0.25,
                         metric='euclidean') -> float:
    n = X.shape[0]
    mask_C = _np.zeros(n, dtype=bool)
    mask_C[C_idx] = True
    notC_idx = _np.where(~mask_C)[0]
    XC = X[C_idx]
    if notC_idx.size == 0 or XC.shape[0] == 0:
        return _math.nan

    p_eff = min(p, notC_idx.size)
    nn = _NearestNeighbors(n_neighbors=p_eff, metric=metric)
    nn.fit(X[notC_idx])
    dists, _ = nn.kneighbors(XC, return_distance=True)
    delta_p = dists[:, -1]
    return float(_np.quantile(delta_p, q))

def _svm_margin(X: _np.ndarray,
                C_idx: _np.ndarray,
                C: float = 1.0,
                gamma: float = 1.0,
                n_components: int = 500,
                random_state: int = 0) -> float:
    n = X.shape[0]
    if n <= 1 or len(C_idx) == 0 or len(C_idx) == n:
        return _math.nan

    y = _np.full(n, -1, dtype=int)
    y[C_idx] = 1

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    rbf_feature = _RBFSampler(gamma=gamma,
                              n_components=n_components,
                              random_state=random_state)
    X_features = rbf_feature.fit_transform(X_scaled)

    clf = _SGDClassifier(loss='hinge',
                         alpha=1.0 / C,
                         max_iter=1000,
                         tol=0.001,
                         random_state=random_state)
    clf.fit(X_features, y)
    w = clf.coef_.ravel()
    norm_w = _np.linalg.norm(w)
    if norm_w == 0 or not _np.isfinite(norm_w):
        return _math.nan
    margin = 1.0 / (norm_w + 1e-12)
    return float(margin)

def compute_separation_profile(
    X: _np.ndarray,
    labels: _np.ndarray,
    metric: str = 'euclidean',
    noise_label: int = -1,
    R_k: int = 6,
    betas: _TupleT[float, ...] = (0.001, 0.01, 0.05),
    p_margin: int = 3,
    q_margin: float = 0.25,
    k_density: int = 15,
    k_graph: int = 10,
    density_q_grid: _OptionalT[_ListT[float]] = None,
    random_state: int = 0,
    enable_S1: bool = True,
    enable_S2: bool = True,
    enable_sec_density_connectivity: bool = False,
    enable_S3: bool = True,
	s12_k_base: int = 20,

) -> _DictT[str, _AnyT]:


    X = _np.asarray(X, dtype=float)
    labels = _np.asarray(labels)
    if X.ndim != 2:
        raise ValueError('X must be a 2D array.')
    if labels.shape[0] != X.shape[0]:
        raise ValueError('labels must have same length as X.')

    n, _ = X.shape
    if density_q_grid is None:
        density_q_grid = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.97, 0.99]

    unique_labels = _unique_sorted(labels[labels != noise_label])

    # ------------------------------------------------------------
    # Multi-k grids (NO CLI changes):
    # ------------------------------------------------------------
    # S1/S2 (neighbors excluding self)
    k_overlap_grid = make_k_grid(
	    n, base=int(s12_k_base), k_min=1, k_max=50, extra=(1, 2, 5, 10, 20, 30)
    )
    k_hub_grid = make_k_grid(
	    n, base=int(s12_k_base), k_min=1, k_max=50, extra=(1, 2, 5, 10, 20, 30)
    )

    # Shared kNN for S1 + S2 (single query at k_max, reuse prefixes)
    nn_idx_max = None
    cross_max = None
    k_max_global = 0
    if n > 1 and ((enable_S1 and k_overlap_grid) or (enable_S2 and k_hub_grid)):
        k_max_global = int(max(k_overlap_grid + k_hub_grid))
        _, nn_idx_max = _build_knn_indices(X, n_neighbors=k_max_global, metric=metric)
        if nn_idx_max.size > 0:
            nbr_labels = labels[nn_idx_max]                    # shape (n, k_max)
            cross_max = (nbr_labels != labels[:, None])        # shape (n, k_max) boolean

    # ------------------------------------------------------------
    # S1: overlap, aggregated over k by nanmedian
    # ------------------------------------------------------------
    # ---------------- S1 (CHB-konform) ----------------
    # O_k(i) = Anteil fremder Labels unter den k-NN von Punkt i
    # µ_c(k) = Mittelwert von O_k(i) über Punkte im Cluster c
    # µ_c    = Median über k von µ_c(k)

    overlap_mu_by_cluster = {}

    if enable_S1 and (nn_idx_max is not None) and (cross_max is not None) and k_overlap_grid:
        # prefix sums: wie viele fremde Nachbarn bis Rang j
        cross_cum = np.cumsum(cross_max, axis=1, dtype=np.int32)

        # optional: Noise ausschließen
        valid = (labels != noise_label) if (noise_label is not None) else np.ones(n, dtype=bool)
        if np.any(valid):
            yv = labels[valid]
            cl_ids, inv = np.unique(yv, return_inverse=True)
            C = cl_ids.size
            sizes = np.bincount(inv, minlength=C).astype(float)

            k_list = sorted(set(int(k) for k in k_overlap_grid))
            mu_mat = np.full((len(k_list), C), np.nan, dtype=float)

            for t, k in enumerate(k_list):
                if k < 1 or k > cross_cum.shape[1]:
                    continue
                Ok = cross_cum[:, k - 1].astype(float) / float(k)  # O_k(i)
                Okv = Ok[valid]
                sums = np.bincount(inv, weights=Okv, minlength=C)
                mu_mat[t] = np.divide(sums, sizes, out=np.full(C, np.nan), where=(sizes > 0))

            mu_c = np.nanmedian(mu_mat, axis=0)  # Median über k
            overlap_mu_by_cluster = {
                int(cid): float(val) for cid, val in zip(cl_ids, mu_c)
            }

    # ------------------------------------------------------------
    # S2: hubness infiltration, aggregated over k by nanmedian
    # ------------------------------------------------------------
    # ---------------- S2 (CHB-konform) ----------------
    # hk(v)       = in-degree in directed kNN graph
    # hcrossk(v)  = in-degree from points with different label
    # Ik(v)       = hcrossk(v)/hk(v) if hk(v)>0 else 0
    # µc(k)       = mean_{v in cluster c} Ik(v)
    # µc          = median_k µc(k)

    hub_mu_by_cluster = {}

    if enable_S2 and (nn_idx_max is not None) and (cross_max is not None) and k_hub_grid:
	    valid = (labels != noise_label) if (noise_label is not None) else _np.ones(n, dtype=bool)
	    if _np.any(valid):
		    yv = labels[valid]
		    cl_ids, inv = _np.unique(yv, return_inverse=True)
		    C = cl_ids.size
		    sizes = _np.bincount(inv, minlength=C).astype(float)

		    k_targets = sorted(set(int(k) for k in k_hub_grid if int(k) >= 1))
		    k_targets = [k for k in k_targets if k <= nn_idx_max.shape[1]]

		    if k_targets:
			    total_counts = _np.zeros(n, dtype=_np.int64)  # hk(v)
			    cross_counts = _np.zeros(n, dtype=_np.int64)  # hcrossk(v)
			    mu_rows = []

			    k_max = int(max(k_targets))
			    k_target_set = set(k_targets)

			    for j in range(k_max):
				    dest = nn_idx_max[:, j]
				    total_counts += _np.bincount(dest, minlength=n)

				    mask_cross_j = cross_max[:, j]
				    if _np.any(mask_cross_j):
					    cross_counts += _np.bincount(dest[mask_cross_j], minlength=n)

				    k_now = j + 1
				    if k_now in k_target_set:
					    Ik = _np.divide(
						    cross_counts.astype(float),
						    total_counts.astype(float),
						    out=_np.zeros(n, dtype=float),
						    where=(total_counts > 0),
					    )
					    Ikv = Ik[valid]
					    sums = _np.bincount(inv, weights=Ikv, minlength=C)
					    mu_k = _np.divide(sums, sizes, out=_np.full(C, _np.nan), where=(sizes > 0))
					    mu_rows.append(mu_k)

			    if mu_rows:
				    mu_c = _np.nanmedian(_np.vstack(mu_rows), axis=0)  # median über k
				    hub_mu_by_cluster = {int(cid): float(val) for cid, val in zip(cl_ids, mu_c)}

    # ------------------------------------------------------------
    # secondary: density connectivity, vary (k_density, k_graph) configs internally
    # ------------------------------------------------------------
    rdc_precomps: _OptionalT[_ListT[_TupleT[int, int, _DictT[str, _AnyT]]]] = None
    if enable_sec_density_connectivity and n > 2:
        def _clip_k(v: float) -> int:
            # k for graph/density uses neighbors excluding self -> max n-1
            return int(max(1, min(int(round(v)), min(50, n - 1))))

        kd0 = _clip_k(k_density)
        kg0 = _clip_k(k_graph)

        cfgs: _ListT[_TupleT[int, int]] = []
        for mult in (0.5, 1.0, 2.0):
            kd = _clip_k(kd0 * mult)
            kg = _clip_k(kg0 * mult)
            if (kd, kg) not in cfgs:
                cfgs.append((kd, kg))

        rdc_precomps = []
        for kd, kg in cfgs:
            dens_scores = _density_scores(X, k_density=kd, metric=metric)
            pre = _prepare_rdc(
                X,
                density_scores=dens_scores,
                k_graph=kg,
                q_grid=density_q_grid,
                metric=metric
            )
            rdc_precomps.append((kd, kg, pre))

    # ------------------------------------------------------------
    # Aggregate outputs
    # ------------------------------------------------------------
    per_cluster: _DictT[int, _AnyT] = {}
    agg_values: _DictT[str, _ListT[float]] = {
        'S1_overlap': [],
        'S2_hubness': [],
        'sec_density_connectivity_auc': [],
        'S3_margin': [],
        'sec_margin_svm': [],
        'sec_margin_robust': [],
    }
    agg_weights: _DictT[str, _ListT[float]] = {k: [] for k in agg_values}

    for c in unique_labels:
        C_idx = _np.where(labels == c)[0]
        nC = int(C_idx.size)
        w = nC / float(n) if n > 0 else 0.0

        info: _DictT[str, _AnyT] = {
            'size': nC,
            'k_grid': {
                'S1_overlap_k': k_overlap_grid if enable_S1 else [],
                'S2_hub_k': k_hub_grid if enable_S2 else [],
                'sec_density_connectivity_configs': [{'k_density': kd, 'k_graph': kg} for kd, kg, _ in (rdc_precomps or [])] if enable_sec_density_connectivity else [],
                'S3_k0_grid': [],
                'S3_p_grid': [],
            },
            'S1_overlap': float('nan'),
            'S2_hubness': float('nan'),
            'S2_details': {},
            'sec_density_connectivity_quantile': float('nan'),
            'sec_density_connectivity_auc': float('nan'),
            'sec_density_connectivity_details': {'configs': []},
            'S3_margin': float('nan'),
            'sec_margin_svm': float('nan'),
            'sec_margin_robust': float('nan'),
        }
        # ---------------- S1 ----------------
        if enable_S1 and overlap_mu_by_cluster:
	        mu_c = overlap_mu_by_cluster.get(int(c), np.nan)
	        info['S1_overlap'] = mu_c
	        agg_values['S1_overlap'].append(mu_c)
	        agg_weights['S1_overlap'].append(w)

        # ---------------- S2 ----------------

        if enable_S2 and hub_mu_by_cluster:
	        mu_c = hub_mu_by_cluster.get(int(c), _math.nan)
	        info['S2_hubness'] = mu_c
	        agg_values['S2_hubness'].append(mu_c)
	        agg_weights['S2_hubness'].append(w)

        # ---------------- S3 (aggregate across configs) ----------------
        if enable_sec_density_connectivity and rdc_precomps is not None:
            mask_C = _np.zeros(n, dtype=bool)
            mask_C[C_idx] = True

            aucs = []
            qs = []
            cfg_details = []
            for kd, kg, pre in rdc_precomps:
                rdc = _rdc_for_cluster(pre, mask_C)
                aucs.append(rdc['RDC_auc'])
                qs.append(rdc['RDC_quantile'])
                cfg_details.append({
                    'k_density': int(kd),
                    'k_graph': int(kg),
                    'RDC_auc': rdc['RDC_auc'],
                    'RDC_quantile': rdc['RDC_quantile'],
                })

            auc_arr = _np.asarray(aucs, dtype=float)
            q_arr = _np.asarray(qs, dtype=float)

            info['sec_density_connectivity_auc'] = (
                float(_np.nanmedian(auc_arr)) if _np.any(_np.isfinite(auc_arr)) else float('nan')
            )
            info['sec_density_connectivity_quantile'] = (
                float(_np.nanmedian(q_arr)) if _np.any(_np.isfinite(q_arr)) else float('nan')
            )
            info['sec_density_connectivity_details'] = {'configs': cfg_details, 'agg': 'nanmedian'}

            agg_values['sec_density_connectivity_auc'].append(info['sec_density_connectivity_auc'])
            agg_weights['sec_density_connectivity_auc'].append(w)

        # ---------------- S3: normalized kNN margin (vary k0 and p) ----------------
        if enable_S3:
            if nC > 1:
                XC = X[C_idx]

                # k0 grid around 5% of cluster size
                k0_base = max(1, int(_math.floor(0.05 * nC)))
                k0_grid = make_k_grid(nC, base=k0_base, k_min=1, k_max=50, extra=(1, 2, 5, 10, 20))
                k0_grid = [int(k) for k in k0_grid if int(k) <= nC - 1]
                if not k0_grid:
                    k0_grid = [1]
                info['k_grid']['S3_k0_grid'] = k0_grid

                # sC: median over k0-grid of median distance-to-k0NN (exclude self at col0)
                k0_max = int(max(k0_grid))
                nnC = _NearestNeighbors(n_neighbors=min(nC, k0_max + 1), metric=metric)
                nnC.fit(XC)
                dC, _ = nnC.kneighbors(XC, return_distance=True)
                sigmas = []
                for k0 in k0_grid:
                    k0_eff = min(int(k0), nC - 1)
                    sigmas.append(float(_np.median(dC[:, k0_eff])))
                sC = float(_np.nanmedian(_np.asarray(sigmas, dtype=float))) if sigmas else float('nan')

                notC_idx = _np.where(labels != c)[0]
                if notC_idx.size > 0 and _np.isfinite(sC) and sC > 0:
                    # p grid for outside margin
                    p_grid = make_k_grid(n, base=p_margin, k_min=1, k_max=25, extra=(1, 2, 3, 5, 10))
                    p_grid_eff = sorted({int(min(p, notC_idx.size)) for p in p_grid if int(p) >= 1})
                    info['k_grid']['S3_p_grid'] = p_grid_eff

                    if p_grid_eff:
                        p_max = int(max(p_grid_eff))
                        nn_out = _NearestNeighbors(n_neighbors=min(p_max, notC_idx.size), metric=metric)
                        nn_out.fit(X[notC_idx])
                        d_out, _ = nn_out.kneighbors(XC, return_distance=True)

                        ratios = []
                        for p in p_grid_eff:
                            p_eff = int(min(p, d_out.shape[1]))
                            delta_p = d_out[:, p_eff - 1]
                            m_knn = float(_np.quantile(delta_p, q_margin))
                            ratios.append(m_knn / sC)
                        ratios_arr = _np.asarray(ratios, dtype=float)
                        info['S3_margin'] = (
                            float(_np.nanmedian(ratios_arr)) if _np.any(_np.isfinite(ratios_arr)) else float('nan')
                        )

                    # SVM margin (keep behavior; normalize by sC)
                    m_svm = _svm_margin(
                        X, C_idx,
                        C=1.0, gamma=1.0,
                        n_components=500,
                        random_state=random_state
                    )
                    info['sec_margin_svm'] = (
                        float(m_svm / sC) if _np.isfinite(m_svm) else float('nan')
                    )

                    if (_np.isfinite(info['S3_margin']) and _np.isfinite(info['sec_margin_svm'])
                            and info['S3_margin'] > 0 and info['sec_margin_svm'] > 0):
                        info['sec_margin_robust'] = float(_math.sqrt(info['S3_margin'] * info['sec_margin_svm']))

            agg_values['S3_margin'].append(info['S3_margin'])
            agg_weights['S3_margin'].append(w)
            agg_values['sec_margin_svm'].append(info['sec_margin_svm'])
            agg_weights['sec_margin_svm'].append(w)
            agg_values['sec_margin_robust'].append(info['sec_margin_robust'])
            agg_weights['sec_margin_robust'].append(w)

        per_cluster[int(c)] = info

    # ------------------------------------------------------------
    # Dataset-level aggregation (unchanged)
    # ------------------------------------------------------------
    def sep_weighted_trimmed_mean(values: _ListT[float],
                                  weights: _ListT[float],
                                  trim: float = 0.1) -> float:
        vals = _np.array(values, dtype=float)
        w = _np.array(weights, dtype=float)
        if vals.size == 0 or w.size == 0:
            return _math.nan
        mask = ~_np.isnan(vals) & (w > 0)
        if mask.sum() == 0:
            return _math.nan
        vals = vals[mask]
        w = w[mask]
        order = _np.argsort(vals)
        vals = vals[order]
        w = w[order]
        cumw = _np.cumsum(w) / w.sum()
        keep = (cumw >= trim) & (cumw <= 1.0 - trim)
        if keep.sum() == 0:
            return float(_np.average(vals, weights=w))
        return float(_np.average(vals[keep], weights=w[keep]))

    def dataset_metric(key: str, enabled: bool) -> _OptionalT[float]:
        if not enabled:
            return None
        vals = agg_values[key]
        wts = agg_weights[key]
        if len(vals) == 0:
            return float('nan')
        return sep_weighted_trimmed_mean(vals, wts)

    dataset_summary = {
        'n_samples': int(n),
        'n_clusters': int(len(per_cluster)),
        'S1_overlap': dataset_metric('S1_overlap', enable_S1),
        'S2_hubness': dataset_metric('S2_hubness', enable_S2),
        'sec_density_connectivity_auc': dataset_metric('sec_density_connectivity_auc', enable_sec_density_connectivity),
        'S3_margin': dataset_metric('S3_margin', enable_S3),
        'sec_margin_svm': dataset_metric('sec_margin_svm', enable_S3),
        'sec_margin_robust': dataset_metric('sec_margin_robust', enable_S3),
        'orientation': {
            'S1_overlap': 'higher is harder (CHB S1)',
            'S2_hubness': 'higher is harder (CHB S2)',
            'S3_margin': 'lower is harder (CHB S3)',
            'sec_density_connectivity_auc': 'larger is better (secondary)',
            'sec_margin_svm': 'larger is better (secondary)',
            'sec_margin_robust': 'larger is better (secondary)',
        },
    }

    return {
        'per_cluster': per_cluster,
        'dataset_summary': dataset_summary,
    }


def sep_load_data(args) -> Tuple[_np.ndarray, _OptionalT[_np.ndarray], _ListT[str]]:
    if args.input is None:
        from sklearn import datasets
        iris = datasets.load_iris()
        X = iris.data.astype(float)
        y = iris.target.astype(int)
        names = [f'f{i}' for i in range(X.shape[1])]
        return X, y, names

    df = _pd.read_csv(args.input)
    if args.label_col is not None:
        if args.label_col not in df.columns:
            raise ValueError(f"--label-col '{args.label_col}' not found in CSV columns.")
        y = df[args.label_col].to_numpy()
        features_df = df.drop(columns=[args.label_col])
    else:
        y = None
        features_df = df

    features_df = features_df.select_dtypes(include=[_np.number]).copy()
    if features_df.shape[1] == 0:
        raise ValueError('No numeric feature columns found.')
    X = features_df.to_numpy(dtype=float)
    names = list(features_df.columns)
    return X, y, names

def sep_get_or_make_labels(X: _np.ndarray,
                           y: _OptionalT[_np.ndarray],
                           n_clusters: _OptionalT[int],
                           random_state: int = 42) -> _np.ndarray:
    n = X.shape[0]
    if y is not None:
        y_arr = _np.asarray(y).reshape(-1)
        if y_arr.shape[0] != n:
            raise ValueError(
                f'get_or_make_labels: len(y)={y_arr.shape[0]} != n={n}'
            )
        return y_arr.astype(int)

    if n_clusters is None or n_clusters <= 1:
        max_k = min(10, max(2, n - 1))
        best_k, best_score, best_labels = None, -_math.inf, None
        for k in range(2, max_k + 1):
            km = _KMeans(n_clusters=k, n_init=10, random_state=random_state)
            labels = km.fit_predict(X)
            try:
                score = _silhouette_score(X, labels)
            except Exception:
                score = -_math.inf
            if score > best_score:
                best_k, best_score, best_labels = k, score, labels
        if best_labels is None:
            raise RuntimeError('Failed to auto-select number of clusters.')
        return best_labels.astype(int)

    km = _KMeans(n_clusters=int(n_clusters), n_init=10, random_state=random_state)
    labels = km.fit_predict(X)
    return labels.astype(int)

def sep__json_safe(obj: _AnyT) -> _AnyT:
    if isinstance(obj, float):
        return obj if _math.isfinite(obj) else None
    if isinstance(obj, (_np.floating,)):
        val = float(obj)
        return val if _math.isfinite(val) else None
    if isinstance(obj, dict):
        return {k: sep__json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sep__json_safe(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(sep__json_safe(v) for v in obj)
    if isinstance(obj, (_np.integer,)):
        return int(obj)
    if isinstance(obj, (_np.ndarray,)):
        return sep__json_safe(obj.tolist())
    return obj

def sep_compute_kmeans_ref_scores(X: _np.ndarray,
                                  y: _OptionalT[_np.ndarray],
                                  random_state: int = 42) -> _DictT[str, _AnyT]:
    scores: _DictT[str, _AnyT] = {
        'has_ground_truth': False,
        'true_num_clusters': None,
        'ari': None,
        'nmi': None,
    }
    if y is None:
        return scores

    y_arr = _np.asarray(y)
    if y_arr.ndim > 1:
        y_arr = y_arr.reshape(-1)
    if y_arr.shape[0] != X.shape[0]:
        _warnings.warn(
            f'compute_kmeans_ref_scores: len(y)={y_arr.shape[0]} '
            f'!= n_samples={X.shape[0]}; skipping ARI/NMI.'
        )
        return scores

    uniq, inv = _np.unique(y_arr, return_inverse=True)
    k_true = int(len(uniq))
    if k_true < 2 or X.shape[0] < 2:
        return scores

    km = _KMeans(n_clusters=k_true, n_init=10, random_state=random_state)
    km_labels = km.fit_predict(X)
    ari = _adjusted_rand_score(inv, km_labels)
    nmi = _normalized_mutual_info_score(inv, km_labels)

    scores.update({
        'has_ground_truth': True,
        'true_num_clusters': k_true,
        'ari': float(ari),
        'nmi': float(nmi),
    })
    return scores

def run_separation_on_arrays(
    X: _np.ndarray,
    y: _OptionalT[_np.ndarray] = None,
    n_clusters: _OptionalT[int] = None,
    standardize: bool = True,
    metric: str = 'euclidean',
    noise_label: int = -1,
    R_k: int = 6,
    betas: _TupleT[float, ...] = (0.001, 0.01, 0.05),
    p_margin: int = 3,
    q_margin: float = 0.25,
    k_density: int = 15,
    k_graph: int = 10,
    density_q_grid: _OptionalT[_ListT[float]] = None,
    random_state: int = 0,
    output_prefix: _OptionalT[str] = None,
    write_files: bool = False,
    enable_S1: bool = True,
    enable_S2: bool = True,
    enable_sec_density_connectivity: bool = False,
    enable_S3: bool = True,
	s12_k_base: int = 20,

) -> _TupleT[_pd.DataFrame, _DictT[str, _AnyT]]:
    X = _np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError('X must be 2D array.')
    n = X.shape[0]

    if standardize:
        scaler = StandardScaler()
        X_proc = scaler.fit_transform(X)
    else:
        X_proc = X

    kmeans_scores = sep_compute_kmeans_ref_scores(X_proc, y, random_state=random_state)
    labels = sep_get_or_make_labels(X_proc, y, n_clusters, random_state=random_state)

    result = compute_separation_profile(
        X_proc,
        labels,
        metric=metric,
        noise_label=noise_label,
        R_k=R_k,
        betas=betas,
        p_margin=p_margin,
        q_margin=q_margin,
        k_density=k_density,
        k_graph=k_graph,
        density_q_grid=density_q_grid,
        random_state=random_state,
        enable_S1=enable_S1,
        enable_S2=enable_S2,
        enable_sec_density_connectivity=enable_sec_density_connectivity,
        enable_S3=enable_S3,
	    s12_k_base=int(s12_k_base),

    )
    result['dataset_summary']['kmeans_trueK_external_scores'] = kmeans_scores

    rows = []
    for cid, info in sorted(result['per_cluster'].items(), key=lambda kv: kv[0]):
        row = {'cluster_id': int(cid)}
        row.update({
            k: v for k, v in info.items()
            if k not in ('k_grid', 'S2_details', 'sec_density_connectivity_details')
        })
        rows.append(row)
    df = _pd.DataFrame(rows)
    report_safe = sep__json_safe(result)

    if write_files and output_prefix:
        json_path = f'{output_prefix}_separation_report.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(report_safe, f, indent=2, allow_nan=False)

    return df, report_safe

def sep__print_console_summary(report: _DictT[str, _AnyT], title: str) -> None:
    ds = report['dataset_summary']
    km_scores = ds.get('kmeans_trueK_external_scores', None)
    print(f'\n=== {title} ===')
    print(f"Samples: {ds['n_samples']} | Clusters: {ds['n_clusters']}")
    print('Dataset-level separation (cluster-weighted 10% trimmed means):')

    def _fmt_val(v, text_if_none):
        if v is None:
            return text_if_none
        try:
            return f"{float(v):.6f}"
        except Exception:
            return "nan"

    print(f"  S1 overlap (CHB)                    : {_fmt_val(ds.get('S1_overlap'), '[disabled]')}")
    print(f"  S2 hubness infiltration (CHB)       : {_fmt_val(ds.get('S2_hubness'), '[disabled]')}")
    print(f"  S3 margin, kNN-normalized (CHB)     : {_fmt_val(ds.get('S3_margin'), '[disabled]')}")
    print(f"  sec: density-connectivity AUC       : {_fmt_val(ds.get('sec_density_connectivity_auc'), '[disabled]')}")
    print(f"  sec: margin SVM (RFF)               : {_fmt_val(ds.get('sec_margin_svm'), '[disabled]')}")
    print(f"  sec: margin robust (geom. mean)     : {_fmt_val(ds.get('sec_margin_robust'), '[disabled]')}")

    if km_scores and km_scores.get('has_ground_truth'):
        print('\nKMeans reference (K = #true labels):')
        print(f"  true #clusters (K)            : {km_scores['true_num_clusters']}")
        print(f"  ARI (KMeans vs. truth)        : {km_scores['ari']:.6f}")
        print(f"  NMI (KMeans vs. truth)        : {km_scores['nmi']:.6f}")

    def _fmt(v: _AnyT) -> str:
        try:
            return f"{float(v):.5f}"
        except Exception:
            return "  nan "

    print('\nPer-cluster snapshot:')
    for cid, info in sorted(report['per_cluster'].items(), key=lambda kv: kv[0]):
        s1 = info.get('S1_overlap')
        s2 = info.get('S2_hubness')
        s3 = info.get('S3_margin')
        sec_auc = info.get('sec_density_connectivity_auc')
        sec_rob = info.get('sec_margin_robust')
        print(f"  Cluster {cid:>3} | n={info['size']:>4} | "
              f"S1={_fmt(s1)}  S2={_fmt(s2)}  S3={_fmt(s3)}  "
              f"sec_auc={_fmt(sec_auc)}  sec_robust={_fmt(sec_rob)}")

def sep_main():
    parser = argparse.ArgumentParser(
        description='Compute CHB Separation Profile (S1 overlap, S2 hubness, S3 margin; + secondary metrics).'
    )
    parser.add_argument('--input', type=str, default=None,
                        help='CSV file path (defaults to Iris if omitted).')
    parser.add_argument('--label-col', type=str, default=None,
                        help='Column name containing labels (if present).')
    parser.add_argument('--n-clusters', type=int, default=None,
                        help='If labels absent, use this K for KMeans.')
    parser.add_argument('--no-standardize', action='store_true',
                        help='Disable StandardScaler on features.')
    parser.add_argument('--metric', type=str, default='euclidean',
                        help='Distance metric.')
    parser.add_argument('--noise-label', type=int, default=-1,
                        help='Label value treated as noise.')
    parser.add_argument('--R-k', dest='R_k', type=int, default=6,
                        help='Legacy param (unused in new S1/S2).')
    parser.add_argument('--output-prefix', type=str, default='separation',
                        help='Prefix for output files.')
    parser.add_argument('--no-S1', dest='enable_S1', action='store_false',
                        help='Disable CHB S1 (overlap).')
    parser.add_argument('--no-S2', dest='enable_S2', action='store_false',
                        help='Disable CHB S2 (hubness infiltration).')
    parser.add_argument('--no-S3', dest='enable_S3', action='store_false',
                        help='Disable CHB S3 (normalized kNN margin; also skips secondary SVM/robust margins).')
    parser.add_argument('--sec-density-connectivity', dest='enable_sec_density_connectivity',
                        action='store_true', default=False,
                        help='Enable the secondary density-connectivity metric (off by default).')

    args = parser.parse_args()
    X, y, _ = sep_load_data(args)
    df, report = run_separation_on_arrays(
        X, y=y, n_clusters=args.n_clusters,
        standardize=not args.no_standardize,
        metric=args.metric,
        noise_label=args.noise_label,
        R_k=args.R_k,
        betas=(0.001, 0.01, 0.05),
        p_margin=3,
        q_margin=0.25,
        k_density=15,
        k_graph=10,
        density_q_grid=None,
        random_state=0,
        output_prefix=args.output_prefix,
        write_files=True,
        enable_S1=args.enable_S1,
        enable_S2=args.enable_S2,
        enable_sec_density_connectivity=args.enable_sec_density_connectivity,
        enable_S3=args.enable_S3,
    )
    sep__print_console_summary(report, 'CSV-based run (Separation)')


# =====================================================================
# Density profile
# =====================================================================

from typing import Sequence
from dataclasses import dataclass as _DensityDataclass

def _knn_density_scores_refined(
    X: np.ndarray,
    k_density: int = 15,
    metric: str = "euclidean",
    use_dimension_correction: bool = True,
    use_log: bool = True,
) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    n, d = X.shape
    if n <= 1:
        return np.zeros(n, dtype=float)

    k_eff = min(max(k_density + 1, 2), n)
    nn = NearestNeighbors(n_neighbors=k_eff, metric=metric)
    nn.fit(X)
    dists, _ = nn.kneighbors(X, return_distance=True)
    dists = dists[:, 1:]
    r_bar = np.maximum(dists.mean(axis=1), 1e-12)

    if not use_dimension_correction:
        score = 1.0 / r_bar
    else:
        log_p = -float(d) * np.log(r_bar)
        if use_log:
            score = log_p
        else:
            score = np.exp(log_p - log_p.max())
    score = score - score.min()
    return score

def compute_normalized_density_for_plot_refined(
    X: np.ndarray,
    k_density: int = 15,
    metric: str = "euclidean",
) -> np.ndarray:
    """
    Plot-matching behavior, but safe when k_density > n:

    1) density_i = 1 / mean(kNN distance_i)   (exclude self)
    2) replace zeros/inf -> NaN -> max finite density
    3) clip densities to 99th percentile
    4) normalize to [0, 1] with MinMaxScaler
    """
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import MinMaxScaler
    import numpy as np

    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    if n <= 1:
        return np.zeros(n, dtype=float)

    # IMPORTANT: clamp to avoid NearestNeighbors crash on small n
    k_eff = int(max(2, min(int(k_density), n)))

    nbrs = NearestNeighbors(n_neighbors=k_eff, metric=metric).fit(X)
    distances, _ = nbrs.kneighbors(X)

    avg_distance = distances[:, 1:].mean(axis=1) if k_eff > 1 else distances[:, 0]
    avg_distance[avg_distance == 0] = np.nan

    density = 1.0 / avg_distance
    density[np.isinf(density)] = np.nan
    density = np.nan_to_num(density, nan=np.nanmax(density))

    density = np.clip(density, 0, np.percentile(density, 99))
    normalized_density = MinMaxScaler().fit_transform(density.reshape(-1, 1)).ravel()
    return normalized_density

@_DensityDataclass
class ClusterDensityStats:
    cluster_id: int
    size: int
    mean: float
    q10: float
    q90: float
    spread_low: float
    spread_high: float
    skewness: float
    kurtosis: float
    uniformity: float
    level_ratio: float
    lowtail_ratio: float
    sec_density_composite_badness: float
    sec_density_shape_composite: float
    C1_density_complexity: float  # NEW: combined density complexity score

def density_composite_badness(
    C1_ms_uniformity: Sequence[float],
    D_level_ratio: Sequence[float],
    D_tail_ratio: Sequence[float],
    cluster_sizes: Sequence[int],
    weights_components: Tuple[float, float, float] = (0.4, 0.4, 0.2),
    use_log_hinge: bool = True,
) -> Tuple[np.ndarray, float]:
    u = np.asarray(C1_ms_uniformity, float)
    l = np.asarray(D_level_ratio, float)
    t = np.asarray(D_tail_ratio, float)
    n = np.asarray(cluster_sizes, float)

    total_n = float(n.sum() if n.sum() > 0 else 1.0)
    w = n / total_n

    mu, ml, mt = np.nanmedian(u), np.nanmedian(l), np.nanmedian(t)
    eps = 1e-12

    if use_log_hinge:
        pu = np.maximum(np.log((u + eps) / (mu + eps)), 0.0)
        pl = np.maximum(np.log((l + eps) / (ml + eps)), 0.0)
        pt = np.maximum(np.log((t + eps) / (mt + eps)), 0.0)
    else:
        def rz(x: np.ndarray) -> np.ndarray:
            med = np.nanmedian(x)
            mad = np.nanmedian(np.abs(x - med)) + eps
            z = (x - med) / (1.4826 * mad)
            return np.maximum(z, 0.0)
        pu, pl, pt = rz(u), rz(l), rz(t)

    cap = 3.0
    pu = np.clip(pu, 0.0, cap)
    pl = np.clip(pl, 0.0, cap)
    pt = np.clip(pt, 0.0, cap)

    wu, wl, wt = weights_components
    wvec = np.asarray([wu, wl, wt], float)
    if not np.all(np.isfinite(wvec)) or wvec.sum() <= 0:
        wvec = np.array([1.0, 1.0, 1.0])
    wvec = wvec / (wvec.sum() + eps)
    wu, wl, wt = wvec

    per_cluster = wu * pu + wl * pl + wt * pt
    dataset_score = weighted_trimmed_mean(per_cluster.tolist(), w.tolist(), trim=0.10)
    return per_cluster.astype(float), float(dataset_score)

def _robust_01_badness(values: Sequence[float],
                       direction: str,
                       clip_z: float = 3.0,
                       eps: float = 1e-12) -> np.ndarray:
    v = np.asarray(values, float)
    out = np.full_like(v, np.nan, dtype=float)
    mask = np.isfinite(v)
    if not np.any(mask):
        return out
    vc = v[mask]
    if direction == "lower_is_worse":
        vc = -vc
    med = np.nanmedian(vc)
    mad = np.nanmedian(np.abs(vc - med)) + eps
    z = (vc - med) / (1.4826 * mad)
    z_pos = np.maximum(z, 0.0)
    z_pos = np.clip(z_pos, 0.0, clip_z)
    out[mask] = z_pos / clip_z
    return out

def density_shape_composite(
    means: Sequence[float],
    spread_low: Sequence[float],
    spread_high: Sequence[float],
    skew_vals: Sequence[float],
    kurt_vals: Sequence[float],
    cluster_sizes: Sequence[int],
    weights_components: Tuple[float, float, float, float, float] = (
        0.4, 0.3, 0.1, 0.1, 0.1
    ),
) -> Tuple[np.ndarray, float]:
    means = np.asarray(means, float)
    spread_low = np.asarray(spread_low, float)
    spread_high = np.asarray(spread_high, float)
    skew_vals = np.asarray(skew_vals, float)
    kurt_vals = np.asarray(kurt_vals, float)
    n = np.asarray(cluster_sizes, float)

    total_n = float(n.sum() if n.sum() > 0 else 1.0)
    w = n / total_n

    B_level = _robust_01_badness(means, direction="lower_is_worse")
    B_low   = _robust_01_badness(spread_low, direction="higher_is_worse")
    B_high  = _robust_01_badness(spread_high, direction="higher_is_worse")
    B_skew  = _robust_01_badness(np.abs(skew_vals), direction="higher_is_worse")
    B_kurt  = _robust_01_badness(kurt_vals, direction="higher_is_worse")

    wl, wlow, whigh, wsk, wku = weights_components
    wvec = np.asarray([wl, wlow, whigh, wsk, wku], float)
    if not np.all(np.isfinite(wvec)) or wvec.sum() <= 0:
        wvec = np.array([1, 1, 1, 1, 1], float)
    wvec = wvec / (wvec.sum() + 1e-12)
    wl, wlow, whigh, wsk, wku = wvec

    per_cluster = (
        wl * B_level
        + wlow * B_low
        + whigh * B_high
        + wsk * B_skew
        + wku * B_kurt
    )
    dataset_score = weighted_trimmed_mean(per_cluster.tolist(), w.tolist(), trim=0.10)
    return per_cluster.astype(float), float(dataset_score)


def compute_density_like_plot(X: np.ndarray, k_density: int = 15, metric: str = "euclidean") -> np.ndarray:
    """
    Match plotting script behavior:
    - density = 1 / mean(kNN distance)
    - clip at 99th percentile
    - normalize to [0, 1]
    """
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import MinMaxScaler
    import numpy as np

    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    if n <= 1:
        return np.zeros(n, dtype=float)

    k_eff = min(k_density + 1, n)
    nbrs = NearestNeighbors(n_neighbors=k_eff, metric=metric).fit(X)
    dists, _ = nbrs.kneighbors(X)
    avg_d = dists[:, 1:].mean(axis=1)

    avg_d[avg_d == 0] = np.nan
    dens = 1.0 / avg_d
    dens[np.isinf(dens)] = np.nan
    dens = np.nan_to_num(dens, nan=np.nanmax(dens))

    cap = np.percentile(dens, 99)
    dens = np.clip(dens, 0, cap)
    dens_norm = MinMaxScaler().fit_transform(dens.reshape(-1, 1)).ravel()
    return dens_norm


def evaluate_density_profiles(
    X: np.ndarray,
    labels: np.ndarray,
    k_density: int = 15,
    metric: str = "euclidean",
) -> Tuple[List[ClusterDensityStats], Dict[str, Any]]:
    """
    Compute per-cluster density profile + C1_density_complexity.

    C1_density_complexity is computed over a small internal k-grid (no CLI):
      - For each k in grid:
          spread_k       = Q90(dens_k) - Q10(dens_k)
          between_k      = max(median_global_k - median_cluster_k, 0)
          lowtail_k      = (median_k - Q10_k) / ((Q90_k - Q10_k) + eps)
          complexity_k   = spread_k + between_k + lowtail_k
      - Aggregate per cluster by nanmedian over k.
      - Dataset-level: weighted 10% trimmed mean by cluster size.
    """
    import numpy as np
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import MinMaxScaler
    from scipy.stats import skew, kurtosis

    X = np.asarray(X, float)
    labels = np.asarray(labels, int)
    n, _ = X.shape
    clusters = sorted(np.unique(labels))

    if n <= 1 or len(clusters) == 0:
        stats_list: List[ClusterDensityStats] = []
        dataset_summary = {
            "n_samples": int(n),
            "n_clusters": int(len(clusters)),
            "k_density": int(k_density),
            "sec_density_composite_badness": float("nan"),
            "sec_density_shape_composite": float("nan"),
            "C1_density_complexity": float("nan"),
            "C1_density_complexity_k_grid": [],
        }
        return stats_list, dataset_summary

    # ------------------------------------------------------------
    # Multi-k density precompute with ONE kNN run:
    #   - k here is NearestNeighbors(n_neighbors=k), INCLUDING self.
    # ------------------------------------------------------------
    base_k = int(max(2, min(int(k_density), n)))
    cap = int(min(50, n))

    # deterministic internal grid (no CLI knobs)
    candidates = [
        base_k,
        max(2, base_k // 2),
        min(cap, base_k * 2),
        5, 10, 15, 20, 30
    ]
    k_grid = sorted({int(k) for k in candidates if 2 <= int(k) <= cap})
    if base_k not in k_grid:
        k_grid.append(base_k)
        k_grid = sorted(set(k_grid))

    k_max = int(max(k_grid))
    nbrs = NearestNeighbors(n_neighbors=min(n, k_max + 1), metric=metric).fit(X)
    dists, _ = nbrs.kneighbors(X)  # shape (n, k_max), includes self at col0

    def _dens_norm_from_prefix(k_eff: int) -> np.ndarray:
        # avg over neighbors 1..k_eff-1 (exclude self)
        avg_d = dists[:, 1:k_eff+1].mean(axis=1)
        avg_d[avg_d == 0] = np.nan

        dens = 1.0 / avg_d
        dens[np.isinf(dens)] = np.nan
        dens = np.nan_to_num(dens, nan=np.nanmax(dens))

        dens = np.clip(dens, 0, np.percentile(dens, 99))
        return MinMaxScaler().fit_transform(dens.reshape(-1, 1)).ravel()

    dens_by_k: Dict[int, np.ndarray] = {k: _dens_norm_from_prefix(int(k)) for k in k_grid}

    # baseline dens (for your existing mean/q10/q90/etc outputs)
    dens = dens_by_k[base_k]

    # indices per cluster (reuse for all k)
    cluster_to_idx: Dict[int, np.ndarray] = {cid: np.where(labels == cid)[0] for cid in clusters}

    means: List[float] = []
    medians: List[float] = []
    q10s: List[float] = []
    q90s: List[float] = []
    low_spreads: List[float] = []
    high_spreads: List[float] = []
    skews: List[float] = []
    kurts: List[float] = []
    uniformities: List[float] = []
    sizes: List[int] = []

    eps = 1e-12

    # ---------- per-cluster baseline stats ----------
    for cid in clusters:
        idx = cluster_to_idx[cid]
        vals = dens[idx]
        sizes.append(int(idx.size))

        if vals.size == 0:
            mean = med = q10 = q90 = spread_low = spread_high = sk = ku = uni = float("nan")
        else:
            mean = float(np.mean(vals))
            med = float(np.median(vals))
            q10 = float(np.quantile(vals, 0.10))
            q90 = float(np.quantile(vals, 0.90))
            spread_low = float(mean - q10)
            spread_high = float(q90 - mean)
            sk = float(skew(vals)) if vals.size > 2 else 0.0
            ku = float(kurtosis(vals, fisher=True)) if vals.size > 3 else 0.0
            uni = float(np.std(vals))

        means.append(mean)
        medians.append(med)
        q10s.append(q10)
        q90s.append(q90)
        low_spreads.append(spread_low)
        high_spreads.append(spread_high)
        skews.append(sk)
        kurts.append(ku)
        uniformities.append(uni)

    means_arr = np.asarray(means, float)
    medians_arr = np.asarray(medians, float)
    q10_arr = np.asarray(q10s, float)
    q90_arr = np.asarray(q90s, float)
    low_spreads_arr = np.asarray(low_spreads, float)
    high_spreads_arr = np.asarray(high_spreads, float)
    sizes_arr = np.asarray(sizes, int)

    # Existing ratios (kept)
    mean_med = np.nanmedian(means_arr)
    low_med = np.nanmedian(low_spreads_arr)
    level_ratio = (mean_med + eps) / (means_arr + eps)
    lowtail_ratio = (low_spreads_arr + eps) / (low_med + eps)

    # Existing composites (kept)
    bad_per_cluster, bad_dataset = density_composite_badness(
        uniformities, level_ratio, lowtail_ratio, sizes_arr,
        weights_components=(0.4, 0.4, 0.2),
        use_log_hinge=True,
    )
    shape_per_cluster, shape_dataset = density_shape_composite(
        means_arr, low_spreads_arr, high_spreads_arr,
        skews, kurts, sizes_arr,
    )

    # ------------------------------------------------------------
    # NEW: C1_density_complexity over k-grid (nanmedian aggregation)
    # ------------------------------------------------------------
    C = len(clusters)
    complexity_mat = []

    for k in k_grid:
        dens_k = dens_by_k[int(k)]

        med_k = np.full(C, np.nan, dtype=float)
        spread_k = np.full(C, np.nan, dtype=float)
        lowtail_k = np.full(C, np.nan, dtype=float)

        for i, cid in enumerate(clusters):
            vals = dens_k[cluster_to_idx[cid]]
            if vals.size == 0:
                continue
            med = float(np.median(vals))
            q10 = float(np.quantile(vals, 0.10))
            q90 = float(np.quantile(vals, 0.90))

            med_k[i] = med
            spread_k[i] = q90 - q10

            # low-tail fraction: more mass toward low end => worse
            L = med - q10
            H = q90 - med
            lowtail_k[i] = float(L / (L + H + eps))

        med_global_k = float(np.nanmedian(med_k))
        between_k = np.maximum(med_global_k - med_k, 0.0)

        complexity_mat.append(spread_k + between_k)

    C1_density_complexity_per_cluster = np.nanmedian(np.vstack(complexity_mat), axis=0)

    # Dataset-level: size-weighted trimmed mean (consistent with your other density composites)
    w = sizes_arr.astype(float)
    w = w / (w.sum() if w.sum() > 0 else 1.0)
    C1_density_complexity_dataset = weighted_trimmed_mean(
        C1_density_complexity_per_cluster.tolist(),
        w.tolist(),
        trim=0.10
    )

    # ---------- build per-cluster stats list ----------
    stats_list: List[ClusterDensityStats] = []
    for i, cid in enumerate(clusters):
        stats_list.append(
            ClusterDensityStats(
                cluster_id=int(cid),
                size=int(sizes_arr[i]),
                mean=float(means_arr[i]),
                q10=float(q10_arr[i]),
                q90=float(q90_arr[i]),
                spread_low=float(low_spreads_arr[i]),
                spread_high=float(high_spreads_arr[i]),
                skewness=float(skews[i]),
                kurtosis=float(kurts[i]),
                uniformity=float(uniformities[i]),
                level_ratio=float(level_ratio[i]),
                lowtail_ratio=float(lowtail_ratio[i]),
                sec_density_composite_badness=float(bad_per_cluster[i]),
                sec_density_shape_composite=float(shape_per_cluster[i]),
                C1_density_complexity=float(C1_density_complexity_per_cluster[i]),
            )
        )

    dataset_summary = {
        "n_samples": int(n),
        "n_clusters": int(len(clusters)),
        "k_density": int(k_density),
        "sec_density_composite_badness": float(bad_dataset),
        "sec_density_shape_composite": float(shape_dataset),
        "C1_density_complexity": float(C1_density_complexity_dataset),
        "C1_density_complexity_k_grid": [int(k) for k in k_grid],
    }

    return stats_list, dataset_summary


def run_density_on_arrays(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    n_clusters: Optional[int] = None,
    standardize: bool = True,
    k_density: int = 15,
    metric: str = "euclidean",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    X = np.asarray(X, float)
    if X.ndim != 2:
        raise ValueError("X must be 2D.")

    if standardize:
        scaler = StandardScaler()
        X_proc = scaler.fit_transform(X)
    else:
        X_proc = X

    # NOTE: labels must be the reference labels when y is provided (CHB is
    # label-conditional). A previous version accidentally overwrote them with
    # KMeans labels here, which corrupted C1_density_complexity.
    labels = get_or_make_labels(X_proc, y, n_clusters)

    stats_list, ds_summary = evaluate_density_profiles(
        X_proc, labels, k_density=k_density, metric=metric
    )

    df = pd.DataFrame([s.__dict__ for s in stats_list])
    report = {
        "dataset_summary": ds_summary,
        "per_cluster": [s.__dict__ for s in stats_list],
        "notes": {
            "standardized": standardize,
            "k_density": k_density,
            "metric": metric,
            "densities": "refined kNN density; normalized to [0,1]",
        },
    }
    return df, report



# =====================================================================
# Directional profile  (A1 .. A5)
# =====================================================================

from sklearn.preprocessing import normalize as _l2_normalize

def _ensure_l2(X):
    norms = np.linalg.norm(X, axis=1)
    if np.allclose(norms, 1.0, atol=1e-6):
        return X
    return _l2_normalize(X, norm="l2", axis=1)

# ── A1: Angular Concentration ──────────────────────────────────────
def _angular_concentration(XC, max_pairs_sample=50_000, rng_seed=2025):
    n = XC.shape[0]
    if n <= 1:
        return float("nan")
    XC_n = _ensure_l2(XC)
    num_pairs = n * (n - 1) // 2
    if num_pairs <= max_pairs_sample:
        S = XC_n @ XC_n.T
        iu = np.triu_indices(n, k=1)
        sims = S[iu]
    else:
        rng = np.random.default_rng(rng_seed)
        m = min(max_pairs_sample, num_pairs)
        ii = rng.integers(0, n, size=m)
        jj = rng.integers(0, n, size=m)
        mask = ii != jj
        ii, jj = ii[mask], jj[mask]
        sims = np.sum(XC_n[ii] * XC_n[jj], axis=1)
    if sims.size == 0:
        return float("nan")
    return float(np.mean(sims))

# ── A2: Separation Dimensionality ──────────────────────────────────
def _separation_dimensionality(X, mask_C, max_components=50,
                                max_train=10_000, rng_seed=2025):
    n = X.shape[0]
    n_C = int(mask_C.sum())
    if n_C < 2 or (n - n_C) < 2:
        return float("nan")
    y = mask_C.astype(int)
    rng = np.random.default_rng(rng_seed)
    if n > max_train:
        idx = rng.choice(n, size=max_train, replace=False)
        X_sub, y_sub = X[idx], y[idx]
    else:
        X_sub, y_sub = X, y
    d = X_sub.shape[1]
    n_comp = min(max_components, d, X_sub.shape[0] - 1)
    if n_comp < 1:
        return 1.0
    try:
        pca = PCA(n_components=n_comp, svd_solver="randomized", random_state=rng_seed)
        Z = pca.fit_transform(X_sub)
    except Exception:
        return float("nan")
    m0, m1 = (y_sub == 0), (y_sub == 1)
    if m0.sum() < 2 or m1.sum() < 2:
        return float("nan")
    mu0, mu1 = Z[m0].mean(axis=0), Z[m1].mean(axis=0)
    var0, var1 = Z[m0].var(axis=0) + EPS, Z[m1].var(axis=0) + EPS
    fisher = np.maximum((mu1 - mu0) ** 2 / (var0 + var1), 0.0)
    total = fisher.sum()
    if total <= 0:
        return float(n_comp)
    p = fisher / total
    return float(1.0 / (np.sum(p ** 2) + EPS))

# ── A3: Boundary Linearity ─────────────────────────────────────────
def _boundary_linearity(X, mask_C, k_boundary=15,
                         n_boundary_points=200, local_patch_k=50,
                         rng_seed=2025):
    from sklearn.linear_model import SGDClassifier as _SGD
    n, d = X.shape
    n_C = int(mask_C.sum())
    if n_C < 5 or (n - n_C) < 5:
        return float("nan")
    rng = np.random.default_rng(rng_seed)
    k_eff = min(k_boundary, n - 1)
    nn = NearestNeighbors(n_neighbors=k_eff + 1, metric="euclidean", n_jobs=-1)
    nn.fit(X)
    C_idx = np.where(mask_C)[0]
    _, indices = nn.kneighbors(X[C_idx], return_distance=True)
    nbr_idx = indices[:, 1:]
    y_full = mask_C.astype(int)
    has_foreign = np.any(y_full[nbr_idx] == 0, axis=1)
    boundary_local = np.where(has_foreign)[0]
    if boundary_local.size < 3:
        return float("nan")
    if boundary_local.size > n_boundary_points:
        boundary_local = rng.choice(boundary_local, size=n_boundary_points, replace=False)
    boundary_global = C_idx[boundary_local]
    patch_k = min(local_patch_k, n - 1)
    nn_patch = NearestNeighbors(n_neighbors=patch_k + 1, metric="euclidean", n_jobs=-1)
    nn_patch.fit(X)
    normals = []
    for gi in boundary_global:
        _, pidx = nn_patch.kneighbors(X[gi].reshape(1, -1), return_distance=True)
        pidx = pidx[0, 1:]
        X_p, y_p = X[pidx], y_full[pidx]
        if np.unique(y_p).size < 2:
            continue
        try:
            clf = _SGD(loss="hinge", alpha=1e-3, max_iter=200, tol=1e-3, random_state=rng_seed)
            clf.fit(X_p, y_p)
            w = clf.coef_.ravel()
            nw = np.linalg.norm(w)
            if nw > EPS:
                normals.append(w / nw)
        except Exception:
            continue
    if len(normals) < 3:
        return float("nan")
    N = np.array(normals)
    S = np.abs(N @ N.T)
    m = S.shape[0]
    iu = np.triu_indices(m, k=1)
    return float(np.mean(S[iu]))

# ── A4: Effective Intrinsic Dimensionality ─────────────────────────
def _effective_intrinsic_dim(XC):
    n, d = XC.shape
    if n <= 2 or d < 1:
        return float("nan")
    Xc = XC - XC.mean(axis=0, keepdims=True)
    s = np.linalg.svd(Xc, full_matrices=False, compute_uv=False)
    lam = s ** 2
    lam = lam[lam > EPS]
    if lam.size == 0:
        return float("nan")
    total = lam.sum()
    return float((total ** 2) / (np.sum(lam ** 2) + EPS))

# ── A5: Angular Margin ─────────────────────────────────────────────
def _angular_margin(X, labels, class_id):
    mask_C = (labels == class_id)
    n_C = int(mask_C.sum())
    if n_C < 1:
        return float("nan")
    unique_labels = np.unique(labels)
    foreign_labels = unique_labels[unique_labels != class_id]
    if foreign_labels.size == 0:
        return float("nan")
    X_n = _ensure_l2(X)
    XC = X_n[mask_C]
    fc = []
    for fl in foreign_labels:
        c = X_n[labels == fl].mean(axis=0)
        nm = np.linalg.norm(c)
        if nm > EPS:
            fc.append(c / nm)
    if not fc:
        return float("nan")
    FC = np.array(fc)
    sims = XC @ FC.T
    max_sim = np.clip(np.max(sims, axis=1), -1.0, 1.0)
    ang_dist = np.arccos(max_sim)
    return float(np.quantile(ang_dist, 0.25))

# ── Directional pipeline ──────────────────────────────────────────
def run_directional_on_arrays(
    X, y=None, n_clusters=None, standardize=True,
    output_prefix=None, write_files=False,
):
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be 2D.")
    n = X.shape[0]
    X_proc = StandardScaler().fit_transform(X) if standardize else X.copy()
    labels = get_or_make_labels(X_proc, y, n_clusters)
    unique_classes = np.unique(labels)

    A1L, A2L, A3L, A4L, A5L = [], [], [], [], []
    weights = []
    per_cluster = []

    for cid in unique_classes:
        mask_C = (labels == cid)
        n_C = int(mask_C.sum())
        XC = X_proc[mask_C]
        w = n_C / n

        a1 = _angular_concentration(XC)
        a2 = _separation_dimensionality(X_proc, mask_C)
        a3 = _boundary_linearity(X_proc, mask_C)
        a4 = _effective_intrinsic_dim(XC)
        a5 = _angular_margin(X_proc, labels, int(cid))

        A1L.append(a1); A2L.append(a2); A3L.append(a3)
        A4L.append(a4); A5L.append(a5)
        weights.append(w)
        per_cluster.append({
            "cluster_id": int(cid), "size": n_C,
            "A1_angular_concentration": a1,
            "A2_separation_dimensionality": a2,
            "A3_boundary_linearity": a3,
            "A4_effective_intrinsic_dim": a4,
            "A5_angular_margin": a5,
        })

    report = {
        "dataset_summary": {
            "num_samples": int(n),
            "num_clusters": int(len(unique_classes)),
            "weighted_trimmed_mean": {
                "A1_angular_concentration": weighted_trimmed_mean(A1L, weights),
                "A2_separation_dimensionality": weighted_trimmed_mean(A2L, weights),
                "A3_boundary_linearity": weighted_trimmed_mean(A3L, weights),
                "A4_effective_intrinsic_dim": weighted_trimmed_mean(A4L, weights),
                "A5_angular_margin": weighted_trimmed_mean(A5L, weights),
            },
        },
        "per_cluster": per_cluster,
        "notes": {"standardized": standardize},
    }
    report = _json_safe(report)
    if write_files and output_prefix:
        jp = f"{output_prefix}_directional_report.json"
        with open(jp, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, allow_nan=False)
    return pd.DataFrame(per_cluster), report


# =====================================================================
# CHB primary block: fingerprint h(D), separability gate, topology
# evidence T_evid, and deterministic regime assignment (paper notation)
# =====================================================================

CHB_GATE_TAU1_S1 = 0.5    # local-majority boundary (fails if S1 > tau1)
CHB_GATE_TAU2_S2 = 0.33   # hub-infiltration ceiling (fails if S2 > tau2)
CHB_GATE_TAU3_S3 = 1.0    # unit-margin boundary (fails if S3 < tau3)
CHB_TAU_TOP = 15.0        # 95th percentile of T_evid under Gaussian-blob null


def _chb_pick(d: Dict[str, Any], *keys: str) -> Optional[float]:
    """Return the first finite float among `keys` in dict d (legacy-key aware)."""
    if not isinstance(d, dict):
        return None
    for k in keys:
        v = d.get(k, None)
        if v is None:
            continue
        try:
            v = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(v):
            return v
    return None


def extract_chb_fingerprint(combined: Dict[str, Any]) -> Dict[str, Optional[float]]:
    """
    Assemble the CHB fingerprint h(D) = (S1, S2, S3; C1, C2; T1, T2, T3)
    from a combined report. Legacy key names from earlier code versions are
    accepted as fallbacks, so old reports can still be annotated. NOTE:
    legacy C1 values (key 'DensityComplexity') may stem from a version with
    a label bug and should be recomputed for publication-grade results.
    """
    sep = ((combined.get("separation") or {}).get("dataset_summary") or {})
    coh = (((combined.get("cohesion") or {}).get("dataset_summary") or {})
           .get("weighted_trimmed_mean") or {})
    den = ((combined.get("density") or {}).get("dataset_summary") or {})

    return {
        "S1": _chb_pick(sep, "S1_overlap", "S1_overlap_ms"),
        "S2": _chb_pick(sep, "S2_hubness", "S2_hubness_tail"),
        "S3": _chb_pick(sep, "S3_margin", "S4_margin_kNN"),
        "C1": _chb_pick(den, "C1_density_complexity", "DensityComplexity"),
        "C2": _chb_pick(coh, "C2_elongation", "C6_linear_elongation"),
        "T1": _chb_pick(coh, "T1_ph0_persistence",
                        "C2b_ph0_component_persistence_resampled"),
        "T2": _chb_pick(coh, "T2_ph1_persistence",
                        "C4b_loop_persistence_resampled"),
        "T3": _chb_pick(coh, "T3_ph2_persistence",
                        "C7b_ph2_void_persistence_resampled"),
    }


def chb_separability_gate(
    S1: Optional[float],
    S2: Optional[float],
    S3: Optional[float],
    tau1: float = CHB_GATE_TAU1_S1,
    tau2: float = CHB_GATE_TAU2_S2,
    tau3: float = CHB_GATE_TAU3_S3,
) -> Dict[str, Any]:
    """
    Strict 2-of-3 separability gate, implemented via the exactly equivalent
    median-of-failure-margins rule:
        SEPF = median(S1 - tau1, S2 - tau2, tau3 - S3)
    The gate fails (separability collapse, Regime A) iff SEPF > 0.
    """
    if S1 is None or S2 is None or S3 is None:
        return {"available": False, "SEPF": None, "gate_fails": None,
                "thresholds": {"tau1_S1": tau1, "tau2_S2": tau2, "tau3_S3": tau3}}
    f1 = float(S1) - float(tau1)
    f2 = float(S2) - float(tau2)
    f3 = float(tau3) - float(S3)
    sepf = float(np.median([f1, f2, f3]))
    return {
        "available": True,
        "SEPF": sepf,
        "gate_fails": bool(sepf > 0.0),
        "n_strict_failures": int(f1 > 0) + int(f2 > 0) + int(f3 > 0),
        "failure_margins": {"f1_S1": f1, "f2_S2": f2, "f3_S3": f3},
        "thresholds": {"tau1_S1": tau1, "tau2_S2": tau2, "tau3_S3": tau3},
    }


def chb_topology_evidence(
    T1: Optional[float],
    T2: Optional[float],
    T3: Optional[float],
    tau_top: float = CHB_TAU_TOP,
) -> Dict[str, Any]:
    """T_evid = log(1+T1) + log(1+T2) + log(1+T3), blob-null-calibrated cutoff."""
    if T1 is None or T2 is None or T3 is None:
        return {"available": False, "T_evid": None, "tau_top": float(tau_top)}
    vals = [max(float(v), 0.0) for v in (T1, T2, T3)]
    tev = float(np.log1p(vals[0]) + np.log1p(vals[1]) + np.log1p(vals[2]))
    return {"available": True, "T_evid": tev, "tau_top": float(tau_top),
            "exceeds_blob_null": bool(tev > float(tau_top))}


def chb_assign_regime(gate: Dict[str, Any], tevid: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deterministic regime rule:
      A  if the separability gate fails,
      B  if the gate passes and T_evid > tau_top,
      C  if the gate passes and T_evid <= tau_top.
    """
    if not gate.get("available"):
        return {"regime": None, "status": "undetermined",
                "reason": "separation descriptors (S1, S2, S3) missing"}
    if gate.get("gate_fails"):
        return {"regime": "A", "status": "ok",
                "reason": "separability gate fails (SEPF > 0): separability collapse"}
    if not tevid.get("available"):
        return {"regime": None, "status": "undetermined",
                "reason": "gate passes but topology descriptors (T1, T2, T3) missing"}
    if tevid["T_evid"] > tevid["tau_top"]:
        return {"regime": "B", "status": "ok",
                "reason": "gate passes and T_evid > tau_top: topology mismatch"}
    return {"regime": "C", "status": "ok",
            "reason": "gate passes and T_evid <= tau_top: scale heterogeneity"}


def compute_chb_block(combined: Dict[str, Any]) -> Dict[str, Any]:
    """
    Derive the CHB primary block (fingerprint, gate, T_evid, regime) from a
    combined report. Cheap and idempotent; recomputed on every write.
    """
    fp = extract_chb_fingerprint(combined)
    gate = chb_separability_gate(fp["S1"], fp["S2"], fp["S3"])
    tevid = chb_topology_evidence(fp["T1"], fp["T2"], fp["T3"])
    reg = chb_assign_regime(gate, tevid)
    block = {
        "fingerprint": fp,
        "separability_gate": gate,
        "topology_evidence": tevid,
        "regime": reg.get("regime"),
        "regime_info": reg,
        "orientation": {
            "S1": "higher is harder (cross-label kNN overlap)",
            "S2": "higher is harder (hubness infiltration)",
            "S3": "lower is harder (normalized margin thickness)",
            "C1": "higher is harder (multi-scale density complexity)",
            "C2": "higher is harder (linear elongation)",
            "T1": "higher is harder (PH0 component persistence)",
            "T2": "higher is harder (PH1 loop persistence)",
            "T3": "higher is harder (PH2 void persistence)",
        },
        "notes": {
            "definition": ("h(D) = (S1,S2,S3; C1,C2; T1,T2,T3). Regime rule: A if the "
                           "separability gate fails; B if it passes and T_evid > tau_top; "
                           "C otherwise."),
            "gate": ("strict 2-of-3 failure, implemented as "
                     "SEPF = median(S1 - tau1, S2 - tau2, tau3 - S3) > 0."),
            "tau_top_calibration": ("tau_top = 15.0 is the 95th percentile of T_evid "
                                    "under an isotropic Gaussian-blob null."),
        },
    }
    return _json_safe(block)


def print_chb_summary(chb: Dict[str, Any], title: str = "CHB") -> None:
    fp = chb.get("fingerprint", {}) or {}
    gate = chb.get("separability_gate", {}) or {}
    tev = chb.get("topology_evidence", {}) or {}

    def _f(v):
        try:
            return f"{float(v):.4f}"
        except (TypeError, ValueError):
            return "n/a"

    print(f"\n=== {title}: CHB hardness fingerprint h(D) ===")
    print("  S1={} S2={} S3={} | C1={} C2={} | T1={} T2={} T3={}".format(
        *[_f(fp.get(k)) for k in ("S1", "S2", "S3", "C1", "C2", "T1", "T2", "T3")]))
    if gate.get("available"):
        verdict = "FAIL (separability collapse)" if gate.get("gate_fails") else "pass"
        print(f"  Gate: SEPF={_f(gate.get('SEPF'))} -> {verdict}")
    if tev.get("available"):
        print(f"  T_evid={_f(tev.get('T_evid'))} (tau_top={_f(tev.get('tau_top'))})")
    print(f"  Regime: {chb.get('regime')}")


def chb_annotate_cli(argv=None):
    """Add/refresh the top-level 'chb' block on existing combined report JSONs."""
    parser = argparse.ArgumentParser(
        description="Annotate combined report JSON(s) with the CHB block "
                    "(fingerprint, separability gate, T_evid, regime). "
                    "Understands legacy key names from earlier code versions."
    )
    parser.add_argument("--report", type=str, required=True,
                        help="Path to a *_combined_report.json file or a directory of them.")
    args = parser.parse_args(argv)

    if os.path.isdir(args.report):
        paths = [os.path.join(args.report, f)
                 for f in sorted(os.listdir(args.report))
                 if f.lower().endswith(".json")]
    else:
        paths = [args.report]

    rows = []
    for pth in paths:
        try:
            with open(pth, "r", encoding="utf-8") as f:
                combined = json.load(f)
        except Exception as e:
            print(f"[ERR] {pth}: {e}")
            continue
        combined["chb"] = compute_chb_block(combined)
        with open(pth, "w", encoding="utf-8") as f:
            json.dump(_json_safe(combined), f, indent=2, allow_nan=False)
        name = (combined.get("input", {}) or {}).get("dataset_name") \
            or os.path.splitext(os.path.basename(pth))[0]
        print_chb_summary(combined["chb"], title=str(name))
        rows.append((str(name), combined["chb"].get("regime")))

    if rows:
        print("\n=== Regime overview ===")
        for name, reg in rows:
            print(f"  {name:<32} regime={reg}")
    print(f"\nAnnotated {len(rows)} report(s).")


# =====================================================================
# Combined JSON writer: cohesion + separation + density + caching
# =====================================================================

def both_cli(argv=None):
    """
    CLI: run cohesion + separation + density + directional for a single dataset
    and write one combined JSON including the CHB block (fingerprint, gate,
    T_evid, regime).

    Per-metric caching:
      - if the combined JSON already contains 'cohesion', we skip cohesion;
      - same for 'separation' and 'density' (but 'density' must be new-format
        with C1_density_complexity; legacy 'DensityComplexity' triggers a
        recompute because those values were affected by a label bug);
      - the 'chb' block is always recomputed (cheap, derived).
    """
    from types import SimpleNamespace

    parser = argparse.ArgumentParser(
        description="Run cohesion + separation + density on a single dataset "
                    "and write one combined JSON."
    )
    parser.add_argument("--input", type=str, required=True,
                        help="CSV or NPZ path.")
    parser.add_argument("--label-col", type=str, default=None,
                        help="CSV label column (if applicable).")
    parser.add_argument("--label-path", type=str, default=None,
                        help="For tab/text datasets: label file path.")
    parser.add_argument("--n-clusters", type=int, default=None,
                        help="If labels absent, use this K for KMeans.")
    parser.add_argument("--no-standardize", action="store_true",
                        help="Disable StandardScaler in all metrics.")
    parser.add_argument("--metric", type=str, default="euclidean",
                        help="Distance metric for separation/density.")
    parser.add_argument("--output-prefix", type=str, default="both",
                        help="Prefix (file prefix or directory) for combined JSON.")
    parser.add_argument("--tab", action="store_true",
                        help="Treat --input as plain text with separate label-path.")
    args = parser.parse_args(argv)

    load_args = SimpleNamespace(input=args.input,
                                label_col=args.label_col,
                                label_path=args.label_path)
    if args.input and (args.tab or args.input.lower().endswith(".npz")
                       or args.label_path is not None):
        X, y = load_npz(load_args, max_n=50000, tab=args.tab)
    else:
        X, y, _ = load_data(load_args)

    dataset_name = os.path.splitext(os.path.basename(args.input))[0]

    if os.path.isdir(args.output_prefix):
        os.makedirs(args.output_prefix, exist_ok=True)
        out_path = os.path.join(args.output_prefix,
                                f"{dataset_name}_combined_report.json")
    else:
        out_dir = os.path.dirname(os.path.abspath(args.output_prefix)) or os.getcwd()
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.abspath(f"{args.output_prefix}_combined_report.json")

    if os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as f:
            combined: Dict[str, Any] = json.load(f)
    else:
        combined = {"input": {"path": args.input}}
    # Baseline meta-features (compute once, store in combined JSON)
    if "baseline" not in combined:
        X_base = StandardScaler().fit_transform(X) if (not args.no_standardize) else np.asarray(X, dtype=float)
        combined["baseline"] = {
            "dataset_summary": compute_baseline_metafeatures(
                X_base,
                y=y,
                n_clusters_hint=args.n_clusters,
                rng_seed=2025
            ),
            "notes": {
                "standardized": (not args.no_standardize),
                "indices_labels": "ground-truth if available else KMeans(K=hint or sqrt(n))",
            },
        }

    # Cohesion
    if "cohesion" not in combined:
        print(f"[RUN] {dataset_name} / cohesion")
        _, cohesion_report = run_cohesion_on_arrays(
            X, y,
            n_clusters=args.n_clusters,
            standardize=not args.no_standardize,
            k_fraction=0.10,
            mst_approx_threshold=600,
            geo_max_pairs=10_000,
            output_prefix=None,
            write_files=False,
        )
        combined["cohesion"] = cohesion_report
    else:
        print(f"[SKIP] {dataset_name} / cohesion already present in {out_path}")

    # Separation
    if "separation" not in combined:
        print(f"[RUN] {dataset_name} / separation")
        _, sep_report = run_separation_on_arrays(
            X, y,
            n_clusters=args.n_clusters,
            standardize=not args.no_standardize,
            metric=args.metric,
            noise_label=-1,
            R_k=6,
            betas=(0.001, 0.01, 0.05),
            p_margin=3,
            q_margin=0.25,
            k_density=15,
            k_graph=10,
            density_q_grid=None,
            random_state=0,
            output_prefix=None,
            write_files=False,
            enable_S1=True,
            enable_S2=True,
            enable_sec_density_connectivity=False,
            enable_S3=True,
        )
        combined["separation"] = sep_report
    else:
        print(f"[SKIP] {dataset_name} / separation already present in {out_path}")

    # Density (only skip if it's the *new* density with C1_density_complexity)
    need_density = True
    if "density" in combined:
        if isinstance(combined["density"], dict):
            ds = combined["density"].get("dataset_summary")
            if isinstance(ds, dict) and "C1_density_complexity" in ds:
                need_density = False

    if need_density:
        print(f"[RUN] {dataset_name} / density")
        _, den_report = run_density_on_arrays(
            X, y,
            n_clusters=args.n_clusters,
            standardize=not args.no_standardize,
            k_density=15,
            metric=args.metric,
        )
        combined["density"] = den_report
    else:
        print(f"[SKIP] {dataset_name} / density already present in {out_path}")
    # Directional (A1..A5)
    if "directional" not in combined:
        print(f"[RUN] {dataset_name} / directional")
        _, dir_report = run_directional_on_arrays(
            X, y,
            n_clusters=args.n_clusters,
            standardize=not args.no_standardize,
        )
        combined["directional"] = dir_report
    else:
        print(f"[SKIP] {dataset_name} / directional already present in {out_path}")

    # Hyperparameter stability (compute once)
    # Hyperparameter stability (compute once)  <-- ADD
    # if "hyperparam_stability" not in combined:
    #     combined["hyperparam_stability"] = compute_hyperparam_stability_block(
    # 	    X,
    # 	    y=y,
    # 	    n_clusters=None,
    # 	    standardize=True,
    # 	    metric="euclidean",
    #     )

    combined.setdefault("input", {}).update({
        "path": args.input,
        "label_col": args.label_col,
        "label_path": args.label_path,
        "tab": bool(args.tab),
    })

    # CHB primary block (fingerprint + gate + T_evid + regime); recomputed on every write
    combined["chb"] = compute_chb_block(combined)
    print_chb_summary(combined["chb"], title=dataset_name)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(_json_safe(combined), f, indent=2, allow_nan=False)

    print(f"Saved combined JSON: {out_path}")
    return out_path


# --- batch config + helpers (for kdd_data / kdd_data_org) ---

BASE_DIR = os.path.abspath(os.getenv("CLUSTERING_BASE_DIR", os.getcwd()))
KDD_DATA_ORG_DIR = os.path.join(BASE_DIR, "kdd_data_org")
KDD_DATA_DIR = os.path.join(BASE_DIR, "kdd_data")
OUTPUT_DIR = os.path.join(BASE_DIR, "combined_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def _dataset_name_from_pair(data_path: str) -> str:
    b = os.path.basename(data_path)
    m = re.match(r"^data[_\-](.+?)(\.[^.]+)?$", b, flags=re.IGNORECASE)
    if m:
        return m.group(1)
    return os.path.splitext(b)[0]

def _dataset_name_from_npz(npz_path: str) -> str:
    return os.path.splitext(os.path.basename(npz_path))[0]

def _load_from_text_pair(data_path: str, label_path: str):
    from types import SimpleNamespace
    ns = SimpleNamespace(input=data_path, label_path=label_path)
    X, y = load_npz(ns, max_n=50000, tab=True)
    return X, y

def _load_from_npz(npz_path: str):
    from types import SimpleNamespace
    ns = SimpleNamespace(input=npz_path, label_path=None)
    X, y = load_npz(ns, max_n=50000, tab=False)
    return X, y

def _has_new_density(combined: Dict[str, Any]) -> bool:
    """
    Return True if the combined JSON already contains a density block
    with the new C1_density_complexity dataset-level field.
    """
    den = combined.get("density")
    if not isinstance(den, dict):
        return False
    ds = den.get("dataset_summary")
    if not isinstance(ds, dict):
        return False
    return "C1_density_complexity" in ds


def _has_baseline(combined: Dict[str, Any]) -> bool:
    """
    True if baseline block exists and has at least the core fields we expect.
    """
    b = combined.get("baseline")
    if not isinstance(b, dict):
        return False
    ds = b.get("dataset_summary")
    if not isinstance(ds, dict):
        return False
    # core fields always defined in compute_baseline_metafeatures
    return ("n_samples" in ds) and ("dimensions" in ds)


def _dataset_fully_done(dataset_name: str) -> bool:
    out_path = os.path.join(OUTPUT_DIR, f"{dataset_name}_combined_report.json")
    if not os.path.exists(out_path):
        return False
    try:
        with open(out_path, "r", encoding="utf-8") as f:
            combined = json.load(f)
    except Exception:
        return False

    # Must have all three metric blocks
    if not all(key in combined for key in ("cohesion", "separation", "density", "directional")):
        return False

    # Density must be new-format (with C1_density_complexity)
    if not _has_new_density(combined):
        return False

    # NEW: baseline must exist
    if not _has_baseline(combined):
        return False

    # CHB primary block (fingerprint + gate + regime) must exist
    if "chb" not in combined:
        return False

    return True


def _write_combined_for_arrays(
    X: np.ndarray,
    y: Optional[np.ndarray],
    dataset_name: str,
) -> str:
    out_path = os.path.join(OUTPUT_DIR, f"{dataset_name}_combined_report.json")

    if os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as f:
            combined = json.load(f)
    else:
        combined = {"input": {"dataset_name": dataset_name}}

    # Baseline meta-features (compute once, store in combined JSON)
    if "baseline" not in combined:
        X_base = StandardScaler().fit_transform(X)  # batch uses standardize=True everywhere in your current code
        combined["baseline"] = {
            "dataset_summary": compute_baseline_metafeatures(
                X_base,
                y=y,
                n_clusters_hint=None,
                rng_seed=2025
            ),
            "notes": {
                "standardized": True,
                "indices_labels": "ground-truth if available else KMeans(K=sqrt(n))",
            },
        }



    if "cohesion" not in combined:
        print(f"[RUN] {dataset_name} / cohesion")
        _, coh_report = run_cohesion_on_arrays(
            X, y,
            n_clusters=None,
            standardize=True,
            k_fraction=0.10,
            mst_approx_threshold=600,
            geo_max_pairs=10_000,
            output_prefix=None,
            write_files=False,
        )
        combined["cohesion"] = coh_report
    else:
        print(f"[SKIP] {dataset_name} / cohesion already present")

    if "separation" not in combined:
        print(f"[RUN] {dataset_name} / separation")
        _, sep_report = run_separation_on_arrays(
            X, y,
            n_clusters=None,
            standardize=True,
            metric="euclidean",
            noise_label=-1,
            R_k=6,
            betas=(0.001, 0.01, 0.05),
            p_margin=3,
            q_margin=0.25,
            k_density=15,
            k_graph=10,
            density_q_grid=None,
            random_state=0,
            output_prefix=None,
            write_files=False,
            enable_S1=True,
            enable_S2=True,
            enable_sec_density_connectivity=False,
            enable_S3=True,
        )
        combined["separation"] = sep_report
    else:
        print(f"[SKIP] {dataset_name} / separation already present")

    # Density: recompute if missing or old-format
    need_density = True
    if "density" in combined and _has_new_density(combined):
        need_density = False

    if need_density:
        print(f"[RUN] {dataset_name} / density")
        _, den_report = run_density_on_arrays(
            X, y,
            n_clusters=None,
            standardize=True,
            k_density=15,
            metric="euclidean",
        )
        combined["density"] = den_report
    else:
        print(f"[SKIP] {dataset_name} / density already present")

    # Directional (A1..A5)
    if "directional" not in combined:
        print(f"[RUN] {dataset_name} / directional")
        _, dir_report = run_directional_on_arrays(
            X, y,
            n_clusters=None,
            standardize=True,
        )
        combined["directional"] = dir_report
    else:
        print(f"[SKIP] {dataset_name} / directional already present")

    combined.setdefault("input", {})["dataset_name"] = dataset_name

    # Optional: hyperparameter stability (Appendix I); expensive, off by default.
    # if "hyperparam_stability" not in combined:
    #     combined["hyperparam_stability"] = compute_hyperparam_stability_block(
    #         X, y=y, n_clusters=None, standardize=True, metric="euclidean",
    #     )

    # CHB primary block (fingerprint + gate + T_evid + regime); recomputed on every write
    combined["chb"] = compute_chb_block(combined)
    print_chb_summary(combined["chb"], title=dataset_name)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(_json_safe(combined), f, indent=2, allow_nan=False)

    return out_path

def main_batch_combined():
    """
    Batch:
      - scans kdd_data_org for text data/label pairs,
      - scans kdd_data for .npz datasets,
      - per dataset, runs any missing metric blocks and updates combined JSON.

    If all three metric blocks already exist *and* density is new-format,
    dataset is fully skipped.
    """
    processed = []
    errors = []

    if os.path.isdir(KDD_DATA_ORG_DIR):
        files = {
            f.lower(): os.path.join(KDD_DATA_ORG_DIR, f)
            for f in os.listdir(KDD_DATA_ORG_DIR)
            if os.path.isfile(os.path.join(KDD_DATA_ORG_DIR, f))
        }
        data_files = [p for name, p in files.items() if name.startswith("data_")]

        for data_path in sorted(data_files):
            base = os.path.basename(data_path)
            token = re.sub(r"^data[_\-]", "", os.path.splitext(base)[0],
                           flags=re.IGNORECASE)
            label_candidates = [
                p for name, p in files.items()
                if name.startswith("label_") and token.lower() in name
            ]
            if not label_candidates:
                print(f"[WARN] No label_* found for {data_path}; skipping.")
                continue
            label_path = sorted(label_candidates)[0]
            name = _dataset_name_from_pair(data_path)

            if _dataset_fully_done(name):
                print(f"[SKIP] {name}: all metrics already computed (new-format density).")
                processed.append(name)
                continue

            try:
                X, y = _load_from_text_pair(data_path, label_path)
                out = _write_combined_for_arrays(X, y, name)
                print(f"[OK] {name} -> {out}")
                processed.append(name)
            except Exception as e:
                print(f"[ERR] {name}: {e}")
                errors.append((name, str(e)))
    else:
        print(f"[INFO] kdd_data_org directory not found: {KDD_DATA_ORG_DIR}")

    if os.path.isdir(KDD_DATA_DIR):
        for fname in sorted(os.listdir(KDD_DATA_DIR)):
            if not fname.lower().endswith(".npz"):
                continue
            npz_path = os.path.join(KDD_DATA_DIR, fname)
            name = _dataset_name_from_npz(npz_path)

            if _dataset_fully_done(name):
                print(f"[SKIP] {name}: all metrics already computed (new-format density + baseline).")
                processed.append(name)
                continue

            try:
                X, y = _load_from_npz(npz_path)
                out = _write_combined_for_arrays(X, y, name)
                print(f"[OK] {name} -> {out}")
                processed.append(name)
            except Exception as e:
                print(f"[ERR] {name}: {e}")
                errors.append((name, str(e)))
    else:
        print(f"[INFO] kdd_data directory not found: {KDD_DATA_DIR}")

    print("\n=== Summary ===")
    print(f"Processed: {len(processed)} datasets")
    if errors:
        print("Errors:")
        for n, msg in errors:
            print(" -", n, ":", msg)


# --- Combined top-level entry point ---

def _combined_cli_main():
    import sys
    if len(sys.argv) < 2 or sys.argv[1] in ('-h', '--help'):
        print("""Usage:
  python chb_metrics.py both       [shared-args...]   # full CHB run on one dataset (fingerprint + regime)
  python chb_metrics.py batch                         # run all datasets in configured dirs
  python chb_metrics.py chb        --report PATH      # annotate existing combined JSON(s) with the CHB block
  python chb_metrics.py cohesion   [cohesion-args...] # cohesion/topology block only
  python chb_metrics.py separation [separation-args...] # separation block only
""")
        return
    mode = sys.argv[1].lower()
    rest = sys.argv[2:]
    if mode in ('cohesion', 'coh'):
        sys.argv = [sys.argv[0]] + rest
        return main()
    elif mode in ('separation', 'sep'):
        sys.argv = [sys.argv[0]] + rest
        return sep_main()
    elif mode in ('both', 'both-json'):
        return both_cli(rest)
    elif mode in ('chb', 'annotate'):
        return chb_annotate_cli(rest)
    elif mode in ('batch', 'main_batch'):
        return main_batch_combined()
    else:
        print('Unknown mode:', mode)
        raise SystemExit(2)

if __name__ == "__main__":
    import sys
    if len(sys.argv) == 1:
        main_batch_combined()
    else:
        _combined_cli_main()