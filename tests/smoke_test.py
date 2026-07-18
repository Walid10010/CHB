#!/usr/bin/env python3
"""Smoke tests for chb_metrics.py, mirroring the paper's Appendix E.1 checks.

Run selected cases:  python smoke_test.py separated legacy
Run all cases:       python smoke_test.py
Cases: separated | overlapping | elongated | legacy
"""
import json
import os
import sys
import time
import numpy as np

import chb.metrics as chb


def run_case(name, X, y):
    t0 = time.time()
    combined = {"input": {"dataset_name": name}}
    _, combined["cohesion"] = chb.run_cohesion_on_arrays(X, y, standardize=True)
    _, combined["separation"] = chb.run_separation_on_arrays(X, y, standardize=True)
    _, combined["density"] = chb.run_density_on_arrays(X, y, standardize=True)
    block = chb.compute_chb_block(combined)
    chb.print_chb_summary(block, title=name)
    print(f"  [{name}] took {time.time() - t0:.1f}s")
    return block


def make_blobs(low, high, sigma, n_per=400, k=5, d=10, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.uniform(low, high, size=(k, d))
    X = np.vstack([c + sigma * rng.standard_normal((n_per, d)) for c in centers])
    y = np.repeat(np.arange(k), n_per)
    return X, y


def case_separated():
    X, y = make_blobs(-15, 15, sigma=0.5, seed=1)
    return run_case("separated_blobs", X, y)


def case_overlapping():
    X, y = make_blobs(-5, 5, sigma=4.0, seed=2)
    return run_case("overlapping_blobs", X, y)


def case_elongated():
    rng = np.random.default_rng(3)
    parts, labels = [], []
    for i, (cx, cy) in enumerate([(-20, -20), (20, -20), (-20, 20), (20, 20)]):
        z = rng.standard_normal((500, 2)) * np.array([8.0, 0.3])
        parts.append(z + np.array([cx, cy]))
        labels.append(np.full(500, i))
    return run_case("elongated_clusters", np.vstack(parts), np.concatenate(labels))


def case_legacy():
    legacy = {
        "input": {"dataset_name": "legacy_mnist_style"},
        "separation": {"dataset_summary": {
            "S1_overlap_ms": 0.081, "S2_hubness_tail": 0.107, "S4_margin_kNN": 1.096}},
        "cohesion": {"dataset_summary": {"weighted_trimmed_mean": {
            "C6_linear_elongation": 67.236,
            "C2b_ph0_component_persistence_resampled": 12102.879,
            "C4b_loop_persistence_resampled": 179.404,
            "C7b_ph2_void_persistence_resampled": 24.541}}},
        "density": {"dataset_summary": {"DensityComplexity": 0.196}},
    }
    os.makedirs("smoke_out", exist_ok=True)
    path = "smoke_out/legacy_combined_report.json"
    with open(path, "w") as f:
        json.dump(legacy, f)
    chb.chb_annotate_cli(["--report", path])
    with open(path) as f:
        return json.load(f)["chb"]


CASES = {
    "separated": case_separated,
    "overlapping": case_overlapping,
    "elongated": case_elongated,
    "legacy": case_legacy,
}

CHECKS = {
    "separated": [
        ("regime == C", lambda r: r["regime"] == "C"),
        ("gate passes", lambda r: r["separability_gate"]["gate_fails"] is False),
    ],
    "overlapping": [
        ("regime == A", lambda r: r["regime"] == "A"),
        ("gate fails", lambda r: r["separability_gate"]["gate_fails"] is True),
    ],
    "elongated": [
        ("regime == C", lambda r: r["regime"] == "C"),
        ("C2 large (>20)", lambda r: (r["fingerprint"]["C2"] or 0) > 20),
    ],
    "legacy": [
        ("regime == B (MNIST-style)", lambda r: r["regime"] == "B"),
        ("fingerprint complete", lambda r: all(
            r["fingerprint"][k] is not None
            for k in ("S1", "S2", "S3", "C1", "C2", "T1", "T2", "T3"))),
    ],
}


def main():
    selected = sys.argv[1:] or list(CASES)
    ok = True
    results = {}
    for name in selected:
        results[name] = CASES[name]()
    print("\n=== Assertions ===")
    for name in selected:
        for label, cond in CHECKS[name]:
            passed = bool(cond(results[name]))
            print(("PASS  " if passed else "FAIL  ") + f"[{name}] {label}")
            ok = ok and passed
    print("\nALL PASS" if ok else "\nSOME CHECKS FAILED")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
