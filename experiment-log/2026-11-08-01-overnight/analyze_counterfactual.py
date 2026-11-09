"""Paired identity uncertainty for frozen target-removal diagnostics."""
import argparse
import csv
import json
from pathlib import Path

import torch
import numpy as np
from scipy.special import expit
from scipy.stats import t

import training

base = training.base


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def interval(values):
    values = np.asarray(values, dtype=float)
    half = t.ppf(.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values))
    return [float(values.mean() - half), float(values.mean() + half)]


def samples(models, absent, arm, seed, gap):
    positive = np.load(models / f"{arm}_{seed}" / f"paired_gap{gap}_logits.npz")
    negative = np.load(absent / f"{arm}_{seed}_absent_logits.npz")
    lookup = np.searchsorted(negative["positions"], positive["positions"])
    np.testing.assert_array_equal(negative["positions"][lookup], positive["positions"])
    targets = positive["targets"]
    np.testing.assert_array_equal(negative["targets"][lookup], targets)
    pos = positive["logits"][np.arange(len(targets)), targets]
    neg = negative["logits"][lookup, targets]
    thresholds = np.asarray(read(models / f"{arm}_{seed}" / "calibration.json")["thresholds"]["raw"])[targets]
    values = np.column_stack((expit(pos) >= thresholds, expit(neg) >= thresholds,
                              pos > neg, pos - neg, np.logaddexp(0, -pos),
                              np.logaddexp(0, neg), pos >= 0, neg >= 0))
    return values, positive["positions"], targets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    conditions = {
        "W": (root / "stage2", root / "stage5_counterfactual_main", "W"),
        "M": (root / "stage2", root / "stage5_counterfactual_main", "M"),
        "scratch16": (root / "capacity_IF3_w16", root / "stage5_counterfactual_scratch16", "M"),
        "scratch32": (root / "capacity_IF3_w32", root / "stage5_counterfactual_scratch32", "M"),
    }
    columns = ["TPR", "matched_absent_FPR", "paired_ranking", "mean_paired_logit_difference",
               "positive_BCE", "absent_BCE", "fixed05_TPR", "fixed05_absent_FPR"]
    results = []
    for reference, candidate in (("W", "M"), ("scratch16", "scratch32"), ("W", "scratch32")):
        for gap, label in [(g, None) for g in (0, 5, 10, 15)] + [(15, c) for c in (2, 4, 5)]:
            series = {reference: [], candidate: []}
            for seed in range(2080, 2085):
                identity = []
                for name in (reference, candidate):
                    values, positions, targets = samples(*conditions[name], seed, gap)
                    mask = np.ones(len(targets), dtype=bool) if label is None else targets == label
                    series[name].append(values[mask])
                    identity.append(positions[mask])
                np.testing.assert_array_equal(*identity)
            ref, cand = np.asarray(series[reference]), np.asarray(series[candidate])
            per_identity = (cand - ref).mean(axis=0)
            rng = np.random.default_rng(2026110901)
            bootstraps = []
            for _ in range(100):
                indices = rng.integers(0, len(per_identity), size=(20, len(per_identity)))
                bootstraps.append(per_identity[indices].mean(axis=1))
            bounds = np.quantile(np.concatenate(bootstraps), [.025, .975], axis=0)
            for index, metric in enumerate(columns):
                delta = (cand - ref).mean(axis=1)[:, index]
                results.append({"reference": reference, "candidate": candidate, "gap": gap,
                    "label": "all_targets" if label is None else training.previous.LABELS[label],
                    "metric": metric, "identities": len(per_identity),
                    "reference_mean": float(ref[:, :, index].mean()),
                    "candidate_mean": float(cand[:, :, index].mean()),
                    "candidate_minus_reference": float(delta.mean()),
                    "seed_t95": interval(delta), "identity_bootstrap95": bounds[:, index].tolist()})
    output = {"role": "adaptive_development_counterfactual", "source_sha256": base.sha(Path(__file__)),
        "bootstrap": "2000 ordinary identity-cluster resamples. Seeds and both physical views stay paired. Seed t intervals are separate and conditional on the same data split.",
        "paired_ranking_scope": "requires two controlled observations of the same identity", "results": results}
    base.write(root / "counterfactual_comparison.json", output)
    for row in results:
        if row["gap"] == 15 and row["label"] == "all_targets" and row["metric"] in ("TPR", "matched_absent_FPR", "paired_ranking", "positive_BCE"):
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
