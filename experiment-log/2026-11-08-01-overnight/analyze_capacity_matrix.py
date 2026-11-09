"""Matched capacity effects and fixed-mixture loss diagnostics."""
import argparse
import json
from pathlib import Path

import torch
import numpy as np
from scipy.special import expit
from scipy.stats import t

import training
from compare_runs import compare

base = training.base
helper = training.previous
SEEDS = list(range(2080, 2085))
HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def summary(values):
    values = np.asarray(values)
    half = t.ppf(.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values))
    return {"values": values.tolist(), "mean": float(values.mean()),
            "seed_t95": [float(values.mean() - half), float(values.mean() + half)]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    sources = {"IF3": root / "data", "balanced": root / "balanced_data"}
    manifests = {name: read(path / "manifest.json") for name, path in sources.items()}
    for key in ("dev_path", "dev_sha256", "development", "calibration", "early_stop"):
        assert manifests["IF3"][key] == manifests["balanced"][key], key
    identity = read(sources["IF3"] / "development_identities.json")
    codes = np.array([row["code"] for row in identity])
    mixtures = {"uniform_combinations": np.r_[0, np.full(63, 1 / 63)]}
    for name, source in sources.items():
        fit = read(source / "fit_A_identities.json")
        mixtures[f"{name}_fit_combinations"] = np.bincount([r["code"] for r in fit], minlength=64) / len(fit)
    counts = np.bincount(codes, minlength=64)
    weights = {name: distribution[codes] / counts[codes] for name, distribution in mixtures.items()}
    for weight in weights.values():
        np.testing.assert_allclose(weight.sum(), 1)
    comparisons, cells, diagnostic_rows, audit_rows = {}, {}, [], []
    raw_losses = {}
    for distribution, source in sources.items():
        comparison = compare(root / f"capacity_{distribution}_w16", root / f"capacity_{distribution}_w32",
                             source, SEEDS, "M_{seed}", "M_{seed}")
        comparisons[distribution] = comparison
        base.write(root / f"capacity_{distribution}_w32/comparison_vs_scratch16.json", comparison)
        for width in (16, 32):
            name = f"{distribution}_w{width}"
            models = root / f"capacity_{name}"
            rows, losses_by_seed = [], []
            for seed in SEEDS:
                folder = models / f"M_{seed}"
                saved = np.load(folder / "evaluation_logits.npz")
                np.testing.assert_array_equal(saved["indices"], manifests[distribution]["development"])
                np.testing.assert_array_equal(saved["codes"], codes)
                z, y = saved["logits"], saved["labels"]
                calibration = read(folder / "calibration.json")
                loss = helper.binary_loss(z, y)
                probability = expit(z)
                prediction = probability >= np.asarray(calibration["thresholds"]["raw"])
                exact = (prediction == y).all(axis=1)
                losses_by_seed.append(loss.mean(axis=1))
                receipt = read(folder / "receipt.json")
                history = read(folder / "history.json")
                rows.append({"seed": seed, "BCE": float(loss.mean()), "exact": float(exact.mean()),
                    "best_BCE_epoch": receipt["best_BCE_epoch"], "best_early_BCE": receipt["best_early_BCE"],
                    "final_fit_BCE": history[-1]["fit_eval_BCE"], "final_early_BCE": history[-1]["early_BCE"]})
                for mixture, weight in weights.items():
                    diagnostic_rows.append({"model": name, "seed": seed, "mixture": mixture, "label": "all",
                        "BCE": float(weight @ loss.mean(axis=1)), "exact": float(weight @ exact),
                        "positive_BCE": float((weight[:, None] * loss * y).sum() / (weight[:, None] * y).sum()),
                        "negative_BCE": float((weight[:, None] * loss * (1 - y)).sum() / (weight[:, None] * (1 - y)).sum())})
                    for j, label in enumerate(helper.LABELS):
                        positive = y[:, j] == 1
                        negative = ~positive
                        diagnostic_rows.append({"model": name, "seed": seed, "mixture": mixture, "label": label,
                            "BCE": float(weight @ loss[:, j]), "exact": float(weight @ (prediction[:, j] == y[:, j])),
                            "positive_BCE": float(weight[positive] @ loss[positive, j] / weight[positive].sum()),
                            "negative_BCE": float(weight[negative] @ loss[negative, j] / weight[negative].sum())})
                confident_error = ((y == 1) & (probability <= .1)) | ((y == 0) & (probability >= .9))
                audit_rows.append({"model": name, "seed": seed,
                    "confident_error_label_fraction": float(confident_error.mean()),
                    "confident_error_loss_share": float(loss[confident_error].sum() / loss.sum()),
                    "positive_loss_share": float((loss * y).sum() / loss.sum())})
            cells[name] = rows
            raw_losses[name] = np.asarray(losses_by_seed)
    matching = []
    for seed in SEEDS:
        receipts = {(d, w): read(root / f"capacity_{d}_w{w}/M_{seed}/receipt.json")
                    for d in sources for w in (16, 32)}
        for d in sources:
            assert receipts[d, 16]["order_sha256"] == receipts[d, 32]["order_sha256"]
        for w in (16, 32):
            assert receipts["IF3", w]["initial_state_sha256"] == receipts["balanced", w]["initial_state_sha256"]
            assert receipts["IF3", w]["order_sha256"] == receipts["balanced", w]["order_sha256"]
        matching.append({"seed": seed, "within_distribution_identity_order_match": True,
                         "within_width_cross_distribution_initial_state_match": True,
                         "cross_distribution_position_order_match": True})
    capacity_if3 = raw_losses["IF3_w16"] - raw_losses["IF3_w32"]
    capacity_balanced = raw_losses["balanced_w16"] - raw_losses["balanced_w32"]
    effects = {"width_gain_IF3": capacity_if3, "width_gain_balanced": capacity_balanced,
        "balanced_gain_w16": raw_losses["IF3_w16"] - raw_losses["balanced_w16"],
        "balanced_gain_w32": raw_losses["IF3_w32"] - raw_losses["balanced_w32"],
        "width_gain_IF3_minus_width_gain_balanced": capacity_if3 - capacity_balanced}
    effect_summary = {name: {**summary(value.mean(axis=1)),
        "identity_bootstrap95": helper.paired_bootstrap(value.mean(axis=0), codes)} for name, value in effects.items()}
    result = {"role": "adaptive_development", "source_sha256": base.sha(Path(__file__)),
        "prompt_sha256": base.sha(HERE / "05D_LOSS_DISTRIBUTION_GOAL.txt"),
        "same_development_roles_verified": True, "matching": matching, "cells": cells,
        "effects": effect_summary, "width_comparisons": comparisons,
        "mixture_weight_effective_sample_size": {name: float(1 / (w @ w)) for name, w in weights.items()},
        "confident_error_analysis": audit_rows,
        "interpretation": "Balanced fit changes labels, combination cooccurrence, identity subset and feasible coverage counts. Reweighted scores are descriptive estimates on the same development identities."}
    base.write(root / "capacity_matrix_review.json", result)
    helper.table(root / "capacity_mixture_diagnostics.csv", diagnostic_rows)
    print(json.dumps(effect_summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
