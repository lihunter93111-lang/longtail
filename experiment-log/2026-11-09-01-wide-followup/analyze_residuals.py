"""Frozen development error and BCE decomposition for six-label models."""
import argparse
import json
from pathlib import Path

import torch
import numpy as np
from scipy.special import expit

import training

base = training.base
helper = training.previous
HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = read(args.models / "protocol.json")
    assert protocol["manifest_sha256"] == base.sha(args.source / "manifest.json")
    identities = read(args.source / "development_identities.json")
    power = np.array([[np.nan if value is None else value for value in row["component_JNR_dB"]] for row in identities])
    gaps = np.full_like(power, np.nan)
    for j in range(6):
        other = power.copy()
        other[:, j] = -np.inf
        gaps[:, j] = np.max(np.where(np.isfinite(other), other, -np.inf), axis=1) - power[:, j]
    factor_rows, loss_rows, run_rows = [], [], []
    for seed in protocol["seeds"]:
        folder = args.models / f"M_{seed}"
        saved = np.load(folder / "evaluation_logits.npz")
        z, y = saved["logits"], saved["labels"].astype(bool)
        threshold = np.array(read(folder / "calibration.json")["thresholds"]["raw"])
        prediction = expit(z) >= threshold
        losses = np.maximum(z, 0) - z * y + np.log1p(np.exp(-np.abs(z)))
        errors = np.empty((len(y), 12), dtype=bool)
        for j in range(6):
            errors[:, 2 * j] = prediction[:, j] & ~y[:, j]
            errors[:, 2 * j + 1] = ~prediction[:, j] & y[:, j]
        error_count = errors.sum(1)
        allocation = errors / np.maximum(error_count[:, None], 1)
        np.testing.assert_allclose(allocation.sum(1), error_count > 0)
        for j, label in enumerate(helper.LABELS):
            weak = y[:, j] & (gaps[:, j] >= 10)
            for direction, index in (("FP", 2 * j), ("FN", 2 * j + 1)):
                factor_rows.append({"seed": seed, "factor": f"{label}_{direction}",
                    "identity_error_fraction": float(errors[:, index].mean()),
                    "Shapley_exact_loss_allocation": float(allocation[:, index].mean()),
                    "singleton_oracle_exact_gain": float((errors[:, index] & (error_count == 1)).mean())})
            loss_rows.append({"seed": seed, "label": label,
                "positive_BCE_contribution": float((losses[:, j] * y[:, j]).mean() / 6),
                "negative_BCE_contribution": float((losses[:, j] * ~y[:, j]).mean() / 6),
                "positive_BCE": float(losses[y[:, j], j].mean()), "negative_BCE": float(losses[~y[:, j], j].mean()),
                "weak_positive_identities": int(weak.sum()), "weak_FNR": float((~prediction[weak, j]).mean()),
                "weak_BCE": float(losses[weak, j].mean())})
        selected = [0, 5, 9, 11]
        oracle = (errors[:, selected].sum(1) == error_count) & (error_count > 0)
        run_rows.append({"seed": seed, "raw_BCE": float(losses.mean()), "exact_loss": float((error_count > 0).mean()),
            "empty_prediction_fraction": float((prediction.sum(1) == 0).mean()),
            "cardinality_underprediction": float((prediction.sum(1) < y.sum(1)).mean()),
            "cardinality_overprediction": float((prediction.sum(1) > y.sum(1)).mean()),
            "wrong_with_correct_cardinality": float(((prediction.sum(1) == y.sum(1)) & (error_count > 0)).mean()),
            "FMB_FP_SawFM_FN_MFT_FN_ISRJ_FN_joint_oracle_gain": float(oracle.mean()),
            "wrong_label_count_distribution": np.bincount(error_count, minlength=7).tolist()})
    summary = []
    for factor in sorted({row["factor"] for row in factor_rows}):
        chosen = [row for row in factor_rows if row["factor"] == factor]
        summary.append({"factor": factor, **{key: float(np.mean([row[key] for row in chosen])) for key in
            ("identity_error_fraction", "Shapley_exact_loss_allocation", "singleton_oracle_exact_gain")}})
    summary.sort(key=lambda row: row["Shapley_exact_loss_allocation"], reverse=True)
    result = {"role": "frozen_adaptive_development_diagnostic", "source_sha256": base.sha(Path(__file__)),
        "model_freeze_sha256": base.sha(args.models / "freeze.json"), "prompt_sha256": base.sha(HERE / "05A_RESIDUAL_ERROR_GOAL.txt"),
        "factors": factor_rows, "losses": loss_rows, "runs": run_rows, "factor_summary": summary,
        "oracle_scope": "Oracle diagnostic under perfect correction of selected factors. Achievability requires a trained method experiment."}
    base.write(args.output, result)
    print(json.dumps({"top_factors": summary[:6], "run_means": {key: float(np.mean([r[key] for r in run_rows])) for key in
        ("raw_BCE", "exact_loss", "empty_prediction_fraction", "cardinality_underprediction", "cardinality_overprediction", "FMB_FP_SawFM_FN_MFT_FN_ISRJ_FN_joint_oracle_gain")}}, indent=2), flush=True)


if __name__ == "__main__":
    main()
