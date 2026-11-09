"""Aggregate head outcomes and verify epoch-zero numerical equivalence."""
import csv
import json
from pathlib import Path

import torch
import numpy as np
from scipy.special import expit

import training

HERE = Path(__file__).resolve().parent
ROOT = Path("E:/LJQ_code/LJQ_code_experiment_storage/daytime_v1/20261109_01")
PREVIOUS = Path("E:/LJQ_code/LJQ_code_experiment_storage/overnight_v1/20261108_01")
base = training.base


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path):
    with path.open(encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def main():
    summaries, equality = [], []
    for split, backbone, seeds in (("primary", "capacity_IF3_w32", range(2080, 2085)),
                                  ("replication", "replication_M32", range(2090, 2095))):
        for prefix in ("wide_heads", "guarded_heads"):
            parent = ROOT / f"{prefix}_{split}"
            records = rows(parent / "metrics.csv")
            for variant in ("hard", "balanced"):
                selected = [r for r in records if r["variant"] == variant]
                summaries.append({"split": split, "method": prefix, "variant": variant,
                    "epochs": [int(r["best_epoch"]) for r in selected],
                    **{key: float(np.mean([float(r[key]) for r in selected])) for key in
                       ("BCE", "macro_AP", "f1_exact", "ECE", "confident_wrong_fraction", "confident_wrong_loss_share")}})
        for seed in seeds:
            original = PREVIOUS / backbone / f"M_{seed}"
            guarded = ROOT / f"guarded_heads_{split}/M_{seed}_hard"
            head = torch.load(guarded / "best_head.pth", map_location="cpu", weights_only=False)
            full = torch.load(original / "best_BCE.pth", map_location="cpu", weights_only=False)
            assert head["epoch"] == 0
            for key, value in head["head_state_dict"].items():
                torch.testing.assert_close(value, full["model_state_dict"]["classifier." + key], atol=0, rtol=0)
            z0 = np.load(original / "evaluation_logits.npz")["logits"]
            z1 = np.load(guarded / "evaluation_logits.npz")["logits"]
            threshold0 = read(original / "calibration.json")["thresholds"]["raw"]
            threshold1 = read(guarded / "calibration.json")["thresholds"]["raw"]
            np.testing.assert_array_equal(expit(z0) >= threshold0, expit(z1) >= threshold1)
            equality.append({"split": split, "seed": seed, "head_tensors_identical": True,
                "all4032_binary_predictions_identical": True, "maximum_logit_difference": float(np.abs(z0 - z1).max())})
    affine = []
    for split in ("primary", "replication"):
        for variant in ("hard", "balanced"):
            source = ROOT / f"wide_heads_{split}_{variant}_affine/metrics.csv"
            selected = [r for r in rows(source) if r["variant"] == "class_affine"]
            affine.append({"split": split, "variant": variant,
                           "postprocessed_BCE": float(np.mean([float(r["BCE"]) for r in selected]))})
    result = {"head_summaries": summaries, "class_affine_summaries": affine, "epoch0_equivalence": equality,
        "interpretation": "Hard guarded heads select the original head in all10 cases. Tiny BCE differences arise from batched feature/head arithmetic and do not constitute gains.",
        "source_sha256": base.sha(Path(__file__))}
    base.write(ROOT / "head_review.json", result)
    print(json.dumps({"summaries": summaries, "affine": affine}, indent=2), flush=True)


if __name__ == "__main__":
    main()
