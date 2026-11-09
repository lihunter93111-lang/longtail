"""Explain head changes through calibration-only affine logit diagnostics."""
import json
from pathlib import Path

import torch
import numpy as np

import training

ROOT = Path("E:/LJQ_code/LJQ_code_experiment_storage/daytime_v1/20261109_01")
PREVIOUS = Path("E:/LJQ_code/LJQ_code_experiment_storage/overnight_v1/20261108_01")
base = training.base


def main():
    rows = []
    for split, parent, seeds in (("primary", "capacity_IF3_w32", range(2080, 2085)),
                                 ("replication", "replication_M32", range(2090, 2095))):
        for seed in seeds:
            original = PREVIOUS / parent / f"M_{seed}"
            saved = np.load(original / "calibration_logits.npz")
            old_weight = torch.load(original / "best_BCE.pth", map_location="cpu", weights_only=False)["model_state_dict"]["classifier.1.weight"].numpy().astype(float)
            for variant in ("hard", "balanced"):
                folder = ROOT / f"wide_heads_{split}/M_{seed}_{variant}"
                adapted = np.load(folder / "calibration_logits.npz")
                np.testing.assert_array_equal(saved["labels"], adapted["labels"])
                new_weight = torch.load(folder / "best_head.pth", map_location="cpu", weights_only=False)["head_state_dict"]["1.weight"].numpy().astype(float)
                for j, label in enumerate(training.previous.LABELS):
                    x, y = saved["logits"][:, j], adapted["logits"][:, j]
                    design = np.column_stack((x, np.ones(len(x))))
                    coefficient = np.linalg.lstsq(design, y, rcond=None)[0]
                    residual = y - design @ coefficient
                    rows.append({"split": split, "seed": seed, "variant": variant, "label": label,
                        "logit_scale": float(coefficient[0]), "logit_bias": float(coefficient[1]),
                        "affine_R2": float(1 - np.square(residual).sum() / np.square(y - y.mean()).sum()),
                        "weight_cosine": float(old_weight[j] @ new_weight[j] / (np.linalg.norm(old_weight[j]) * np.linalg.norm(new_weight[j]))),
                        "weight_norm_ratio": float(np.linalg.norm(new_weight[j]) / np.linalg.norm(old_weight[j]))})
    summary = {key: [float(min(row[key] for row in rows)), float(max(row[key] for row in rows))]
               for key in ("logit_scale", "affine_R2", "weight_cosine", "weight_norm_ratio")}
    base.write(ROOT / "head_logit_geometry.json", {"role": "calibration_only_diagnostic", "rows": rows,
        "ranges": summary, "source_sha256": base.sha(Path(__file__))})
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
