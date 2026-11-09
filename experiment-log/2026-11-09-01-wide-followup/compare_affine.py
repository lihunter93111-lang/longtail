"""Paired probability-calibration control for the primary sampling experiment."""
import json
from pathlib import Path

import torch
import numpy as np
from scipy.stats import t

import training

ROOT = Path("E:/LJQ_code/LJQ_code_experiment_storage/daytime_v1/20261109_01")
HERE = Path(__file__).resolve().parent
base = training.base
helper = training.previous


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parent_names = {"reference": "sampling_reference_primary", "candidate": "sampling_candidate_primary"}
    losses = {role: [] for role in parent_names}
    for role, name in parent_names.items():
        parent = ROOT / name
        calibration = ROOT / f"{name}_affine"
        assert read(calibration / "protocol.json")["model_freeze_sha256"] == base.sha(parent / "freeze.json")
        settings = read(calibration / "freeze.json")
        for seed in range(2100, 2105):
            saved = np.load(parent / f"M_{seed}/evaluation_logits.npz")
            parameters = settings[f"M_{seed}"]["class_affine"]["parameters"]
            scale = np.array([parameter["scale"] for parameter in parameters])
            bias = np.array([parameter["bias"] for parameter in parameters])
            values = saved["logits"] * scale + bias
            losses[role].append(helper.binary_loss(values, saved["labels"]).mean(1))
            codes = saved["codes"]
    difference = np.asarray(losses["reference"]) - np.asarray(losses["candidate"])
    seed_gain = difference.mean(1)
    half = t.ppf(.975, 4) * seed_gain.std(ddof=1) / np.sqrt(5)
    original = read(ROOT / "sampling_primary_comparison.json")
    result = {"role": "adaptive_development_calibration_control",
        "raw_BCE_gain": original["reference_BCE"] - original["candidate_BCE"],
        "reference_class_affine_BCE": float(np.mean(losses["reference"])),
        "candidate_class_affine_BCE": float(np.mean(losses["candidate"])),
        "class_affine_BCE_gain": float(seed_gain.mean()), "paired_gains": seed_gain.tolist(),
        "seed_t95": [float(seed_gain.mean() - half), float(seed_gain.mean() + half)],
        "identity_bootstrap95": helper.paired_bootstrap(difference.mean(0), codes),
        "source_sha256": base.sha(Path(__file__)), "prompt_sha256": base.sha(HERE / "03B_CALIBRATION_CONTROL_GOAL.txt")}
    base.write(ROOT / "sampling_probability_review.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
