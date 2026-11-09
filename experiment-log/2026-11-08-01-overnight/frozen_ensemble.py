"""Calibration-only decisions for existing four-member DFMS ensembles."""
import argparse
from pathlib import Path

import torch
import numpy as np
from scipy.special import expit, logit

import training
from run_stage2 import read

base = training.base
helper = training.previous
HERE = Path(__file__).resolve().parent


def aggregate(values, mode):
    if mode == "mean_logit":
        return np.mean(values, axis=0)
    probability = np.mean(expit(values), axis=0)
    return logit(np.clip(probability, 1e-12, 1 - 1e-12))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    assert read(args.models / "protocol.json")["manifest_sha256"] == base.sha(args.source / "manifest.json")
    seeds = list(range(2080, 2085))
    base.write(args.output / "protocol.json", {"primary_members": seeds[:4], "source_sha256": base.sha(Path(__file__)),
        "prompt_sha256": base.sha(HERE / "04D_FROZEN_ENSEMBLE_GOAL.txt"),
        "parent_freeze_sha256": base.sha(args.models / "freeze.json"), "aggregate_parameters": 4 * 257511,
        "independence": "Leave-one-out ensemble variants overlap in three of four members and are descriptive sensitivity analyses.",
        "evaluation_role": "adaptive_development", "calibration_fit_role": "calibration"})
    calibration = [np.load(args.models / f"M_{seed}/calibration_logits.npz") for seed in seeds]
    development = [np.load(args.models / f"M_{seed}/evaluation_logits.npz") for seed in seeds]
    for data in (calibration, development):
        for saved in data[1:]:
            np.testing.assert_array_equal(saved["labels"], data[0]["labels"])
    frozen = {}
    for excluded in seeds:
        members = [seed for seed in seeds if seed != excluded]
        for mode in ("mean_logit", "mean_probability"):
            name = f"exclude{excluded}_{mode}"
            z = aggregate([calibration[seeds.index(seed)]["logits"] for seed in members], mode)
            y = calibration[0]["labels"]
            temperature = helper.fit_temperature(z, y)
            frozen[name] = {"members": members, "mode": mode, "primary": excluded == 2084,
                **temperature, "thresholds": helper.fit_thresholds(z, y)}
    base.write(args.output / "freeze.json", frozen)
    identity = read(args.source / "development_identities.json")
    power = np.array([[np.nan if v is None else v for v in row["component_JNR_dB"]] for row in identity])
    other = power.copy()
    other[:, 5] = -np.inf
    gap = np.max(np.where(np.isfinite(other), other, -np.inf), axis=1) - power[:, 5]
    weak = np.isfinite(power[:, 5]) & (gap >= 10)
    quota = read(HERE.parent / "2026-11-06-01-lt-abc/2026-11-06-lt-abc-quotas.json")
    rows, labels = [], []
    for name, settings in frozen.items():
        raw = aggregate([development[seeds.index(seed)]["logits"] for seed in settings["members"]], settings["mode"])
        y = development[0]["labels"]
        for variant in ("raw", "temperature"):
            z = raw if variant == "raw" else raw / settings["T"]
            threshold = np.asarray(settings["thresholds"])
            if variant == "temperature":
                threshold = expit(logit(threshold) / settings["T"])
            metrics, per_label, _, _ = helper.metrics(z, y, threshold, quota["tail_codes"])
            metrics.update(FMB_FPR=per_label[0]["FPR"], SawFM_FPR=per_label[2]["FPR"],
                ISRJ_weak_FNR=float((expit(z[weak, 5]) < threshold[5]).mean()))
            meta = {"ensemble": name, "primary": settings["primary"], "variant": variant}
            rows.append({**meta, **metrics})
            labels.extend({**meta, **row} for row in per_label)
    helper.table(args.output / "metrics.csv", rows)
    helper.table(args.output / "labels.csv", labels)
    base.write(args.output / "status.json", {"status": "completed", "primary_variants": 2, "sensitivity_variants": 8})
    for row in rows:
        if row["primary"]:
            print(row, flush=True)


if __name__ == "__main__":
    main()
