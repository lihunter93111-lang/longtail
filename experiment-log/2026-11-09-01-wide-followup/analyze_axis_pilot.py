"""Same-evaluation comparison for the fixed-histogram quota permutation pilot."""
import json
from pathlib import Path

import torch
import numpy as np
from scipy.special import expit
from scipy.stats import t

import training

base = training.base
helper = training.previous
HERE = Path(__file__).resolve().parent
ROOT = Path("E:/LJQ_code/LJQ_code_experiment_storage/daytime_v1/20261109_01")
PREVIOUS = Path("E:/LJQ_code/LJQ_code_experiment_storage/overnight_v1/20261108_01")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    epochs = read(ROOT / "budget_decision.json")["selected_epochs"]
    reference, candidate = ROOT / f"budget_e{epochs}", ROOT / "axis_pilot"
    datasets = {"reference": PREVIOUS / "data", "candidate": ROOT / "axis_data"}
    protocols = [read(path / "protocol.json") for path in (reference, candidate)]
    for key in ("model", "width", "coverage", "initialization", "epochs", "batch", "updates", "lr", "weight_decay", "dropout", "clip"):
        assert protocols[0][key] == protocols[1][key], key
    manifests = [read(path / "manifest.json") for path in datasets.values()]
    for key in ("fit_path", "fit_sha256", "dev_path", "dev_sha256", "early_stop", "calibration", "development"):
        assert manifests[0][key] == manifests[1][key], key
    counts = [np.array(read(path / "fit_A_distribution.json")["counts"]) for path in datasets.values()]
    cardinalities = np.array([code.bit_count() for code in range(1, 64)])
    for k in range(1, 7):
        np.testing.assert_array_equal(np.sort(counts[0][cardinalities == k]), np.sort(counts[1][cardinalities == k]))
    original_tail = read(HERE.parent / "2026-11-06-01-lt-abc/2026-11-06-lt-abc-quotas.json")["tail_codes"]
    new_tail = (np.lexsort((np.arange(1, 64), counts[1]))[:13] + 1).tolist()
    identities = read(datasets["reference"] / "development_identities.json")
    power = np.array([[np.nan if value is None else value for value in row["component_JNR_dB"]] for row in identities])
    other = power.copy()
    other[:, 5] = -np.inf
    weak = np.isfinite(power[:, 5]) & (np.max(np.where(np.isfinite(other), other, -np.inf), axis=1) - power[:, 5] >= 10)
    metrics, losses, matching = {"reference": [], "candidate": []}, {"reference": [], "candidate": []}, []
    label_rows = []
    for seed in (2100, 2101, 2102):
        receipts = [read(path / f"M_{seed}/receipt.json") for path in (reference, candidate)]
        assert receipts[0]["initial_state_sha256"] == receipts[1]["initial_state_sha256"]
        assert receipts[0]["order_sha256"] == receipts[1]["order_sha256"]
        matching.append({"seed": seed, "initial_state_match": True, "position_permutation_match": True})
        truths = []
        for role, parent in (("reference", reference), ("candidate", candidate)):
            folder = parent / f"M_{seed}"
            saved = np.load(folder / "evaluation_logits.npz")
            z, y = saved["logits"], saved["labels"]
            truths.append(y)
            threshold = read(folder / "calibration.json")["thresholds"]["raw"]
            values, labels, _, per_sample = helper.metrics(z, y, threshold, original_tail)
            alternate, _, _, _ = helper.metrics(z, y, threshold, new_tail)
            values.update(new_condition_tail_exact=alternate["tail_exact"], FMB_FPR=labels[0]["FPR"],
                SawFM_FPR=labels[2]["FPR"], ISRJ_weak_FNR=float((expit(z[weak, 5]) < threshold[5]).mean()))
            metrics[role].append(values)
            losses[role].append(per_sample)
            label_rows.extend({"role": role, "seed": seed, **row} for row in labels)
        np.testing.assert_array_equal(*truths)
    difference = np.asarray(losses["reference"]) - np.asarray(losses["candidate"])
    gains = difference.mean(1)
    half = t.ppf(.975, 2) * gains.std(ddof=1) / np.sqrt(3)
    baseline = float(np.mean([row["BCE"] for row in metrics["reference"]]))
    codes = y.astype(int) @ (1 << np.arange(6))
    result = {"role": "three_seed_development_distribution_pilot", "eligible_for_final_confirmation": False,
        "epochs": epochs, "seeds": [2100, 2101, 2102], "reference_BCE": baseline,
        "candidate_BCE": float(baseline - gains.mean()), "relative_BCE_reduction": float(gains.mean() / baseline),
        "paired_gains": gains.tolist(), "seed_t95": [float(gains.mean() - half), float(gains.mean() + half)],
        "identity_bootstrap95": helper.paired_bootstrap(difference.mean(0), codes),
        "metrics": metrics, "per_label": label_rows, "matching": matching, "original_tail_codes": original_tail,
        "new_condition_tail_codes": new_tail, "same_raw_pool_and_evaluation_verified": True,
        "within_cardinality_quota_multisets_identical": True,
        "shared_fit_identities": read(ROOT / "axis_data/preflight.json")["shared_fit_identities"],
        "source_sha256": base.sha(Path(__file__)), "prompt_sha256": base.sha(HERE / "07B_AXIS_PILOT_GOAL.txt")}
    base.write(ROOT / "axis_pilot_review.json", result)
    print("AXIS_PILOT_COMPLETE", result["reference_BCE"], result["candidate_BCE"], result["relative_BCE_reduction"], flush=True)


if __name__ == "__main__":
    main()
