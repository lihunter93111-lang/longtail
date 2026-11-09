"""Independent metadata-only audit of the preregistered sampling distribution."""
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = Path("E:/LJQ_code/LJQ_code_experiment_storage/daytime_v1/20261109_01")
PREVIOUS = Path("E:/LJQ_code/LJQ_code_experiment_storage/overnight_v1/20261108_01")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    result = {}
    for split, source in (("primary", "data"), ("replication", "replication_data")):
        identities = read(PREVIOUS / source / "fit_A_identities.json")
        manifest = read(PREVIOUS / source / "manifest.json")
        assert [row["raw_row"] for row in identities] == manifest["fit_A"]
        codes = np.array([row["code"] for row in identities])
        labels = (codes[:, None] >> np.arange(6)) & 1
        cardinality = labels.sum(1)
        combination_counts = np.bincount(codes, minlength=64)
        cardinality_counts = np.bincount(cardinality, minlength=7)
        types_per_cardinality = np.bincount([int(c).bit_count() for c in np.flatnonzero(combination_counts)], minlength=7)
        probability = .5 / len(codes) + .5 * cardinality_counts[cardinality] / (
            len(codes) * types_per_cardinality[cardinality] * combination_counts[codes])
        probability /= probability.sum()
        expected_labels = probability @ labels
        expected_cardinality = np.bincount(cardinality, weights=probability, minlength=7)
        np.testing.assert_allclose(expected_cardinality, cardinality_counts / len(codes), atol=1e-12, rtol=1e-12)
        result[split] = {"identities": len(codes), "all63_combinations": bool((combination_counts[1:] > 0).all()),
            "original_label_fraction": labels.mean(0).tolist(), "expected_label_fraction": expected_labels.tolist(),
            "original_label_IF": float(labels.sum(0).max() / labels.sum(0).min()),
            "expected_label_IF": float(expected_labels.max() / expected_labels.min()),
            "expected_cardinality_fraction": expected_cardinality.tolist(),
            "expected_unique_identities_per_epoch": float(np.sum(1 - (1 - probability)**8000)),
            "expected_repetition_fraction": float(np.mean((1 - probability)**8000)),
            "weight_effective_sample_size": float(1 / np.square(probability).sum()),
            "probability_sha256": hashlib.sha256(probability.tobytes()).hexdigest()}
    payload = {"status": "passed", "role": "pretraining_metadata_design_audit", "results": result,
               "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (ROOT / "sampling_design_audit.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
