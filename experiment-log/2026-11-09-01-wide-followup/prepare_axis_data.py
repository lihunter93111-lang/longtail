"""Materialize an audited fixed-quota-histogram label-balanced fit selection."""
import argparse
import json
import shutil
from pathlib import Path

import torch
import numpy as np

import training

base = training.base
helper = training.previous
HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--quotas", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    assert not (args.output / "manifest.json").exists()
    solution = read(args.quotas)
    assert solution["feasible"] and solution["within_cardinality_multisets_identical"]
    original = read(args.source / "manifest.json")
    original_counts = np.array(read(args.source / "fit_A_distribution.json")["counts"])
    counts = np.array(solution["quotas"], dtype=int)
    codes = np.arange(1, 64)
    labels = ((codes[:, None] >> np.arange(6)) & 1).astype(int)
    cardinality = labels.sum(1)
    for k in range(1, 7):
        np.testing.assert_array_equal(np.sort(counts[cardinality == k]), np.sort(original_counts[cardinality == k]))
    pool = base.candidates(args.source, "fit_pool", helper.LABELS)
    rng = np.random.default_rng(20261108)
    selected = []
    for code, count in zip(codes, counts):
        available = np.flatnonzero(pool["codes"] == code)
        assert count <= len(available)
        selected.extend(rng.permutation(available)[:count].tolist())
    assert len(selected) == len(set(selected)) == 8000
    manifest = {**original, "fit_A": selected, "distribution_role": "within-cardinality quota permutation with near-balanced label margins"}
    base.write(args.output / "manifest.json", manifest)
    records = [{"raw_row": int(index), "source_file_hash": pool["sha256"], "sample_seed": int(pool["seeds"][index]),
        "receiver_noise_seed": int(pool["noise"][index]), "component_seeds": pool["components"][index],
        "code": int(pool["codes"][index]), "component_JNR_dB": [None if not np.isfinite(v) else float(v) for v in pool["power"][index]]}
        for index in selected]
    base.write(args.output / "fit_A_identities.json", records)
    for role in ("early_stop", "calibration", "development"):
        shutil.copy2(args.source / f"{role}_identities.json", args.output / f"{role}_identities.json")
    copied = {}
    for path in args.source.glob("diagnostic_*"):
        if path.is_file():
            shutil.copy2(path, args.output / path.name)
            digest = base.sha(args.output / path.name)
            assert digest == base.sha(path)
            copied[path.name] = digest
    y = pool["y"][selected].astype(int)
    np.testing.assert_array_equal(y.sum(0), np.array(solution["positive_margins"]))
    base.write(args.output / "fit_A_distribution.json", {"n": 8000, "unique": 8000, "positive": y.sum(0).tolist(),
        "counts": counts.tolist(), "cardinality_counts": [int(counts[cardinality == k].sum()) for k in range(1, 7)],
        "cooccurrence": (y.T @ y).tolist(), "label_IF": float(y.sum(0).max() / y.sum(0).min())})
    base.write(args.output / "preflight.json", {**read(args.source / "preflight.json"),
        "role": "same audited candidate pool, quota histogram matched label-margin design",
        "source_manifest_sha256": base.sha(args.source / "manifest.json"), "preparation_sha256": base.sha(Path(__file__)),
        "prompt_sha256": base.sha(HERE / "07B_AXIS_PILOT_GOAL.txt"), "quota_solution_sha256": base.sha(args.quotas),
        "shared_fit_identities": len(set(selected) & set(original["fit_A"])), "same_development_roles": True,
        "copied_diagnostic_sha256": copied, "quota_feasibility_independently_verified": True})
    power_rows = []
    for j, label in enumerate(helper.LABELS):
        for name, indices in (("IF3", original["fit_A"]), ("permuted", selected)):
            values = pool["power"][indices, j]
            values = values[np.isfinite(values)]
            power_rows.append({"distribution": name, "label": label, "n": len(values), "mean_JNR": float(values.mean()),
                **{f"bin{low}_{low+5}": int(((values >= low) & (values < low + 5)).sum()) for low in (10, 15, 20, 25)}})
    helper.table(args.output / "power_distribution.csv", power_rows)
    print("AXIS_DATA_READY", y.sum(0).tolist(), flush=True)


if __name__ == "__main__":
    main()
