"""Reference completed same-budget control models with original receipts."""
import argparse
import csv
import importlib.util
import shutil
from pathlib import Path

import torch
import training

base = training.base
helper = training.previous
HERE = Path(__file__).resolve().parent


def read(path):
    import json
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=Path, required=True)
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    parents = [args.budget, args.extension]
    protocols = [read(path / "protocol.json") for path in parents]
    for key in ("model", "width", "coverage", "initialization", "epochs", "batch", "updates", "lr", "weight_decay", "dropout", "clip", "manifest_sha256"):
        assert protocols[0][key] == protocols[1][key], key
    assert protocols[0]["seeds"] == [2100, 2101, 2102]
    assert protocols[1]["seeds"] == [2103, 2104]
    args.output.mkdir(parents=True, exist_ok=True)
    protocol = {key: protocols[0][key] for key in ("model", "width", "coverage", "initialization", "epochs", "batch", "updates", "lr", "weight_decay", "dropout", "clip", "manifest_sha256")}
    protocol.update(operation="reference five completed same-budget controls", seeds=list(range(2100, 2105)),
        source_sha256=base.sha(Path(__file__)), source_protocols={str(path / "protocol.json"): base.sha(path / "protocol.json") for path in parents},
        role="adaptive_development")
    base.write(args.output / "protocol.json", protocol)
    models = {}
    for parent, source_protocol in zip(parents, protocols):
        frozen = read(parent / "freeze.json")
        for seed in source_protocol["seeds"]:
            name = f"M_{seed}"
            source = parent / name
            target = args.output / name
            target.mkdir(exist_ok=True)
            for filename in ("best_BCE.pth", "calibration.json", "calibration_logits.npz", "evaluation_logits.npz",
                             "paired_gap0_logits.npz", "paired_gap5_logits.npz", "paired_gap10_logits.npz", "paired_gap15_logits.npz"):
                shutil.copy2(source / filename, target / filename)
                assert base.sha(source / filename) == base.sha(target / filename)
            base.write(target / "source_receipt.json", {"source": str(source), "receipt": read(source / "receipt.json"),
                "receipt_sha256": base.sha(source / "receipt.json"), "count_as_new_training": False})
            models[name] = frozen["models"][name]
    base.write(args.output / "freeze.json", {"models": models, "protocol_sha256": base.sha(args.output / "protocol.json"),
        "all_models_complete": True, "evaluation_role": "adaptive_development"})
    for path in parents[0].glob("*.csv"):
        if path.name == "summary_metrics.csv" or not (parents[1] / path.name).exists():
            continue
        rows = []
        for parent in parents:
            with (parent / path.name).open(encoding="utf-8-sig") as handle:
                rows.extend(csv.DictReader(handle))
        helper.table(args.output / path.name, rows)
    base.write(args.output / "status.json", {"status": "completed", "models": 5, "new_training_runs": 0})
    print("FIVE_CONTROLS_REFERENCED", flush=True)


if __name__ == "__main__":
    main()
