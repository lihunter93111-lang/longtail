"""Register a fully replicated candidate before reserved confirmation access."""
import argparse
from datetime import datetime, timezone
from pathlib import Path

import training

base = training.base
HERE = Path(__file__).resolve().parent
ROOT = Path("E:/LJQ_code/LJQ_code_experiment_storage/daytime_v1/20261109_01")


def read(path):
    import json
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--primary-comparison", type=Path, required=True)
    parser.add_argument("--replication-comparison", type=Path)
    parser.add_argument("--use-primary-models", action="store_true")
    parser.add_argument("--allow-single-training-split", action="store_true")
    parser.add_argument("--allow-partial-replication", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--reference-pattern", default="M_{seed}")
    parser.add_argument("--candidate-pattern", default="M_{seed}")
    args = parser.parse_args()
    assert not (ROOT / "final_freeze.json").exists()
    first = read(args.primary_comparison)
    second = read(args.replication_comparison) if args.replication_comparison else None
    assert first["passes"] and len(first["seeds"]) == 5
    if second is None:
        assert args.allow_single_training_split and args.use_primary_models
        assert read(ROOT / "stage3_plan.json")["replication_seeds"] == []
    else:
        assert second["passes"]
        if first.get("recipe") is not None:
            assert second.get("recipe") == first["recipe"]
        if len(second["seeds"]) != 5:
            assert args.allow_partial_replication and len(second["seeds"]) == 3
            assert second["role"] == "three_seed_exploratory_independent_training_split_replication"
    model_evidence = first if args.use_primary_models else second
    assert model_evidence is not None and len(args.seeds) == 5 and model_evidence["seeds"] == args.seeds
    assert Path(model_evidence["reference"]) == args.reference and Path(model_evidence["candidate"]) == args.candidate
    assert model_evidence["reference_pattern"] == args.reference_pattern
    assert model_evidence["candidate_pattern"] == args.candidate_pattern
    models = []
    manifest_hashes = []
    for group, parent, pattern in (("reference", args.reference, args.reference_pattern),
                                   ("candidate", args.candidate, args.candidate_pattern)):
        protocol = read(parent / "protocol.json")
        manifest_hashes.append(protocol["manifest_sha256"])
        for seed in args.seeds:
            folder = parent / pattern.format(seed=seed)
            calibration = read(folder / "calibration.json")
            checkpoint = folder / "best_BCE.pth"
            assert base.sha(checkpoint) == calibration["checkpoint_sha256"]
            models.append({"id": f"{group}_{seed}", "group": group, "seed": seed, "model": protocol.get("model", "dfms_iq"),
                "width": protocol.get("width", 32), "checkpoint": str(checkpoint),
                "checkpoint_sha256": base.sha(checkpoint), "calibration": calibration, "affine": None,
                "training_protocol_path": str(parent / "protocol.json"),
                "training_protocol_sha256": base.sha(parent / "protocol.json"),
                "training_configuration": {key: protocol.get(key) for key in ("epochs", "coverage", "initialization", "batch", "updates", "lr", "weight_decay", "dropout", "clip", "label_smoothing_epsilon", "sampler_sha256", "manifest_sha256")}})
    assert len(set(manifest_hashes)) == 1
    selected = {"name": args.name, "passes_independent_replication": second is not None and len(second["seeds"]) == 5,
        "eligible_for_reserved_confirmation": True, "replication_seed_count": 0 if second is None else len(second["seeds"]),
        "time_bounded_protocol_sha256": base.sha(HERE / "04B_TIME_BOUNDED_REPLICATION_GOAL.txt"),
        "seeds": args.seeds, "models": models,
        "candidate_development_BCE": model_evidence["candidate_BCE"], "manifest_sha256": manifest_hashes[0],
        "primary_comparison": str(args.primary_comparison), "primary_comparison_sha256": base.sha(args.primary_comparison),
        "replication_comparison": str(args.replication_comparison) if second else None,
        "replication_comparison_sha256": base.sha(args.replication_comparison) if second else None,
        "registered_utc": datetime.now(timezone.utc).isoformat(), "source_sha256": base.sha(Path(__file__))}
    path = ROOT / "candidate_registry" / f"{args.name}.json"
    path.parent.mkdir(exist_ok=True)
    base.write(path, selected)
    current = ROOT / "selection.json"
    if current.exists():
        previous = read(current)
        assert previous["manifest_sha256"] == selected["manifest_sha256"]
        if previous["candidate_development_BCE"] <= selected["candidate_development_BCE"]:
            print("CANDIDATE_REGISTERED_EXISTING_SELECTION_RETAINED", flush=True)
            return
    temporary = current.with_suffix(".tmp")
    base.write(temporary, selected)
    temporary.replace(current)
    print("CANDIDATE_SELECTED", args.name, flush=True)


if __name__ == "__main__":
    main()
