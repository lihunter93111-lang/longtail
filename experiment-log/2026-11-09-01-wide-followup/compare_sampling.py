"""Verify matched sampling controls before computing fixed guardrails."""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.append(str(HERE.parent / "2026-11-08-01-overnight"))
from compare_runs import compare, base


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def matched_compare(reference, candidate, source, seeds):
    protocols = [read(path / "protocol.json") for path in (reference, candidate)]
    for key in ("model", "width", "coverage", "initialization", "epochs", "batch", "updates", "lr", "weight_decay", "dropout", "clip", "manifest_sha256"):
        assert protocols[0][key] == protocols[1][key], key
    matches = []
    for seed in seeds:
        ref = reference / f"M_{seed}"
        receipt = read(ref / "receipt.json") if (ref / "receipt.json").exists() else read(ref / "source_receipt.json")["receipt"]
        alternative = read(candidate / f"M_{seed}/receipt.json")
        for key in ("initial_state_sha256", "epochs", "updates"):
            assert receipt[key] == alternative[key], (seed, key)
        matches.append({"seed": seed, "initial_state_match": True, "epochs_and_updates_match": True})
    result = compare(reference, candidate, source, seeds, "M_{seed}", "M_{seed}")
    result["matching"] = matches
    result["sampling_difference"] = "natural no-replacement versus0.5 cardinality-stratified combo mixture with replacement"
    result["source_sha256"] = base.sha(Path(__file__))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    args = parser.parse_args()
    assert len(args.seeds) == 5
    result = matched_compare(args.reference, args.candidate, args.source, args.seeds)
    base.write(args.output, result)
    print("MATCHED_SAMPLING_COMPARISON", result["passes"], result["relative_BCE_reduction"], flush=True)


if __name__ == "__main__":
    main()
