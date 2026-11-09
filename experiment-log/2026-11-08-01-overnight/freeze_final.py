"""Freeze the independently replicated single-model recipe for confirmation."""
import argparse
from datetime import datetime, timezone
from pathlib import Path

import training
from run_stage2 import read

base = training.base
HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    assert read(args.root / "replication_comparison.json")["passes"]
    destination = args.root / "final_freeze.json"
    assert not destination.exists()
    assert not (args.root / "final_confirmation_data/confirmation_pool").exists()
    records = []
    for group, folder, arm, width in (("reference", "replication_W16", "W", 16),
                                     ("candidate", "replication_M32", "M", 32)):
        parent = args.root / folder
        assert read(parent / "status.json")["status"] == "completed"
        frozen = read(parent / "freeze.json")
        for seed in range(2090, 2095):
            checkpoint = parent / f"{arm}_{seed}/best_BCE.pth"
            calibration = frozen["models"][f"{arm}_{seed}"]
            assert base.sha(checkpoint) == calibration["checkpoint_sha256"]
            records.append({"id": f"{group}_{seed}", "group": group, "seed": seed,
                "model": "dfms_iq", "width": width, "coverage": arm,
                "initialization": "shared F2" if group == "reference" else "scratch",
                "checkpoint": str(checkpoint), "checkpoint_sha256": base.sha(checkpoint),
                "calibration": calibration, "affine": None})
    sources = [HERE / name for name in ("freeze_final.py", "run_confirmation.py", "prepare_final.py",
        "generate_confirmation.m", "generate_confirmation_pairs.m", "06_CONFIRM_GOAL.txt",
        "05C_INDEPENDENT_REPLICATION_GOAL.txt")]
    sources += list((base.REPO / "projects/CJ_Imbalance_0716/cjlt/models").glob("*.py"))
    result = {"all_models_and_calibrations_frozen": True,
        "frozen_utc": datetime.now(timezone.utc).isoformat(), "seeds": list(range(2090, 2095)),
        "models": records, "source_sha256": {str(path): base.sha(path) for path in sources},
        "selection_evidence_sha256": base.sha(args.root / "replication_comparison.json"),
        "primary_metric": "raw unweighted hard BCE", "secondary_probability_metric": "global temperature BCE",
        "decision_rule": "existing independent calibration per-class F1 thresholds",
        "deadline_deviation": "Final freeze and confirmation occurred after the requested 08:00 CST deadline. All full-network training completed before the deadline. Wide-head extension was not run.",
        "confirmation_seed": 3108310800, "confirmation_identities": 4032,
        "post_confirmation_selection": False}
    base.write(destination, result)
    print("FINAL_FREEZE_WRITTEN", str(destination), flush=True)


if __name__ == "__main__":
    main()
