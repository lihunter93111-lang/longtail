"""Conditional second hypothesis after a negative raw-BCE sampling result."""
import argparse
import subprocess
import time
from pathlib import Path

from plan_sampling import ROOT, PREVIOUS, HERE, PYTHON, DEADLINE, append_jobs, job, read, write


def loss_job(name, source, seeds, estimate):
    command = [PYTHON, str(HERE / "run_smoothing.py"), "--source", str(PREVIOUS / source), "--output", str(ROOT / name),
        "--model", "dfms_iq", "--width", "32", "--coverage", "M", "--initialization", "scratch", "--epochs", "40",
        "--seeds", *map(str, seeds), "--hypothesis", "Fixedepsilon0.01 confidence regularization under03C"]
    item = job(name, command, "full", estimate)
    item["prompt"] = str(HERE / "03C_CONFIDENCE_REGULARIZATION_GOAL.txt")
    return item


def comparison_job(name, reference, candidate, source, seeds):
    item = job(name, [PYTHON, str(HERE / "compare_regularization.py"), "--reference", str(reference), "--candidate", str(candidate),
        "--source", str(PREVIOUS / source), "--seeds", *map(str, seeds), "--output", str(ROOT / (name + ".json"))])
    item["prompt"] = str(HERE / "03C_CONFIDENCE_REGULARIZATION_GOAL.txt")
    return item


def plan():
    sampled = read(ROOT / "sampling_primary_comparison.json")
    budget = read(ROOT / "budget_decision.json")
    record = {"epsilon": .01, "epochs": 40, "primary_seeds": list(range(2100, 2105)), "replication_seeds": list(range(2090, 2095))}
    if sampled["checks"]["relative_BCE_reduction_ge2pct"] and sampled["checks"]["positive_seeds_ge4"]:
        record["status"] = "raw_sampling_gain_present_prioritize_operating_point_diagnosis"
    elif budget["selected_epochs"] != 40:
        record["status"] = "fixed40_reference_reuse_condition_not_met"
    else:
        per_run = budget["median_training_seconds"]["40"] * 1.08
        if 10 * per_run + 200 > DEADLINE - time.time():
            record["status"] = "insufficient_time_for_five_seed_two_split_study"
        else:
            additions = [loss_job("smoothed_primary", "data", list(range(2100, 2105)), 5 * per_run + 60),
                comparison_job("smoothing_primary_comparison", ROOT / "sampling_reference_primary", ROOT / "smoothed_primary", "data", list(range(2100, 2105))),
                job("smoothing_advance", [PYTHON, str(HERE / "plan_smoothing.py"), "--phase", "advance"])]
            append_jobs(additions)
            record.update(status="primary_queued", estimated_per_run=per_run,
                reused_replication_reference=str(PREVIOUS / "replication_M32"))
    write(ROOT / "smoothing_plan.json", record)
    print("SMOOTHING_PLAN", record["status"], flush=True)


def advance():
    record = read(ROOT / "smoothing_plan.json")
    comparison = read(ROOT / "smoothing_primary_comparison.json")
    if not comparison["passes"]:
        record.update(status="primary_failed_promotion", checks=comparison["checks"])
    elif 5 * record["estimated_per_run"] + 90 > DEADLINE - time.time():
        record["status"] = "replication_time_insufficient"
    else:
        additions = [loss_job("smoothed_replication", "replication_data", list(range(2090, 2095)), 5 * record["estimated_per_run"] + 60),
            comparison_job("smoothing_replication_comparison", PREVIOUS / "replication_M32", ROOT / "smoothed_replication", "replication_data", list(range(2090, 2095))),
            job("smoothing_register", [PYTHON, str(HERE / "plan_smoothing.py"), "--phase", "register"])]
        append_jobs(additions)
        record["status"] = "independent_replication_queued"
    write(ROOT / "smoothing_plan.json", record)


def register():
    record = read(ROOT / "smoothing_plan.json")
    if not read(ROOT / "smoothing_replication_comparison.json")["passes"]:
        record["status"] = "independent_replication_failed_promotion"
    else:
        command = [PYTHON, str(HERE / "register_candidate.py"), "--name", "label_smoothing_0.01",
            "--reference", str(ROOT / "sampling_reference_primary"), "--candidate", str(ROOT / "smoothed_primary"),
            "--primary-comparison", str(ROOT / "smoothing_primary_comparison.json"),
            "--replication-comparison", str(ROOT / "smoothing_replication_comparison.json"),
            "--use-primary-models", "--seeds", *map(str, range(2100, 2105))]
        assert subprocess.run(command, creationflags=subprocess.CREATE_NO_WINDOW).returncode == 0
        record["status"] = "registered_for_reserved_confirmation"
    write(ROOT / "smoothing_plan.json", record)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["plan", "advance", "register"], required=True)
    args = parser.parse_args()
    {"plan": plan, "advance": advance, "register": register}[args.phase]()
