"""Advance matched sampling experiments using completed evidence and measured time."""
import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path("E:/LJQ_code/LJQ_code_experiment_storage/daytime_v1/20261109_01")
PREVIOUS = Path("E:/LJQ_code/LJQ_code_experiment_storage/overnight_v1/20261108_01")
PYTHON = "D:/anaconda3/envs/PY10/python.exe"
DEADLINE = datetime(2026, 11, 9, 8, 0, tzinfo=timezone.utc).timestamp()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def job(name, command, kind="analysis", estimate=0):
    return {"name": name, "command": command, "kind": kind, "estimated_seconds": estimate,
            "output": str(ROOT / name), "prompt": str(HERE / "03A_COMBINATION_SAMPLING_GOAL.txt")}


def append_jobs(additions):
    path = ROOT / "jobs.json"
    jobs = read(path)
    assert not set(item["name"] for item in additions) & set(item["name"] for item in jobs)
    proposed = jobs + additions
    full_runs = 0
    for item in proposed:
        if item["kind"] == "full":
            command = item["command"]
            start = command.index("--seeds") + 1
            full_runs += next((i for i, value in enumerate(command[start:]) if value.startswith("--")), len(command) - start)
    assert full_runs <= 30, full_runs
    write(path, proposed)


def train_command(name, source, epochs, seeds, sampling):
    script = "run_sampling.py" if sampling else "run_variant.py"
    return [PYTHON, str(HERE / script), "--source", str(PREVIOUS / source), "--output", str(ROOT / name),
        "--model", "dfms_iq", "--width", "32", "--coverage", "M", "--initialization", "scratch",
        "--epochs", str(epochs), "--seeds", *map(str, seeds), "--hypothesis",
        "Matched cardinality-stratified combination sampling under03A" if sampling else "Natural no-replacement matched control under03A"]


def plan():
    decision = read(ROOT / "budget_decision.json")
    epochs = decision["selected_epochs"]
    per_run = decision["median_training_seconds"][str(epochs)] * 1.08
    remaining = DEADLINE - time.time()
    record = {"epochs": epochs, "estimated_training_seconds_per_run": per_run,
        "planning_utc": datetime.now(timezone.utc).isoformat(), "remaining_seconds": remaining,
        "primary_seeds": list(range(2100, 2105)), "reference": str(ROOT / "sampling_reference_primary"),
        "candidate": str(ROOT / "sampling_candidate_primary"), "budget_decision": str(ROOT / "budget_decision.json")}
    if 7 * per_run + 180 > remaining:
        record["status"] = "insufficient_time_for_complete_five_seed_primary"
        write(ROOT / "stage3_plan.json", record)
        return
    record["status"] = "primary_queued"
    write(ROOT / "stage3_plan.json", record)
    extension = "sampling_control_extension"
    candidate = "sampling_candidate_primary"
    additions = [
        job(extension, train_command(extension, "data", epochs, [2103, 2104], False), "full", 2 * per_run + 60),
        job("sampling_reference_primary", [PYTHON, str(HERE / "merge_controls.py"), "--budget", str(ROOT / f"budget_e{epochs}"),
            "--extension", str(ROOT / extension), "--output", str(ROOT / "sampling_reference_primary")]),
        job(candidate, train_command(candidate, "data", epochs, list(range(2100, 2105)), True), "full", 5 * per_run + 60),
        job("sampling_primary_comparison", [PYTHON, str(HERE / "compare_sampling.py"), "--reference", record["reference"],
            "--candidate", record["candidate"], "--source", str(PREVIOUS / "data"), "--seeds", *map(str, range(2100, 2105)),
            "--output", str(ROOT / "sampling_primary_comparison.json")]),
        job("sampling_advance", [PYTHON, str(HERE / "plan_sampling.py"), "--phase", "advance"])]
    calibration_jobs = []
    for name in ("sampling_reference_primary", "sampling_candidate_primary"):
        item = job(name + "_affine", [PYTHON, str(HERE.parent / "2026-11-08-01-overnight/affine_calibration.py"),
            "--models", str(ROOT / name), "--output", str(ROOT / (name + "_affine"))], "calibration")
        item["prompt"] = str(HERE / "03B_CALIBRATION_CONTROL_GOAL.txt")
        calibration_jobs.append(item)
    calibration_jobs.append(job("sampling_probability_review", [PYTHON, str(HERE / "compare_affine.py")]))
    additions[-1:-1] = calibration_jobs
    append_jobs(additions)
    print("SAMPLING_PRIMARY_QUEUED", epochs, per_run, flush=True)


def advance():
    record = read(ROOT / "stage3_plan.json")
    comparison = read(ROOT / "sampling_primary_comparison.json")
    if not comparison["passes"]:
        record.update(status="primary_failed_promotion", primary_checks=comparison["checks"])
        write(ROOT / "stage3_plan.json", record)
        print("SAMPLING_PRIMARY_FAILED", flush=True)
        append_jobs([job("axis_pilot_planning", [PYTHON, str(HERE / "queue_axis_pilot.py")]),
                     job("smoothing_planning", [PYTHON, str(HERE / "plan_smoothing.py"), "--phase", "plan"])])
        return
    remaining = DEADLINE - time.time()
    per_run = record["estimated_training_seconds_per_run"]
    seed_count = 5 if 10 * per_run + 200 <= remaining else 3 if 6 * per_run + 200 <= remaining else 0
    seeds = list(range(2120, 2120 + seed_count))
    record.update(status="replication_queued" if seeds else "single_training_split_confirmation_eligible",
                  replication_seeds=seeds, replication_decision_utc=datetime.now(timezone.utc).isoformat(),
                  replication_remaining_seconds=remaining)
    write(ROOT / "stage3_plan.json", record)
    if not seeds:
        append_jobs([job("sampling_register", [PYTHON, str(HERE / "plan_sampling.py"), "--phase", "register"])])
        return
    reference, candidate = "sampling_reference_replication", "sampling_candidate_replication"
    compare_script = "compare_sampling.py" if seed_count == 5 else "compare_partial_replication.py"
    command = [PYTHON, str(HERE / compare_script), "--reference", str(ROOT / reference), "--candidate", str(ROOT / candidate),
        "--source", str(PREVIOUS / "replication_data"), "--output", str(ROOT / "sampling_replication_comparison.json")]
    if seed_count == 5:
        command += ["--seeds", *map(str, seeds)]
    additions = [job(reference, train_command(reference, "replication_data", record["epochs"], seeds, False), "full", seed_count * per_run + 60),
        job(candidate, train_command(candidate, "replication_data", record["epochs"], seeds, True), "full", seed_count * per_run + 60),
        job("sampling_replication_comparison", command),
        job("sampling_register", [PYTHON, str(HERE / "plan_sampling.py"), "--phase", "register"])]
    for item in additions:
        item["prompt"] = str(HERE / "04B_TIME_BOUNDED_REPLICATION_GOAL.txt")
    append_jobs(additions)
    print("SAMPLING_REPLICATION_QUEUED", seed_count, flush=True)


def register():
    record = read(ROOT / "stage3_plan.json")
    seeds = record["replication_seeds"]
    if seeds and not read(ROOT / "sampling_replication_comparison.json")["passes"]:
        record["status"] = "independent_replication_failed_promotion"
        write(ROOT / "stage3_plan.json", record)
        print("REPLICATION_NEGATIVE_RETAINED", flush=True)
        append_jobs([job("axis_pilot_planning", [PYTHON, str(HERE / "queue_axis_pilot.py")])])
        return
    command = [PYTHON, str(HERE / "register_candidate.py"), "--name", "cardinality_combo_mix", "--reference", record["reference"],
        "--candidate", record["candidate"], "--primary-comparison", str(ROOT / "sampling_primary_comparison.json"),
        "--use-primary-models", "--seeds", *map(str, record["primary_seeds"])]
    if seeds:
        command += ["--replication-comparison", str(ROOT / "sampling_replication_comparison.json")]
        if len(seeds) == 3:
            command.append("--allow-partial-replication")
    else:
        command.append("--allow-single-training-split")
    result = subprocess.run(command, creationflags=subprocess.CREATE_NO_WINDOW)
    assert result.returncode == 0
    record["status"] = "registered_for_reserved_confirmation"
    write(ROOT / "stage3_plan.json", record)
    append_jobs([job("axis_pilot_planning", [PYTHON, str(HERE / "queue_axis_pilot.py")])])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["plan", "advance", "register", "enqueue"], required=True)
    args = parser.parse_args()
    if args.phase == "enqueue":
        append_jobs([job("sampling_plan", [PYTHON, str(HERE / "plan_sampling.py"), "--phase", "plan"])])
    elif args.phase == "plan":
        plan()
    elif args.phase == "advance":
        advance()
    else:
        register()


if __name__ == "__main__":
    main()
