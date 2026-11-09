"""Use remaining time and run budget for the declared three-seed data-axis pilot."""
import json
import time
from pathlib import Path

from plan_sampling import append_jobs, job, DEADLINE, PYTHON

HERE = Path(__file__).resolve().parent
ROOT = Path("E:/LJQ_code/LJQ_code_experiment_storage/daytime_v1/20261109_01")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    status = ROOT / "axis_pilot_plan.json"
    jobs = read(ROOT / "jobs.json")
    if any(item["name"] == "axis_pilot" for item in jobs):
        return
    decision = read(ROOT / "budget_decision.json")
    epochs = decision["selected_epochs"]
    estimate = 3 * decision["median_training_seconds"][str(epochs)] * 1.08 + 90
    remaining = DEADLINE - time.time()
    record = {"epochs": epochs, "estimated_seconds": estimate, "remaining_seconds": remaining,
              "role": "three_seed_development_distribution_pilot"}
    if not (ROOT / "axis_pipeline_status.json").exists() or read(ROOT / "axis_pipeline_status.json")["stage"] != "completed":
        record["status"] = "data_preparation_incomplete"
    elif estimate > remaining:
        record["status"] = "insufficient_time"
    else:
        command = [PYTHON, str(HERE / "run_variant.py"), "--source", str(ROOT / "axis_data"), "--output", str(ROOT / "axis_pilot"),
            "--model", "dfms_iq", "--width", "32", "--coverage", "M", "--initialization", "scratch", "--epochs", str(epochs),
            "--seeds", "2100", "2101", "2102", "--hypothesis", "Quota-histogram and cardinality matched label-margin pilot under07B"]
        additions = [job("axis_pilot", command, "full", estimate),
                     job("axis_pilot_review", [PYTHON, str(HERE / "analyze_axis_pilot.py")])]
        for item in additions:
            item["prompt"] = str(HERE / "07B_AXIS_PILOT_GOAL.txt")
        try:
            append_jobs(additions)
            record["status"] = "queued"
        except AssertionError:
            record["status"] = "full_training_budget_exhausted"
    status.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print("AXIS_PILOT_PLAN", record["status"], flush=True)


if __name__ == "__main__":
    main()
