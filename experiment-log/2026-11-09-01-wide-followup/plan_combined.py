"""Complete five-seed independent replication of a successful combined recipe."""
import argparse
import subprocess
import time

from plan_sampling import ROOT, PREVIOUS, HERE, PYTHON, DEADLINE, append_jobs, job, read, write


def advance():
    record = read(ROOT / "combined_plan.json")
    comparison = read(ROOT / "combined_primary_comparison.json")
    if not comparison["passes"]:
        record.update(status="primary_failed_promotion", checks=comparison["checks"])
    elif 5 * record["estimated_per_run"] + 120 > DEADLINE - time.time():
        record["status"] = "independent_replication_time_insufficient"
    else:
        command = [PYTHON, str(HERE / "run_combined.py"), "--source", str(PREVIOUS / "replication_data"),
            "--output", str(ROOT / "combined_replication"), "--model", "dfms_iq", "--width", "32", "--coverage", "M",
            "--initialization", "scratch", "--epochs", "40", "--seeds", *map(str, range(2090, 2095)),
            "--hypothesis", "Independent identity replication of combo sampling plus epsilon0.01 under03D"]
        comparison_command = [PYTHON, str(HERE / "compare_combined.py"), "--reference", str(PREVIOUS / "replication_M32"),
            "--candidate", str(ROOT / "combined_replication"), "--source", str(PREVIOUS / "replication_data"),
            "--mode", "main", "--seeds", *map(str, range(2090, 2095)), "--output", str(ROOT / "combined_replication_comparison.json")]
        additions = [job("combined_replication", command, "full", 5 * record["estimated_per_run"] + 60),
            job("combined_replication_comparison", comparison_command),
            job("combined_register", [PYTHON, str(HERE / "plan_combined.py"), "--phase", "register"])]
        for item in additions:
            item["prompt"] = str(HERE / "03D_COMBINED_REGULARIZATION_GOAL.txt")
        append_jobs(additions)
        record["status"] = "independent_replication_queued"
    write(ROOT / "combined_plan.json", record)


def register():
    record = read(ROOT / "combined_plan.json")
    if not read(ROOT / "combined_replication_comparison.json")["passes"]:
        record["status"] = "independent_replication_failed_promotion"
    else:
        command = [PYTHON, str(HERE / "register_candidate.py"), "--name", "combo_sampling_smoothing_0.01",
            "--reference", record["reference"], "--candidate", record["candidate"],
            "--primary-comparison", str(ROOT / "combined_primary_comparison.json"),
            "--replication-comparison", str(ROOT / "combined_replication_comparison.json"),
            "--use-primary-models", "--seeds", *map(str, record["primary_seeds"])]
        assert subprocess.run(command, creationflags=subprocess.CREATE_NO_WINDOW).returncode == 0
        record["status"] = "registered_for_reserved_confirmation"
    write(ROOT / "combined_plan.json", record)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["advance", "register"], required=True)
    args = parser.parse_args()
    advance() if args.phase == "advance" else register()
