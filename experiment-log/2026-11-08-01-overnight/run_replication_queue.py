"""Independent recipe replication after the capacity matrix completes."""
import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path("E:/LJQ_code/LJQ_code_experiment_storage/overnight_v1/20261108_01")
PYTHON = "D:/anaconda3/envs/PY10/python.exe"
DEADLINE = datetime(2026, 11, 8, 23, 20, tzinfo=timezone.utc).timestamp()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def status(stage, **details):
    (ROOT / "replication_queue_status.json").write_text(
        json.dumps({"stage": stage, "time": time.time(), **details}, indent=2), encoding="utf-8")


def main():
    status("waiting_for_capacity_matrix")
    while True:
        capacity = read(ROOT / "capacity_queue_status.json")
        if capacity["stage"] == "completed_review_and_replication_required":
            break
        if "stop" in capacity["stage"] or capacity["stage"] == "failed":
            status("capacity_requires_review", capacity=capacity)
            return
        if time.time() > DEADLINE - 5400:
            status("insufficient_time_for_replication")
            return
        time.sleep(10)
    assert read(ROOT / "capacity_IF3_w32/comparison_vs_pretrained_W.json")["passes"]
    prompt = HERE / "05C_INDEPENDENT_REPLICATION_GOAL.txt"
    for name, width, coverage, initialization in (
            ("replication_W16", 16, "W", "shared"),
            ("replication_M32", 32, "M", "scratch")):
        output = ROOT / name
        if (output / "status.json").exists() and read(output / "status.json")["status"] == "completed":
            continue
        completed = sum("unique_fit_per_epoch" in read(p) for p in ROOT.rglob("receipt.json"))
        remaining = 5 - len(list(output.glob("*/receipt.json")))
        assert completed + remaining <= 60, (completed, remaining)
        command = [PYTHON, str(HERE / "run_variant.py"), "--source", str(ROOT / "replication_data"),
            "--output", str(output), "--model", "dfms_iq", "--width", str(width),
            "--coverage", coverage, "--initialization", initialization, "--seeds",
            *map(str, range(2090, 2095)), "--hypothesis",
            "Independent identity replication of the frozen complete recipe under 05C"]
        status(name, command=command, completed_full_runs=completed,
               prompt_sha256=hashlib.sha256(prompt.read_bytes()).hexdigest())
        log_path = ROOT / f"{name}.log"
        if log_path.exists():
            log_path.rename(log_path.with_name(f"{name}.previous_{time.time_ns()}.log"))
        with log_path.open("wb") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                env=dict(os.environ, PYTHONIOENCODING="utf-8"), creationflags=subprocess.CREATE_NO_WINDOW)
        if result.returncode:
            status("failed", job=name, returncode=result.returncode)
            return
        print("REPLICATION_JOB_COMPLETE", name, flush=True)
    status("completed_requires_analysis_and_freeze")


if __name__ == "__main__":
    main()
