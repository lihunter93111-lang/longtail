"""Deadline-triggered frozen confirmation and delivery, independent of the chat loop."""
import json
import os
import msvcrt
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import training

base = training.base
HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
ROOT = Path("E:/LJQ_code/LJQ_code_experiment_storage/daytime_v1/20261109_01")
PREVIOUS = Path("E:/LJQ_code/LJQ_code_experiment_storage/overnight_v1/20261108_01")
PYTHON = "D:/anaconda3/envs/PY10/python.exe"
MATLAB = "D:/Matlab/bin/matlab.exe"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def run(name, command, timeout=None):
    base.write(ROOT / "closeout_status.json", {"stage": name, "utc": datetime.now(timezone.utc).isoformat()})
    with (ROOT / f"{name}.log").open("ab") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
            env=dict(os.environ, PYTHONIOENCODING="utf-8"), creationflags=subprocess.CREATE_NO_WINDOW, timeout=timeout)
    if result.returncode:
        raise RuntimeError(f"{name} failed with return code {result.returncode}")


def finalize_pending_evaluation():
    limit = datetime(2026, 11, 9, 8, 9, 30, tzinfo=timezone.utc).timestamp()
    attempted = {}
    completed = read(ROOT / "supervisor_jobs.json")
    while time.time() < limit:
        eligible = []
        for job in read(ROOT / "jobs.json"):
            name = job["name"]
            if name in attempted or completed.get(name, {}).get("status") == "completed":
                continue
            if job["kind"] == "full":
                command = job["command"]
                start = command.index("--seeds") + 1
                seed_tokens = command[start:]
                count = next((i for i, value in enumerate(seed_tokens) if value.startswith("--")), len(seed_tokens))
                seeds = list(map(int, seed_tokens[:count]))
                output = Path(job["output"])
                if not all((output / f"M_{seed}/receipt.json").exists() for seed in seeds):
                    continue
                # The frozen runner returns immediately from each completed trainer receipt.
                eligible.append(job)
            elif job["kind"] in ("analysis", "evaluation"):
                eligible.append(job)
        if not eligible:
            break
        job = eligible[0]
        timeout = min(120, max(1, limit - time.time()))
        try:
            run("closeout_recover_" + job["name"], job["command"], timeout=timeout)
            attempted[job["name"]] = {"status": "completed", "optimization_updates": 0}
        except (RuntimeError, subprocess.TimeoutExpired) as error:
            attempted[job["name"]] = {"status": "incomplete", "error": str(error)}
        base.write(ROOT / "closeout_recovered_jobs.json", attempted)


def main():
    finalize_pending_evaluation()
    selection_path = ROOT / "selection.json"
    if not selection_path.exists():
        base.write(ROOT / "confirmation_decision.json", {"status": "no_qualified_candidate_registered",
            "reserved_seed_consumed": False, "utc": datetime.now(timezone.utc).isoformat()})
        run("finish_report", [PYTHON, str(HERE / "finish_report.py")])
        base.write(ROOT / "closeout_status.json", {"stage": "completed_without_new_confirmation"})
        return
    selection = read(selection_path)
    assert selection["eligible_for_reserved_confirmation"]
    for key in ("primary_comparison", "replication_comparison"):
        if selection.get(key) is None:
            continue
        assert base.sha(Path(selection[key])) == selection[key + "_sha256"]
        assert read(Path(selection[key]))["passes"]
    freeze_path = ROOT / "final_freeze.json"
    if not freeze_path.exists():
        sources = [HERE / name for name in ("closeout.py", "run_confirmation.py", "prepare_final.py",
            "generate_confirmation.m", "generate_confirmation_pairs.m", "05_CONFIRM_GOAL.txt")]
        sources += list((base.REPO / "projects/CJ_Imbalance_0716/cjlt/models").glob("*.py"))
        base.write(freeze_path, {**selection, "all_models_and_calibrations_frozen": True,
            "frozen_utc": datetime.now(timezone.utc).isoformat(), "selection_sha256": base.sha(selection_path),
            "source_sha256": {str(path): base.sha(path) for path in sources}, "confirmation_seed": 3109310900})
    data = ROOT / "final_confirmation_data"
    source = "E:/论文相关/干扰数据/复合干扰数据0716/九类/full_power_10_30dB_v1/closedset_9class_train.mat"
    generate = f"addpath('{HERE.as_posix()}'); generate_confirmation('{REPO.as_posix()}','{data.as_posix()}','{source}','{freeze_path.as_posix()}')"
    pair = f"addpath('{HERE.as_posix()}'); generate_confirmation_pairs('{REPO.as_posix()}','{data.as_posix()}')"
    if not (data / "confirmation_pool/generation_receipt.json").exists():
        run("final_generate", [MATLAB, "-batch", generate])
    if not (data / "preflight.json").exists():
        run("final_audit", [PYTHON, str(HERE / "prepare_final.py"), "--root", str(PREVIOUS), "--data", str(data), "--freeze", str(freeze_path)])
    if not (data / "diagnostic_pairing_receipt.json").exists():
        run("final_pairing", [MATLAB, "-batch", pair])
    result = ROOT / "final_confirmation_results"
    if not (result / "status.json").exists():
        run("final_score", [PYTHON, str(HERE / "run_confirmation.py"), "--data", str(data), "--freeze", str(freeze_path), "--output", str(result)])
    run("finish_report", [PYTHON, str(HERE / "finish_report.py")])
    base.write(ROOT / "closeout_status.json", {"stage": "completed", "utc": datetime.now(timezone.utc).isoformat()})


if __name__ == "__main__":
    with (ROOT / "closeout.lock").open("a+b") as lock:
        if lock.tell() == 0:
            lock.write(b"0")
            lock.flush()
        lock.seek(0)
        try:
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            print("CLOSEOUT_ALREADY_RUNNING", flush=True)
        else:
            try:
                main()
            finally:
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
