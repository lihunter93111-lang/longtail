"""Independent bounded queue supervision and deadline-triggered closeout."""
import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path("E:/LJQ_code/LJQ_code_experiment_storage/daytime_v1/20261109_01")
STOP_TRAINING = datetime(2026, 11, 9, 8, 0, tzinfo=timezone.utc).timestamp()
START_CLOSEOUT = datetime(2026, 11, 9, 8, 5, tzinfo=timezone.utc).timestamp()
END = datetime(2026, 11, 9, 9, 0, tzinfo=timezone.utc).timestamp()
PYTHON = "D:/anaconda3/envs/PY10/python.exe"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def main():
    receipts = ROOT / "supervisor_jobs.json"
    completed = read(receipts) if receipts.exists() else {}
    closeout_started = False
    while time.time() < END:
        now = time.time()
        write(ROOT / "supervisor_status.json", {"stage": "watching", "utc": datetime.now(timezone.utc).isoformat(),
            "completed_jobs": list(completed), "training_deadline": STOP_TRAINING, "closeout_deadline": END})
        jobs = read(ROOT / "jobs.json")
        pending = [job for job in jobs if job["name"] not in completed]
        if pending and now < STOP_TRAINING:
            job = pending[0]
            output = Path(job["output"])
            if (output / "status.json").exists() and read(output / "status.json")["status"] == "completed":
                completed[job["name"]] = {"status": "completed", "reused_current_campaign_receipt": True}
                write(receipts, completed)
                continue
            if job.get("estimated_seconds", 0) > STOP_TRAINING - now:
                completed[job["name"]] = {"status": "insufficient_time", "utc": datetime.now(timezone.utc).isoformat()}
                write(receipts, completed)
                continue
            prompt = Path(job["prompt"])
            record = {"started_utc": datetime.now(timezone.utc).isoformat(), "command": job["command"],
                      "prompt_sha256": hashlib.sha256(prompt.read_bytes()).hexdigest()}
            with (ROOT / f"{job['name']}.log").open("ab") as log:
                child = subprocess.Popen(job["command"], stdout=log, stderr=subprocess.STDOUT,
                    env=dict(os.environ, PYTHONIOENCODING="utf-8"), creationflags=subprocess.CREATE_NO_WINDOW)
                while child.poll() is None:
                    write(ROOT / "supervisor_status.json", {"stage": job["name"], "pid": child.pid,
                        "utc": datetime.now(timezone.utc).isoformat(), "completed_jobs": list(completed)})
                    if time.time() >= STOP_TRAINING + 30:
                        subprocess.run(["taskkill", "/PID", str(child.pid), "/T", "/F"], capture_output=True)
                        child.wait()
                        break
                    time.sleep(5)
            record.update(returncode=child.returncode, ended_utc=datetime.now(timezone.utc).isoformat(),
                          status="completed" if child.returncode == 0 else "failed_requires_review")
            completed[job["name"]] = record
            write(receipts, completed)
            print("JOB_ENDED", job["name"], child.returncode, flush=True)
            continue
        if now >= START_CLOSEOUT and not closeout_started:
            closeout = HERE / "closeout.py"
            if closeout.exists():
                with (ROOT / "closeout.log").open("ab") as log:
                    child = subprocess.Popen([PYTHON, str(closeout)], stdout=log, stderr=subprocess.STDOUT,
                        env=dict(os.environ, PYTHONIOENCODING="utf-8"), creationflags=subprocess.CREATE_NO_WINDOW)
                write(ROOT / "closeout_launch.json", {"pid": child.pid, "utc": datetime.now(timezone.utc).isoformat()})
                closeout_started = True
            else:
                write(ROOT / "closeout_attention.json", {"status": "closeout_script_missing", "utc": datetime.now(timezone.utc).isoformat()})
        time.sleep(10)
    write(ROOT / "supervisor_status.json", {"stage": "deadline_reached", "utc": datetime.now(timezone.utc).isoformat(),
                                           "completed_jobs": completed, "closeout_started": closeout_started})


if __name__ == "__main__":
    main()
