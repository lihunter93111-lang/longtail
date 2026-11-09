"""Sequential parameter control and preregistered capacity/distribution matrix."""
import json
import os
import shutil
import subprocess
import time
from datetime import datetime,timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path("E:/LJQ_code/LJQ_code_experiment_storage/overnight_v1/20261108_01")
PYTHON = "D:/anaconda3/envs/PY10/python.exe"
DEADLINE = datetime(2026,11,8,23,20,tzinfo=timezone.utc).timestamp()


def write_status(stage,**details):
    (ROOT/"capacity_queue_status.json").write_text(json.dumps({"stage":stage,"time":time.time(),**details},indent=2),encoding="utf-8")


def main():
    write_status("waiting_for_pyramid")
    while not (ROOT/"stage4_pyramid/status.json").exists() or json.loads((ROOT/"stage4_pyramid/status.json").read_text(encoding="utf-8"))["status"]!="completed":
        if time.time()>DEADLINE-3600:
            write_status("deadline_before_capacity")
            return
        time.sleep(5)
    jobs = [("stage4_global_control","run_global_control.py","data","dfms_iq_global_statistics_bypass",16,"shared",
        "Parameter-matched global mean/std control for the learned top8 bypass; identical initial state and dropout calls")]
    for distribution,source in (("IF3","data"),("balanced","balanced_data")):
        for width in (16,32):
            jobs.append((f"capacity_{distribution}_w{width}","run_variant.py",source,"dfms_iq",width,"scratch",
                f"Matched random-initialization capacity test width{width} on {distribution} with fixed M coverage and40 epochs"))
    for name,script,source,model,width,initialization,hypothesis in jobs:
        status_path = ROOT/name/"status.json"
        if status_path.exists() and json.loads(status_path.read_text(encoding="utf-8"))["status"]=="completed":
            continue
        required_time = 3600 if width==32 else 1800
        if time.time()>DEADLINE-required_time:
            write_status("time_budget_stop",pending_job=name)
            return
        complete = [p for p in ROOT.rglob("receipt.json")
            if "unique_fit_per_epoch" in json.loads(p.read_text(encoding="utf-8"))]
        remaining_runs = 5-len(list((ROOT/name).glob("*/receipt.json")))
        if len(complete)+remaining_runs>60:
            write_status("full_run_budget_stop",completed_full_runs=len(complete),pending_job=name)
            return
        command = [PYTHON,str(HERE/script),"--source",str(ROOT/source),"--output",str(ROOT/name),
            "--model",model,"--width",str(width),"--coverage","M","--initialization",initialization,"--hypothesis",hypothesis]
        write_status(name,command=command)
        log_path = ROOT/f"{name}.log"
        if log_path.exists():
            shutil.copy2(log_path,log_path.with_name(log_path.name+f".previous_{time.time_ns()}"))
        with log_path.open("wb") as log:
            result = subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,
                env=dict(os.environ,PYTHONIOENCODING="utf-8"),creationflags=subprocess.CREATE_NO_WINDOW)
        if result.returncode:
            write_status("failed",job=name,returncode=result.returncode)
            return
        print("CAPACITY_QUEUE_JOB_COMPLETE",name,flush=True)
    write_status("completed_review_and_replication_required")


if __name__=="__main__":
    main()
