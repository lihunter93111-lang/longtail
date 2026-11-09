"""Sequential data audit, physical pairing, and first overnight training stage."""
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
ROOT = Path("E:/LJQ_code/LJQ_code_experiment_storage/overnight_v1/20261108_01")
DATA = ROOT/"data"
PYTHON = "D:/anaconda3/envs/PY10/python.exe"
MATLAB = "D:/Matlab/bin/matlab.exe"


def state(stage, **details):
    ROOT.mkdir(parents=True,exist_ok=True)
    path = ROOT/"pipeline_status.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps({"stage":stage,"time":time.time(),**details},ensure_ascii=False,indent=2),encoding="utf-8")
    temporary.replace(path)
    print(stage,flush=True)


def run(stage, command):
    state(stage,command=command)
    environment = dict(os.environ,PYTHONIOENCODING="utf-8")
    with (ROOT/f"{stage}.log").open("wb") as output:
        completed = subprocess.run(command,stdout=output,stderr=subprocess.STDOUT,env=environment,
            creationflags=subprocess.CREATE_NO_WINDOW)
    if completed.returncode:
        raise RuntimeError(f"{stage} failed with code {completed.returncode}; see {ROOT/stage}.log")


def main():
    state("waiting_for_generated_pools")
    deadline = time.time()+1200
    while not all((DATA/role/"generation_receipt.json").exists() for role in ("fit_pool","dev_pool")):
        if time.time()>deadline:
            raise RuntimeError("Pool generation wait exceeded 20 minutes")
        time.sleep(5)
    if not (DATA/"preflight.json").exists():
        run("prepare",[PYTHON,str(HERE/"prepare_fresh.py"),"--prompt",str(HERE/"GOAL_PROMPT.txt"),
            "--original","E:/LJQ_code/LJQ_code_experiment_storage/lt_abc_v1/20261106_local_01","--output",str(DATA)])
    if not (DATA/"diagnostic_pairing_receipt.json").exists():
        expression = f"addpath('{HERE.as_posix()}'); generate_paired_views('{REPO.as_posix()}','{DATA.as_posix()}')"
        run("pairing",[MATLAB,"-batch",expression])
    if not (ROOT/"stage2/decision.json").exists():
        run("stage2",[PYTHON,str(HERE/"resume_stage2.py"),"--source",str(DATA),"--output",str(ROOT/"stage2")])
    state("stage2_complete_next_stage_required")


if __name__=="__main__":
    try:
        main()
    except Exception:
        state("failed",error=traceback.format_exc())
        raise
