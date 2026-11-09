"""Run prespecified decision and frozen-head diagnostics after coverage scoring."""
import json
import os
import subprocess
import time
from datetime import datetime,timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path("E:/LJQ_code/LJQ_code_experiment_storage/overnight_v1/20261108_01")


def state(stage,**details):
    (ROOT/"stage3_pipeline_status.json").write_text(json.dumps({"stage":stage,"time":time.time(),**details},indent=2),encoding="utf-8")


def main():
    state("waiting_for_coverage_results")
    deadline = datetime(2026,11,8,22,0,tzinfo=timezone.utc).timestamp()
    while not (ROOT/"stage2/decision.json").exists():
        if time.time()>deadline:
            state("deadline_before_stage3")
            return
        time.sleep(5)
    decision = json.loads((ROOT/"stage2/decision.json").read_text(encoding="utf-8"))
    state("coverage_review",coverage_comparisons=decision["comparisons"],
        rationale="Inspect per-class operating points and frozen representations for both preregistered coverage arms")
    for stage,script,output in (("decision","decision_diagnostics.py","stage3_decision"),("heads","run_heads.py","stage3_heads")):
        if (ROOT/output/"status.json").exists():
            continue
        command = ["D:/anaconda3/envs/PY10/python.exe",str(HERE/script),"--source",str(ROOT/"data"),
            "--models",str(ROOT/"stage2"),"--output",str(ROOT/output)]
        state(stage,command=command)
        with (ROOT/f"{output}.log").open("wb") as log:
            result = subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,
                env=dict(os.environ,PYTHONIOENCODING="utf-8"),creationflags=subprocess.CREATE_NO_WINDOW)
        if result.returncode:
            state("failed",failed_stage=stage,returncode=result.returncode)
            return
    state("stage3_complete_review_required")
    print("STAGE3_PIPELINE_COMPLETE",flush=True)


if __name__=="__main__":
    main()
