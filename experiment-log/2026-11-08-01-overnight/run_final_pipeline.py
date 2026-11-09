"""Generate, audit, and score the already-frozen confirmation cohort."""
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
ROOT = Path("E:/LJQ_code/LJQ_code_experiment_storage/overnight_v1/20261108_01")
DATA = ROOT / "final_confirmation_data"
FREEZE = ROOT / "final_freeze.json"
PYTHON = "D:/anaconda3/envs/PY10/python.exe"
MATLAB = "D:/Matlab/bin/matlab.exe"


def status(stage, **extra):
    (ROOT / "final_pipeline_status.json").write_text(json.dumps({"stage": stage,
        "utc": datetime.now(timezone.utc).isoformat(), **extra}, indent=2), encoding="utf-8")


def main():
    assert FREEZE.exists()
    source = "E:/论文相关/干扰数据/复合干扰数据0716/九类/full_power_10_30dB_v1/closedset_9class_train.mat"
    generate = f"addpath('{HERE.as_posix()}'); generate_confirmation('{REPO.as_posix()}','{DATA.as_posix()}','{source}','{FREEZE.as_posix()}')"
    pair = f"addpath('{HERE.as_posix()}'); generate_confirmation_pairs('{REPO.as_posix()}','{DATA.as_posix()}')"
    jobs = [
        ("final_generate", [MATLAB, "-batch", generate], DATA / "confirmation_pool/generation_receipt.json"),
        ("final_audit", [PYTHON, str(HERE / "prepare_final.py"), "--root", str(ROOT), "--data", str(DATA),
                         "--freeze", str(FREEZE)], DATA / "preflight.json"),
        ("final_pairing", [MATLAB, "-batch", pair], DATA / "diagnostic_pairing_receipt.json"),
        ("final_score", [PYTHON, str(HERE / "run_confirmation.py"), "--data", str(DATA), "--freeze", str(FREEZE),
                         "--output", str(ROOT / "final_confirmation_results")], ROOT / "final_confirmation_results/status.json")]
    for name, command, receipt in jobs:
        if receipt.exists():
            continue
        status(name, command=command)
        with (ROOT / f"{name}.log").open("wb") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                env=dict(os.environ, PYTHONIOENCODING="utf-8"), creationflags=subprocess.CREATE_NO_WINDOW)
        if result.returncode:
            status("failed", job=name, returncode=result.returncode)
            raise RuntimeError(name)
        print("FINAL_PHASE_COMPLETE", name, flush=True)
    status("completed")


if __name__ == "__main__":
    main()
