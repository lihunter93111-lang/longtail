"""Prepare the conditional axis pilot data independently of GPU training."""
import json
import os
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / "2026-11-08-01-overnight"
ROOT = Path("E:/LJQ_code/LJQ_code_experiment_storage/daytime_v1/20261109_01")
SOURCE = Path("E:/LJQ_code/LJQ_code_experiment_storage/overnight_v1/20261108_01/data")
DATA = ROOT / "axis_data"


def main():
    jobs = []
    if not (DATA / "manifest.json").exists():
        jobs.append(("axis_prepare", ["D:/anaconda3/envs/PY10/python.exe", str(HERE / "prepare_axis_data.py"),
            "--source", str(SOURCE), "--quotas", str(ROOT / "axis_feasibility.json"), "--output", str(DATA)]))
    if not (DATA / "fit_pairing_receipt.json").exists():
        expression = f"addpath('{OLD.as_posix()}'); generate_paired_views('{HERE.parent.parent.as_posix()}','{DATA.as_posix()}')"
        jobs.append(("axis_pairing", ["D:/Matlab/bin/matlab.exe", "-batch", expression]))
    for name, command in jobs:
        (ROOT / "axis_pipeline_status.json").write_text(json.dumps({"stage": name, "command": command}, indent=2), encoding="utf-8")
        with (ROOT / f"{name}.log").open("ab") as log:
            completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                env=dict(os.environ, PYTHONIOENCODING="utf-8"), creationflags=subprocess.CREATE_NO_WINDOW)
        assert completed.returncode == 0, name
    (ROOT / "axis_pipeline_status.json").write_text(json.dumps({"stage": "completed"}), encoding="utf-8")
    print("AXIS_DATA_PIPELINE_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
