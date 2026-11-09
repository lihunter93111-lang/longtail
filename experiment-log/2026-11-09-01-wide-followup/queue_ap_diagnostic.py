import json
from pathlib import Path

here = Path(__file__).resolve().parent
root = Path("E:/LJQ_code/LJQ_code_experiment_storage/daytime_v1/20261109_01")
previous = Path("E:/LJQ_code/LJQ_code_experiment_storage/overnight_v1/20261108_01")
path = root / "jobs.json"
jobs = json.loads(path.read_text(encoding="utf-8"))
assert not any(job["name"].startswith("ap_selection_e") for job in jobs)
additions = [{"name": f"ap_selection_e{epochs}", "kind": "calibration", "output": str(root / f"ap_selection_e{epochs}"),
    "prompt": str(here / "02B_AP_CHECKPOINT_GOAL.txt"),
    "command": ["D:/anaconda3/envs/PY10/python.exe", str(here / "evaluate_ap_checkpoint.py"), "--models", str(root / f"budget_e{epochs}"),
        "--source", str(previous / "data"), "--output", str(root / f"ap_selection_e{epochs}")]} for epochs in (40, 80, 160)]
position = next(i for i, job in enumerate(jobs) if job["name"] == "sampling_plan")
jobs[position:position] = additions
temporary = path.with_suffix(".tmp")
temporary.write_text(json.dumps(jobs, indent=2), encoding="utf-8")
temporary.replace(path)
print("AP_DIAGNOSTICS_QUEUED", flush=True)
