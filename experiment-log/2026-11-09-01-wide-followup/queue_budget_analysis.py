import json
from pathlib import Path

here = Path(__file__).resolve().parent
root = Path("E:/LJQ_code/LJQ_code_experiment_storage/daytime_v1/20261109_01")
path = root / "jobs.json"
jobs = json.loads(path.read_text(encoding="utf-8"))
assert not any(job["name"] == "budget_analysis" for job in jobs)
jobs.append({"name": "budget_analysis", "kind": "analysis", "output": str(root / "budget_analysis"),
    "prompt": str(here / "02_BUDGET_GOAL.txt"),
    "command": ["D:/anaconda3/envs/PY10/python.exe", str(here / "analyze_budget.py")]})
temporary = path.with_suffix(".tmp")
temporary.write_text(json.dumps(jobs, indent=2), encoding="utf-8")
temporary.replace(path)
print("BUDGET_ANALYSIS_QUEUED", flush=True)
