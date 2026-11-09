"""Insert replay, physical evaluation, calibration and matched head comparisons."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / "2026-11-08-01-overnight"
ROOT = Path("E:/LJQ_code/LJQ_code_experiment_storage/daytime_v1/20261109_01")
PREVIOUS = Path("E:/LJQ_code/LJQ_code_experiment_storage/overnight_v1/20261108_01")
PYTHON = "D:/anaconda3/envs/PY10/python.exe"


def main():
    path = ROOT / "jobs.json"
    jobs = json.loads(path.read_text(encoding="utf-8"))
    additions = []
    for split, source, backbone, seeds in (("primary", "data", "capacity_IF3_w32", range(2080, 2085)),
                                          ("replication", "replication_data", "replication_M32", range(2090, 2095))):
        for variant in ("hard", "balanced"):
            name = f"wide_heads_{split}_{variant}"
            output = ROOT / name
            additions.append({"name": name, "kind": "evaluation", "output": str(output),
                "prompt": str(HERE / "01_WIDE_HEAD_GOAL.txt"),
                "command": [PYTHON, str(HERE / "export_evaluate_heads.py"), "--source", str(PREVIOUS / source),
                    "--backbones", str(PREVIOUS / backbone), "--heads", str(ROOT / f"wide_heads_{split}"),
                    "--output", str(output), "--variant", variant]})
            additions.append({"name": f"{name}_comparison", "kind": "analysis", "output": str(output / "comparison_job"),
                "prompt": str(HERE / "01_WIDE_HEAD_GOAL.txt"),
                "command": [PYTHON, str(OLD / "compare_runs.py"), "--reference", str(PREVIOUS / backbone),
                    "--candidate", str(output), "--source", str(PREVIOUS / source), "--reference-pattern", "M_{seed}",
                    "--candidate-pattern", "M_{seed}", "--seeds", *map(str, seeds), "--output", str(output / "comparison.json")]})
            additions.append({"name": f"{name}_affine", "kind": "calibration", "output": str(ROOT / f"{name}_affine"),
                "prompt": str(HERE / "01_WIDE_HEAD_GOAL.txt"),
                "command": [PYTHON, str(OLD / "affine_calibration.py"), "--models", str(output),
                    "--output", str(ROOT / f"{name}_affine")]})
    assert not any(job["name"] in {item["name"] for item in jobs} for job in additions)
    jobs = jobs[:2] + additions + jobs[2:]
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(jobs, indent=2), encoding="utf-8")
    temporary.replace(path)
    print("HEAD_ANALYSIS_QUEUED", len(additions), flush=True)


if __name__ == "__main__":
    main()
