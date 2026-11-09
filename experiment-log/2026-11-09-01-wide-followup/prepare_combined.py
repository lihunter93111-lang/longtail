"""Version the combined sampling and confidence-regularization experiment."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path("E:/LJQ_code/LJQ_code_experiment_storage/daytime_v1/20261109_01")


def main():
    assert not (ROOT / "smoothed_primary").exists()
    source = (HERE / "run_sampling.py").read_text(encoding="utf-8")
    source = source.replace('HERE/"03A_COMBINATION_SAMPLING_GOAL.txt"', 'HERE/"03D_COMBINED_REGULARIZATION_GOAL.txt"')
    source = source.replace('"initialization":args.initialization,"sampler_sha256"',
                            '"initialization":args.initialization,"label_smoothing_epsilon":.01,"checkpoint_metric":"hard early BCE","sampler_sha256"')
    source = source.replace('    assert args.coverage == "M"', '    assert args.coverage == "M" and args.epochs == 40\n    assert training.ARMS["M"]["smoothing"] == .01')
    source = source.replace('def main():', 'def run_experiment():')
    source = source.replace('if __name__=="__main__":\n    main()', '''def main():
    original_smoothing = training.ARMS["M"]["smoothing"]
    training.ARMS["M"]["smoothing"] = .01
    try:
        run_experiment()
    finally:
        training.ARMS["M"]["smoothing"] = original_smoothing


if __name__=="__main__":
    main()''')
    (HERE / "run_combined.py").write_text(source, encoding="utf-8")
    old_plan = json.loads((ROOT / "smoothing_plan.json").read_text(encoding="utf-8"))
    jobs_path = ROOT / "jobs.json"
    jobs = json.loads(jobs_path.read_text(encoding="utf-8"))
    old_names = {"smoothed_primary", "smoothing_primary_comparison", "smoothing_advance"}
    old_jobs = [job for job in jobs if job["name"] in old_names]
    assert len(old_jobs) == 3
    first_position = next(i for i, job in enumerate(jobs) if job["name"] == "smoothed_primary")
    py = "D:/anaconda3/envs/PY10/python.exe"
    previous = Path("E:/LJQ_code/LJQ_code_experiment_storage/overnight_v1/20261108_01")
    def job(name, command, kind="analysis", estimate=0):
        return {"name": name, "command": command, "kind": kind, "output": str(ROOT / name),
            "prompt": str(HERE / "03D_COMBINED_REGULARIZATION_GOAL.txt"), "estimated_seconds": estimate}
    command = [py, str(HERE / "run_combined.py"), "--source", str(previous / "data"), "--output", str(ROOT / "combined_primary"),
        "--model", "dfms_iq", "--width", "32", "--coverage", "M", "--initialization", "scratch", "--epochs", "40",
        "--seeds", *map(str, range(2100, 2105)), "--hypothesis", "Cardinality-stratified combo mixture plus epsilon0.01 under03D"]
    additions = [job("combined_primary", command, "full", 5 * old_plan["estimated_per_run"] + 60)]
    for mode, reference, name in (("main", "sampling_reference_primary", "combined_primary_comparison"),
                                   ("incremental", "sampling_candidate_primary", "combined_incremental_comparison")):
        additions.append(job(name, [py, str(HERE / "compare_combined.py"), "--reference", str(ROOT / reference),
            "--candidate", str(ROOT / "combined_primary"), "--source", str(previous / "data"), "--mode", mode,
            "--seeds", *map(str, range(2100, 2105)), "--output", str(ROOT / f"{name}.json")]))
    additions.append(job("combined_advance", [py, str(HERE / "plan_combined.py"), "--phase", "advance"]))
    revised = [job for job in jobs if job["name"] not in old_names]
    revised[first_position:first_position] = additions
    revision = {"utc": datetime.now(timezone.utc).isoformat(), "old_plan": old_plan, "old_pending_jobs": old_jobs,
        "reason": "raw sampling gain uncertain but matched class-affine control is positive in5/5 seeds, so test the combined mechanism",
        "old_prompt_sha256": hashlib.sha256((HERE / "03C_CONFIDENCE_REGULARIZATION_GOAL.txt").read_bytes()).hexdigest(),
        "new_prompt_sha256": hashlib.sha256((HERE / "03D_COMBINED_REGULARIZATION_GOAL.txt").read_bytes()).hexdigest(),
        "no_original_branch_model_started": True}
    (ROOT / "branch_revision_03D.json").write_text(json.dumps(revision, indent=2), encoding="utf-8")
    old_plan.update(status="deferred_before_training_replaced_by_combined_protocol", replacement="03D")
    (ROOT / "smoothing_plan.json").write_text(json.dumps(old_plan, indent=2), encoding="utf-8")
    combined = {"status": "primary_queued", "epsilon": .01, "sampling": "cardinality_stratified_combo_mix_0.5", "epochs": 40,
        "primary_seeds": list(range(2100, 2105)), "replication_seeds": list(range(2090, 2095)),
        "estimated_per_run": old_plan["estimated_per_run"], "reference": str(ROOT / "sampling_reference_primary"),
        "candidate": str(ROOT / "combined_primary"), "revision_path": str(ROOT / "branch_revision_03D.json")}
    (ROOT / "combined_plan.json").write_text(json.dumps(combined, indent=2), encoding="utf-8")
    temporary = jobs_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(revised, indent=2), encoding="utf-8")
    temporary.replace(jobs_path)
    print("COMBINED_BRANCH_REVISED_BEFORE_TRAINING", flush=True)


if __name__ == "__main__":
    main()
