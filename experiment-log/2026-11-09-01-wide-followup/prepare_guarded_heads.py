"""Prepare the explicit low-step head diagnostic with an epoch-zero anchor."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / "2026-11-08-01-overnight"
ROOT = Path("E:/LJQ_code/LJQ_code_experiment_storage/daytime_v1/20261109_01")
PREVIOUS = Path("E:/LJQ_code/LJQ_code_experiment_storage/overnight_v1/20261108_01")


def main():
    source = (OLD / "run_generic_heads.py").read_text(encoding="utf-8")
    source = source.replace("import training\n", "import time\nimport sys\nfrom datetime import datetime, timezone\nsys.path.append(str(Path(__file__).resolve().parent.parent/'2026-11-08-01-overnight'))\nimport training\n")
    source = source.replace("lr=1e-3", "lr=1e-4").replace('"lr":1e-3', '"lr":1e-4')
    source = source.replace('HERE/"03E_WIDE_HEAD_GOAL.txt"', 'HERE/"01B_GUARDED_HEAD_GOAL.txt"')
    source = source.replace('best = float("inf")', """head.eval()
    with torch.no_grad():
        best = float(F.binary_cross_entropy_with_logits(head(early_x),early_y))
    initial_early_bce = best
    helper.atomic_torch_save({"head_state_dict":head.state_dict(),"epoch":0,"early_BCE":best},folder/"best_head.pth")""")
    source = source.replace('"best_early_BCE":best,"checkpoint_sha256"', '"best_early_BCE":best,"initial_early_BCE":initial_early_bce,"checkpoint_sha256"')
    source = source.replace('"selection":"minimum hard unweighted early-stop BCE"', '"selection":"minimum hard unweighted early-stop BCE including original epoch0"')
    source = source.replace("        for index in order.cuda().split(32):\n", """        for index in order.cuda().split(32):
            if time.time() >= datetime(2026,11,9,8,0,tzinfo=timezone.utc).timestamp():
                helper.atomic_torch_save({"head_state_dict":head.state_dict(),"optimizer":optimizer.state_dict(),
                    "epoch":epoch,"partial_epoch":True},folder/"deadline_partial.pth")
                raise RuntimeError("Head optimization deadline reached")
""")
    target = HERE / "run_guarded_heads.py"
    target.write_text(source, encoding="utf-8")
    queue = ROOT / "jobs.json"
    jobs = json.loads(queue.read_text(encoding="utf-8"))
    additions = []
    for split, data, models in (("primary", "data", "capacity_IF3_w32"),
                                ("replication", "replication_data", "replication_M32")):
        name = f"guarded_heads_{split}"
        additions.append({"name": name, "kind": "head", "output": str(ROOT / name),
            "prompt": str(HERE / "01B_GUARDED_HEAD_GOAL.txt"), "command": ["D:/anaconda3/envs/PY10/python.exe", str(target),
                "--source", str(PREVIOUS / data), "--models", str(PREVIOUS / models), "--output", str(ROOT / name), "--views", "coverage"]})
    assert not any(job["name"] in {item["name"] for item in jobs} for job in additions)
    position = next(i for i, job in enumerate(jobs) if job["name"] == "budget_e40")
    jobs[position:position] = additions
    temporary = queue.with_suffix(".tmp")
    temporary.write_text(json.dumps(jobs, indent=2), encoding="utf-8")
    temporary.replace(queue)
    print("GUARDED_HEADS_QUEUED", flush=True)


if __name__ == "__main__":
    main()
