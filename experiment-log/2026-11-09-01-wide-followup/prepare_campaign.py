"""Create a versioned trainer and initial queue without altering prior runs."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / "2026-11-08-01-overnight"
ROOT = Path("E:/LJQ_code/LJQ_code_experiment_storage/daytime_v1/20261109_01")
PREVIOUS = Path("E:/LJQ_code/LJQ_code_experiment_storage/overnight_v1/20261108_01")
PYTHON = "D:/anaconda3/envs/PY10/python.exe"


def replace_once(source, old, new):
    assert source.count(old) == 1, old
    return source.replace(old, new)


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    trainer = (OLD / "training.py").read_text(encoding="utf-8")
    trainer = replace_once(trainer, "import hashlib\n", "import hashlib\nimport json\n")
    trainer = replace_once(trainer, "range(len(history)+1,41)", "range(len(history)+1,args.epochs+1)")
    trainer = replace_once(trainer, "math.pi*(epoch-1)/39", "math.pi*(epoch-1)/(args.epochs-1)")
    trainer = replace_once(trainer, '"epochs":40,"updates":10000', '"epochs":args.epochs,"updates":250*args.epochs')
    trainer = replace_once(trainer, "        for batch in order.cuda().split(32):\n", """        for batch_number,batch in enumerate(order.cuda().split(32)):
            if time.time() >= args.training_deadline:
                atomic_torch_save({"model_state_dict":model.state_dict(),"optimizer":optimizer.state_dict(),
                    "epoch":epoch,"completed_batches":batch_number,"order":order,
                    "protocol_sha256":fingerprint,"partial_epoch":True},folder/"deadline_partial.pth")
                raise RuntimeError("Optimization deadline reached. Last full epoch recovery and partial snapshot saved.")
""")
    (HERE / "training.py").write_text(trainer, encoding="utf-8")
    runner = (OLD / "run_variant.py").read_text(encoding="utf-8")
    runner = replace_once(runner, "import training\nimport run_stage2", "import training\nimport sys\nsys.path.append(str(Path(__file__).resolve().parent.parent/'2026-11-08-01-overnight'))\nimport run_stage2")
    runner = replace_once(runner, 'parser.add_argument("--model",default="dfms_iq")', 'parser.add_argument("--epochs",type=int,choices=[40,80,160],required=True)\n    parser.add_argument("--model",default="dfms_iq")')
    runner = replace_once(runner, "datetime(2026,11,8,23,20,tzinfo=timezone.utc)", "datetime(2026,11,9,8,0,tzinfo=timezone.utc)")
    runner = replace_once(runner, 'HERE/"run_stage2.py"', 'HERE.parent/"2026-11-08-01-overnight/run_stage2.py"')
    runner = replace_once(runner, 'HERE/"04_STRUCTURE_GOAL.txt"', 'HERE/"02_BUDGET_GOAL.txt"')
    runner = replace_once(runner, '"epochs":40,"batch":32,"updates":10000', '"epochs":args.epochs,"batch":32,"updates":250*args.epochs')
    runner = replace_once(runner, '20*len(p10)', '(args.epochs//2)*len(p10)')
    runner = replace_once(runner, '10*len(p15)', '(args.epochs//4)*len(p15)')
    (HERE / "run_variant.py").write_text(runner, encoding="utf-8")
    jobs = []
    for label, source, models in (("primary", "data", "capacity_IF3_w32"),
                                  ("replication", "replication_data", "replication_M32")):
        name = f"wide_heads_{label}"
        jobs.append({"name": name, "kind": "head", "output": str(ROOT / name),
            "command": [PYTHON, str(OLD / "run_generic_heads.py"), "--source", str(PREVIOUS / source),
                "--models", str(PREVIOUS / models), "--output", str(ROOT / name), "--views", "coverage"],
            "prompt": str(HERE / "01_WIDE_HEAD_GOAL.txt")})
    for epochs in (40, 80, 160):
        name = f"budget_e{epochs}"
        jobs.append({"name": name, "kind": "full", "output": str(ROOT / name),
            "command": [PYTHON, str(HERE / "run_variant.py"), "--source", str(PREVIOUS / "data"),
                "--output", str(ROOT / name), "--model", "dfms_iq", "--width", "32", "--coverage", "M",
                "--initialization", "scratch", "--epochs", str(epochs), "--seeds", "2100", "2101", "2102",
                "--hypothesis", "Independent matched 40/80/160 epoch width32 budget diagnostic"],
            "prompt": str(HERE / "02_BUDGET_GOAL.txt")})
    queue = ROOT / "jobs.json"
    assert not queue.exists()
    queue.write_text(json.dumps(jobs, indent=2), encoding="utf-8")
    evidence = {"previous_commit": "1ecfd499a6bc914d5f7dbd2e7334efe45707ff26",
        "sync": "git pull --ff-only succeeded with per-command proxy disabled after stale localhost proxy failed",
        "prior_root": str(PREVIOUS), "new_root": str(ROOT),
        "source_hashes": {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in (OLD / "training.py", OLD / "run_variant.py", HERE / "training.py", HERE / "run_variant.py")},
        "adaptations": ["explicit json import", "declared variable epoch budget", "new optimization deadline", "per-batch deadline snapshot"],
        "preserved": "prior experiment sources and checkpoints"}
    (ROOT / "preflight.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    print("CAMPAIGN_PREPARED", len(jobs), flush=True)


if __name__ == "__main__":
    main()
