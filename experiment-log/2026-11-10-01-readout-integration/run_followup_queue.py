"""Run predeclared follow-ups after the original expert queue releases training."""
import subprocess
import sys
import time
from pathlib import Path
from common import ROOT, CONFIG, check_deadline, stamp, write

HERE = Path(__file__).resolve().parent


def run(script, arguments, log_name):
    check_deadline()
    with (ROOT / log_name).open("a", encoding="utf-8") as log:
        subprocess.run([sys.executable, str(HERE / script), *arguments], cwd=HERE,
                       stdout=log, stderr=subprocess.STDOUT, check=True)


def main():
    while not (ROOT / "split2" / "expert_status.json").exists():
        check_deadline()
        time.sleep(30)
    for split in CONFIG["splits"]:
        run("train_controls.py", ["--split", split], f"controls_{split}.log")
        run("evaluate_stage.py", ["--split", split, "--tag", "sampling_controls", "--variants",
            "ordinary_expert", "combo_expert", "weighted_expert", "bootstrap_expert"],
            f"controls_evaluation_{split}.log")
        methods = ["reference_affine", "ordinary_mean", "combo_mean", "ordinary_affine", "combo_affine",
                   "bootstrap_mean", "bootstrap_affine", "weighted_mean", "weighted_affine"]
        run("fit_fusions.py", ["--split", split, "--tag", "controls", "--methods", *methods],
            f"controls_fusion_fit_{split}.log")
        run("evaluate_fusions.py", ["--split", split, "--tag", "controls", "--methods", *methods],
            f"controls_fusion_evaluation_{split}.log")
    for split in CONFIG["splits"]:
        run("adaptive_readouts.py", ["--split", split], f"adaptive_{split}.log")
        run("evaluate_adaptive.py", ["--split", split], f"adaptive_evaluation_{split}.log")
    write(ROOT / "followup_queue_status.json", {"status": "completed", "time": stamp()})


if __name__ == "__main__":
    main()
