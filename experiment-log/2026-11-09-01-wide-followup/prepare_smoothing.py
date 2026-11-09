"""Reuse the frozen numerical trainer with an explicit smoothing configuration."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
source = (HERE / "run_variant.py").read_text(encoding="utf-8")
source = source.replace('HERE/"02_BUDGET_GOAL.txt"', 'HERE/"03C_CONFIDENCE_REGULARIZATION_GOAL.txt"')
source = source.replace('"initialization":args.initialization,"seeds":args.seeds,',
                        '"initialization":args.initialization,"label_smoothing_epsilon":.01,"checkpoint_metric":"hard early BCE","seeds":args.seeds,')
source = source.replace('    args.arms = [args.coverage]', '    assert args.coverage == "M" and args.epochs == 40\n    assert training.ARMS["M"]["smoothing"] == .01\n    args.arms = [args.coverage]')
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
(HERE / "run_smoothing.py").write_text(source, encoding="utf-8")
print("SMOOTHING_RUNNER_PREPARED", flush=True)
