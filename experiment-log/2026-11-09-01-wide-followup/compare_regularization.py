"""Verify matched loss-regularization comparisons, including reused controls."""
import argparse
import sys
from pathlib import Path

import torch
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.append(str(HERE.parent / "2026-11-08-01-overnight"))
from compare_runs import compare, base, helper
from compare_sampling import read


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    args = parser.parse_args()
    assert len(args.seeds) == 5
    protocols = [read(path / "protocol.json") for path in (args.reference, args.candidate)]
    for key in ("model", "width", "coverage", "initialization", "epochs", "batch", "updates", "lr", "weight_decay", "dropout", "clip", "manifest_sha256"):
        assert protocols[0][key] == protocols[1][key], key
    assert protocols[0].get("label_smoothing_epsilon", 0) == 0
    assert protocols[1]["label_smoothing_epsilon"] == .01
    manifest = read(args.source / "manifest.json")
    positions = np.linspace(0, len(manifest["development"]) - 1, 64, dtype=int)
    x, y = base.load(manifest["dev_path"], [manifest["development"][index] for index in positions])
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    x = x.cuda()
    matched = []
    for seed in args.seeds:
        folder = args.reference / f"M_{seed}"
        reference = read(folder / "receipt.json") if (folder / "receipt.json").exists() else read(folder / "source_receipt.json")["receipt"]
        candidate = read(args.candidate / f"M_{seed}/receipt.json")
        for key in ("initial_state_sha256", "order_sha256", "epochs", "updates"):
            assert reference[key] == candidate[key], (seed, key)
        replay = {}
        for role, parent, receipt in (("reference", args.reference, reference), ("candidate", args.candidate, candidate)):
            directory = parent / f"M_{seed}"
            checkpoint = directory / "best_BCE.pth"
            assert base.sha(checkpoint) == receipt["checkpoint_sha256"]
            model = base.build_backbone("dfms_iq", num_classes=6, width=32, dropout=.2).cuda()
            model.load_state_dict(torch.load(checkpoint, map_location="cuda", weights_only=False)["model_state_dict"])
            actual = helper.logits(model, x).cpu().numpy()
            saved = np.load(directory / "evaluation_logits.npz")
            np.testing.assert_array_equal(saved["labels"][positions], y)
            np.testing.assert_allclose(actual, saved["logits"][positions], atol=2e-5, rtol=2e-5)
            replay[role] = float(np.max(np.abs(actual - saved["logits"][positions])))
        matched.append({"seed": seed, "initial_state_and_all_epoch_orders_match": True,
                        "cross63_combo_replay_max_error": replay})
    result = compare(args.reference, args.candidate, args.source, args.seeds, "M_{seed}", "M_{seed}")
    result.update(matching=matched, label_smoothing_epsilon=.01, source_sha256=base.sha(Path(__file__)),
                  prompt_sha256=base.sha(HERE / "03C_CONFIDENCE_REGULARIZATION_GOAL.txt"))
    base.write(args.output, result)
    print("REGULARIZATION_COMPARISON", result["passes"], result["relative_BCE_reduction"], flush=True)


if __name__ == "__main__":
    main()
