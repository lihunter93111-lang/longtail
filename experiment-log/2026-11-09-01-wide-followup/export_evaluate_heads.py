"""Compose wide frozen heads, replay normal inference, and evaluate power views."""
import argparse
import importlib.util
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import numpy as np

import training

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / "2026-11-08-01-overnight"
sys.path.append(str(OLD))
from run_stage2 import read

helper = training.previous
base = training.base


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--backbones", type=Path, required=True)
    parser.add_argument("--heads", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", choices=["hard", "balanced"], required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    parent = read(args.backbones / "protocol.json")
    args.seeds = parent["seeds"]
    args.arms = ["M"]
    width = parent["width"]
    manifest = read(args.source / "manifest.json")
    head_freeze = read(args.heads / "freeze.json")
    assert read(args.heads / "protocol.json")["manifest_sha256"] == base.sha(args.source / "manifest.json")
    args.output.mkdir(parents=True, exist_ok=True)
    protocol = {"operation": "compose and replay already frozen wide classifier heads", "model": "dfms_iq",
        "width": width, "seeds": args.seeds, "coverage": "M", "head_variant": args.variant,
        "manifest_sha256": base.sha(args.source / "manifest.json"), "source_sha256": base.sha(Path(__file__)),
        "backbone_freeze_sha256": base.sha(args.backbones / "freeze.json"),
        "head_freeze_sha256": base.sha(args.heads / "freeze.json"), "role": "adaptive_development"}
    base.write(args.output / "protocol.json", protocol)
    models = {}
    replay = []
    replay_positions = np.linspace(0, len(manifest["development"]) - 1, 64, dtype=int)
    x, y = base.load(manifest["dev_path"], [manifest["development"][i] for i in replay_positions])
    x = x.cuda()
    for seed in args.seeds:
        name = f"M_{seed}"
        original = args.backbones / name
        adapted = args.heads / f"{name}_{args.variant}"
        folder = args.output / name
        folder.mkdir(exist_ok=True)
        assert base.sha(adapted / "best_head.pth") == head_freeze["models"][adapted.name]["checkpoint_sha256"]
        checkpoint = torch.load(original / "best_BCE.pth", map_location="cpu", weights_only=False)
        head = torch.load(adapted / "best_head.pth", map_location="cpu", weights_only=False)
        state = checkpoint["model_state_dict"]
        for key, value in head["head_state_dict"].items():
            assert state["classifier." + key].shape == value.shape
            state["classifier." + key] = value
        helper.atomic_torch_save({"model_state_dict": state, "backbone_epoch": checkpoint["epoch"],
                                  "head_epoch": head["epoch"]}, folder / "best_BCE.pth")
        model = base.build_backbone("dfms_iq", num_classes=6, width=width, dropout=.2).cuda()
        model.load_state_dict(state)
        actual = helper.logits(model, x).cpu().numpy()
        saved = np.load(adapted / "evaluation_logits.npz")
        np.testing.assert_array_equal(saved["labels"][replay_positions], y)
        error = float(np.max(np.abs(actual - saved["logits"][replay_positions])))
        np.testing.assert_allclose(actual, saved["logits"][replay_positions], atol=2e-5, rtol=2e-5)
        replay.append({"seed": seed, "samples": 64, "maximum_logit_difference": error})
        calibration = read(adapted / "calibration.json")
        calibration["checkpoint_sha256"] = base.sha(folder / "best_BCE.pth")
        calibration_saved = np.load(adapted / "calibration_logits.npz")
        calibration["thresholds"]["temperature"] = helper.fit_thresholds(
            calibration_saved["logits"] / calibration["T"], calibration_saved["labels"])
        base.write(folder / "calibration.json", calibration)
        shutil.copy2(adapted / "calibration_logits.npz", folder / "calibration_logits.npz")
        np.savez_compressed(folder / "evaluation_logits.npz", logits=saved["logits"], labels=saved["labels"],
            indices=np.array(manifest["development"]), codes=saved["labels"].astype(int) @ (1 << np.arange(6)))
        base.write(folder / "composition_receipt.json", {"backbone_path": str(original / "best_BCE.pth"),
            "backbone_sha256": base.sha(original / "best_BCE.pth"), "head_path": str(adapted / "best_head.pth"),
            "head_sha256": base.sha(adapted / "best_head.pth"), "composed_sha256": calibration["checkpoint_sha256"]})
        models[name] = calibration
    freeze = {"models": models, "protocol_sha256": base.sha(args.output / "protocol.json"),
              "all_models_complete": True, "evaluation_role": "adaptive_development"}
    base.write(args.output / "freeze.json", freeze)
    base.write(args.output / "replay_verification.json", {"status": "passed", "models": replay})
    quotas = read(HERE.parent / "2026-11-06-01-lt-abc/2026-11-06-lt-abc-quotas.json")
    rows, labels, combinations = [], [], []
    for name, calibration in models.items():
        saved = np.load(args.output / name / "evaluation_logits.npz")
        for variant in ("raw", "temperature"):
            z = saved["logits"] / (calibration["T"] if variant == "temperature" else 1)
            values, per_label, per_combo, _ = helper.metrics(z, saved["labels"], calibration["thresholds"][variant], quotas["tail_codes"])
            tags = {"arm": "M", "seed": int(name.split("_")[1]), "variant": variant}
            rows.append({**tags, **values})
            labels.extend({**tags, **row} for row in per_label)
            combinations.extend({**tags, **row} for row in per_combo)
    helper.table(args.output / "evaluation_metrics.csv", rows)
    helper.table(args.output / "evaluation_labels.csv", labels)
    helper.table(args.output / "evaluation_combinations.csv", combinations)
    spec = importlib.util.spec_from_file_location("head_power_analysis_today", helper.HERE / "analyze_goal.py")
    analysis = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(analysis)
    def factory(model_name, num_classes=6, width=16, dropout=.2):
        return base.build_backbone("dfms_iq", num_classes=num_classes, width=parent["width"], dropout=dropout)
    analysis_base = SimpleNamespace(**{**vars(base), "build_backbone": factory})
    analysis_helper = SimpleNamespace(**{**vars(helper), "base": analysis_base})
    analysis.natural_slices(args, manifest, freeze, analysis_helper)
    analysis.paired_evaluation(args, manifest, freeze, analysis_helper)
    helper.table(args.output / "summary_metrics.csv", analysis.summarize(rows, ("arm", "variant"),
        ("BCE", "macro_AP", "f1_exact", "tail_exact", "ECE", "macro_F1")))
    base.write(args.output / "status.json", {"status": "completed", "models": len(models), "replay_passed": True})
    print("WIDE_HEAD_EXPORT_EVALUATED", args.variant, flush=True)


if __name__ == "__main__":
    main()
