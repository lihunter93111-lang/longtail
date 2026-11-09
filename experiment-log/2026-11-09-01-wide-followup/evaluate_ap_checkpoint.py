"""Frozen early-AP checkpoint sensitivity for completed budget runs."""
import argparse
import json
import shutil
from pathlib import Path

import torch
import numpy as np

import training

base = training.base
helper = training.previous
HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert read(args.models / "status.json")["status"] == "completed"
    protocol = read(args.models / "protocol.json")
    manifest = read(args.source / "manifest.json")
    assert protocol["manifest_sha256"] == base.sha(args.source / "manifest.json")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    selected = {}
    for seed in protocol["seeds"]:
        source = args.models / f"M_{seed}"
        receipt = read(source / "receipt.json")
        path = source / "best_AP.pth"
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        assert checkpoint["epoch"] == receipt["best_AP_epoch"]
        assert checkpoint["early_macro_AP"] == receipt["best_early_AP"]
        selected[f"M_{seed}"] = {"path": str(path), "sha256": base.sha(path), "epoch": checkpoint["epoch"]}
    experiment = {"role": "adaptive_development_checkpoint_sensitivity", "selection": "minimum epoch at maximum early_stop macroAP",
        "model": "dfms_iq", "width": 32, "seeds": protocol["seeds"], "epochs": protocol["epochs"],
        "manifest_sha256": protocol["manifest_sha256"], "selected": selected,
        "source_sha256": base.sha(Path(__file__)), "prompt_sha256": base.sha(HERE / "02B_AP_CHECKPOINT_GOAL.txt")}
    base.write(args.output / "protocol.json", experiment)
    x, y = base.load(manifest["dev_path"], manifest["calibration"])
    x = x.cuda()
    frozen = {}
    for name, settings in selected.items():
        folder = args.output / name
        folder.mkdir(exist_ok=True)
        shutil.copy2(settings["path"], folder / "selected_model.pth")
        assert base.sha(folder / "selected_model.pth") == settings["sha256"]
        model = base.build_backbone("dfms_iq", num_classes=6, width=32, dropout=.2).cuda()
        model.load_state_dict(torch.load(folder / "selected_model.pth", map_location="cuda", weights_only=False)["model_state_dict"])
        z = helper.logits(model, x).cpu().numpy().astype(np.float64)
        temperature = helper.fit_temperature(z, y)
        calibration = {**temperature, "thresholds": {"raw": helper.fit_thresholds(z, y),
            "temperature": helper.fit_thresholds(z / temperature["T"], y)}, "checkpoint_sha256": settings["sha256"]}
        base.write(folder / "calibration.json", calibration)
        np.savez_compressed(folder / "calibration_logits.npz", logits=z, labels=y)
        frozen[name] = calibration
    base.write(args.output / "freeze.json", {"models": frozen, "all_models_complete": True,
        "evaluation_role": "adaptive_development", "protocol_sha256": base.sha(args.output / "protocol.json")})
    x, y = base.load(manifest["dev_path"], manifest["development"])
    x = x.cuda()
    quota = read(HERE.parent / "2026-11-06-01-lt-abc/2026-11-06-lt-abc-quotas.json")
    rows, labels = [], []
    for name, calibration in frozen.items():
        folder = args.output / name
        model = base.build_backbone("dfms_iq", num_classes=6, width=32, dropout=.2).cuda()
        model.load_state_dict(torch.load(folder / "selected_model.pth", map_location="cuda", weights_only=False)["model_state_dict"])
        z = helper.logits(model, x).cpu().numpy().astype(np.float64)
        np.savez_compressed(folder / "evaluation_logits.npz", logits=z, labels=y,
            indices=np.array(manifest["development"]), codes=y.astype(int) @ (1 << np.arange(6)))
        for variant in ("raw", "temperature"):
            values = z / (calibration["T"] if variant == "temperature" else 1)
            metrics, per_label, _, _ = helper.metrics(values, y, calibration["thresholds"][variant], quota["tail_codes"])
            tags = {"arm": "M", "seed": int(name.split("_")[1]), "variant": variant, "selected_epoch": selected[name]["epoch"]}
            rows.append({**tags, **metrics})
            labels.extend({**tags, **r} for r in per_label)
    helper.table(args.output / "evaluation_metrics.csv", rows)
    helper.table(args.output / "evaluation_labels.csv", labels)
    base.write(args.output / "status.json", {"status": "completed", "models": len(frozen), "new_training_runs": 0})
    print("AP_CHECKPOINT_DIAGNOSTIC_COMPLETE", protocol["epochs"], flush=True)


if __name__ == "__main__":
    main()
