"""Select a bounded residual amplitude using check identities only."""
import copy
from common import *


def main():
    setup()
    scales = [0, .125, .25, .5, .75, 1]
    for split in CONFIG["splits"]:
        assert read(ROOT / split / "readout_status.json")["status"] == "completed"
        for seed in CONFIG["seeds"]:
            with np.load(ROOT / split / "cache" / f"M_{seed}" / "check.npz") as data:
                check = dict(data)
            with np.load(ROOT / split / "cache" / f"M_{seed}" / "calibration.npz") as data:
                calibration_records = dict(data)
            for variant in CONFIG["variants"]:
                check_deadline()
                source = ROOT / split / "readouts" / f"{variant}_{seed}"
                model, state = load_bundle(source / "model.pth")
                calibration = read(source / "calibration.json")
                thresholds = calibration["original_thresholds"]
                original_z = check["logits"].astype(float)
                residual = cached_logits(model, check) - original_z
                reference = evaluate(check, original_z, thresholds)
                best_loss = reference["raw_bce"]
                best_scale = 0
                rows = []
                for scale in scales:
                    metrics = evaluate(check, original_z + scale*residual, thresholds)
                    guards = preserves_reference(metrics, reference, ["SawFM", "MFT", "ISRJ"])
                    allowed = guards and metrics["exact"] >= reference["exact"] - .002 - 1e-12
                    rows.append({"scale": scale, "BCE": metrics["raw_bce"], "exact": metrics["exact"], "allowed": bool(allowed)})
                    if allowed and metrics["raw_bce"] < best_loss - 1e-12:
                        best_loss = metrics["raw_bce"]
                        best_scale = scale
                with torch.no_grad():
                    for head in model.heads:
                        head.output.weight.mul_(best_scale)
                        head.output.bias.mul_(best_scale)
                z = cached_logits(model, calibration_records)
                new_thresholds = helper.fit_thresholds(z, calibration_records["target"])
                output = source.parent / f"{variant}_scaled_{seed}"
                output.mkdir(exist_ok=True)
                info = {"scale": best_scale, "grid": rows, "source_model_sha256": base.sha(source / "model.pth"),
                        "script_sha256": base.sha(Path(__file__)), "goal_sha256": base.sha(HERE / "03A_RESIDUAL_SCALE_GOAL.txt"),
                        "selection_role": "early_stop", "time": stamp()}
                model.save(output / "model.pth", new_thresholds, info)
                write(output / "calibration.json", {"thresholds": new_thresholds, "original_thresholds": thresholds})
                write(output / "receipt.json", {**info, "checkpoint_sha256": base.sha(output / "model.pth")})
                print("RESIDUAL_SCALE", split, seed, variant, best_scale, flush=True)
                del model
    write(ROOT / "scaled_status.json", {"status": "completed", "time": stamp()})


if __name__ == "__main__":
    main()
