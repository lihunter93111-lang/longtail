"""Explain guarded epoch-zero fallbacks using saved trained residual states."""
from common import *


def main():
    setup()
    rows = []
    for split in CONFIG["splits"]:
        for seed in CONFIG["seeds"]:
            cache = ROOT / split / "cache" / f"M_{seed}" / "check.npz"
            if not cache.exists():
                continue
            with np.load(cache) as data:
                records = dict(data)
            for variant in CONFIG["variants"]:
                folder = ROOT / split / "readouts" / f"{variant}_{seed}"
                if not (folder / "receipt.json").exists():
                    continue
                model, _ = load_bundle(folder / "model.pth")
                thresholds = read(folder / "calibration.json")["original_thresholds"]
                reference_logits = records["logits"].astype(float).copy()
                for index, stage in enumerate(model.stages):
                    reference = evaluate(records, reference_logits, thresholds)
                    recovery = torch.load(folder / f"stage_{index+1}_recovery.pth", map_location="cuda", weights_only=False)
                    trained = ResidualReadout(CONFIG["pca_components"], CONFIG["hidden"], len(stage["targets"])).cuda().eval()
                    trained.load_state_dict(recovery["head_state"])
                    with torch.no_grad():
                        features = model.transforms[index](torch.from_numpy(records[variant]).cuda())
                        logits = reference_logits.copy()
                        logits[:, stage["targets"]] += trained(features).cpu().numpy()
                    candidate = evaluate(records, logits, thresholds)
                    row = {"split": split, "seed": seed, "variant": variant, "stage": index+1,
                           "selected_epoch": recovery["best_epoch"], "last_epoch": recovery["history"][-1]["epoch"],
                           "last_exact_delta": candidate["exact"]-reference["exact"],
                           "last_BCE_delta": candidate["raw_bce"]-reference["raw_bce"], "target_deltas": {}}
                    for column in stage["targets"]:
                        name = LABELS[column]
                        a,b = candidate["per_label"][name],reference["per_label"][name]
                        row["target_deltas"][name] = {key: a[key]-b[key] if a[key] is not None else None
                            for key in ("negative_fpr", "weak_recall", "weak_aupr", "aupr")}
                    rows.append(row)
                    with torch.no_grad():
                        reference_logits[:, stage["targets"]] += model.heads[index](features).cpu().numpy()
                del model
    write(ROOT / "head_fallback_diagnostics.json", {"rows": rows, "time": stamp(), "role": "check_diagnostic"})
    first = [r for r in rows if r["variant"] == "segment" and r["stage"] == 1]
    print(json.dumps({"segment_first_stage": first}), flush=True)


if __name__ == "__main__":
    main()
