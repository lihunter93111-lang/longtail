"""Freeze complete candidate sets, evaluate new identities, and compare pairs."""
import argparse
from scipy.stats import t as student_t
from common import *


def interval(values):
    values = np.asarray(values)
    mean = float(values.mean())
    radius = float(student_t.ppf(.975, len(values)-1) * values.std(ddof=1) / np.sqrt(len(values)))
    return [mean-radius, mean+radius]


def identity_interval(differences, labels, seed=20261110):
    codes = labels.astype(int) @ (1 << np.arange(6))
    values = np.asarray(differences).mean(axis=0)
    grouped = np.stack([values[codes == code] for code in range(1, 64)])
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(50):
        indices = rng.integers(0, grouped.shape[1], size=(100, 63, grouped.shape[1]))
        means.extend(np.take_along_axis(grouped[None, :, :], indices, axis=2).mean(axis=(1, 2)).tolist())
    return np.quantile(means, [.025, .975]).tolist()


def pair_comparison(candidate, reference, root, records, variant="calibrated"):
    metrics = {}
    exact_rows = []
    bce_rows = []
    for group in (candidate, reference):
        metrics[group] = [read(root / f"{group}_{seed}" / "metrics.json")[variant] for seed in CONFIG["seeds"]]
    exact_gain = [a["exact"] - b["exact"] for a, b in zip(metrics[candidate], metrics[reference])]
    bce_gain = [b["raw_bce"] - a["raw_bce"] for a, b in zip(metrics[candidate], metrics[reference])]
    for seed in CONFIG["seeds"]:
        predictions = []
        losses = []
        for group in (candidate, reference):
            with np.load(root / f"{group}_{seed}" / "predictions.npz") as values:
                np.testing.assert_array_equal(values["identity"], records["identity"])
                thresholds = values["thresholds"] if variant == "calibrated" else values["original_thresholds"]
                z = values["logits"].astype(float)
                y = values["target"].astype(bool)
                predictions.append(((expit(z) >= thresholds) == y).all(1))
                losses.append((np.logaddexp(0, z) - y*z).mean(1))
        exact_rows.append(predictions[0].astype(float) - predictions[1])
        bce_rows.append(losses[1] - losses[0])
    per_label = []
    guards = []
    for name in LABELS:
        candidate_rows = [m["per_label"][name] for m in metrics[candidate]]
        reference_rows = [m["per_label"][name] for m in metrics[reference]]
        row = {"label": name}
        for key in ("negative_fpr", "recall", "aupr", "weak_recall", "weak_aupr"):
            values = [a[key]-b[key] for a,b in zip(candidate_rows,reference_rows) if a.get(key) is not None and b.get(key) is not None]
            row[key + "_delta"] = float(np.mean(values)) if values else None
            row[key + "_per_seed"] = values
        guards.append(row["negative_fpr_delta"] <= .005 + 1e-12)
        if name in ("SawFM", "MFT", "ISRJ") and row["weak_recall_delta"] is not None:
            guards.append(row["weak_recall_delta"] >= -.01 - 1e-12)
        per_label.append(row)
    bce_reference = np.mean([m["raw_bce"] for m in metrics[reference]])
    exact_ci = interval(exact_gain)
    checks = {"exact_gain_ge0_5pp": np.mean(exact_gain) >= .005,
              "positive_seeds_ge4": sum(v > 0 for v in exact_gain) >= 4,
              "seed_exact_CI_lower_positive": exact_ci[0] > 0,
              "BCE_degradation_le2pct": -np.mean(bce_gain) <= .02*bce_reference,
              "mean_label_guards": all(guards)}
    return {"candidate": candidate, "reference": reference, "variant": variant,
            "exact_gain": float(np.mean(exact_gain)), "exact_per_seed": exact_gain,
            "exact_seed_t95": exact_ci, "exact_identity_bootstrap95": identity_interval(exact_rows, records["target"]),
            "BCE_gain": float(np.mean(bce_gain)), "BCE_seed_t95": interval(bce_gain),
            "BCE_identity_bootstrap95": identity_interval(bce_rows, records["target"]),
            "checks": {k: bool(v) for k,v in checks.items()}, "passes": bool(all(checks.values())),
            "per_label": per_label, "bootstrap_replicates": 5000}


def evaluate_split(split, role, variants):
    manifest = read(ROOT / split / role / "manifest.json")
    assert base.sha(manifest["path"]) == manifest["sha256"]
    inputs, labels = base.load(manifest["path"], manifest["indices"])
    records = metadata(manifest["path"], manifest["indices"])
    np.testing.assert_array_equal(labels, records["target"])
    destination = ROOT / split / f"{role}_results"
    destination.mkdir(exist_ok=True)
    groups = ["baseline", *variants]
    models_root = ROOT / split / "readouts"
    models = []
    for group in groups:
        for seed in CONFIG["seeds"]:
            folder = models_root / f"{group}_{seed}"
            assert (folder / "model.pth").exists()
            models.append({"group": group, "seed": seed, "path": str(folder / "model.pth"),
                           "sha256": base.sha(folder / "model.pth"), "calibration_sha256": base.sha(folder / "calibration.json")})
    freeze = {"models": models, "time": stamp(), "role": role,
              "all_models_and_calibrations_frozen": True, "manifest_sha256": base.sha(ROOT / split / role / "manifest.json")}
    freeze_path = destination / "freeze.json"
    if freeze_path.exists():
        assert read(freeze_path)["models"] == models
    else:
        write(freeze_path, freeze)
    write(destination / "evaluation_access.json", {"time": stamp(), "freeze_sha256": base.sha(freeze_path), "role": role})
    summary = []
    for record in models:
        check_deadline("evaluation_cutoff")
        name = f'{record["group"]}_{record["seed"]}'
        output = destination / name
        output.mkdir(exist_ok=True)
        model, state = load_bundle(record["path"])
        calibration = read(models_root / name / "calibration.json")
        logits = base.predict(model, inputs, "cuda").astype(float)
        metrics = {"calibrated": evaluate(records, logits, calibration["thresholds"]),
                   "fixed_reference": evaluate(records, logits, calibration["original_thresholds"])}
        np.savez(output / "predictions.npz", logits=logits, target=labels, identity=records["identity"],
                 thresholds=calibration["thresholds"], original_thresholds=calibration["original_thresholds"])
        write(output / "metrics.json", metrics)
        summary.append({"group": record["group"], "seed": record["seed"],
                        **{k: metrics["calibrated"][k] for k in ("exact", "raw_bce", "macro_aupr", "macro_f1", "tail_exact", "ECE")}})
        del model
        print("EVALUATED", split, role, name, metrics["calibrated"]["exact"], flush=True)
    helper.table(destination / "metrics.csv", summary)
    comparisons = []
    pairs = [(name, "baseline") for name in variants]
    if "segment" in variants:
        pairs += [("segment", name) for name in variants if name != "segment"]
    for candidate, reference in pairs:
        for variant in ("calibrated", "fixed_reference"):
            comparisons.append(pair_comparison(candidate, reference, destination, records, variant))
    means = {group: {key: float(np.mean([row[key] for row in summary if row["group"] == group]))
                    for key in ("exact", "raw_bce", "macro_aupr", "macro_f1", "tail_exact", "ECE")} for group in groups}
    write(destination / "summary.json", {"means": means, "comparisons": comparisons, "role": role, "completed_utc": stamp()})
    print("COMPARISONS_COMPLETE", split, role, json.dumps(means), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=CONFIG["splits"], required=True)
    parser.add_argument("--role", choices=["development", "confirmation"], default="development")
    parser.add_argument("--variants", nargs="+", default=CONFIG["variants"])
    args = parser.parse_args()
    setup()
    if args.role == "confirmation":
        freeze = read(ROOT / "final_freeze.json")
        assert freeze["all_models_and_calibrations_frozen"]
        assert args.variants == freeze["variants"]
    else:
        assert read(ROOT / args.split / "readout_status.json")["status"] == "completed"
    evaluate_split(args.split, args.role, args.variants)


if __name__ == "__main__":
    main()
