"""Independent arithmetic, training coverage, and provenance verification."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import torch
import numpy as np
from scipy.special import expit
from sklearn.metrics import average_precision_score

import training

base = training.base
HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def independent_metrics(z, y, thresholds, tail):
    loss = np.maximum(z, 0) - z * y + np.log1p(np.exp(-np.abs(z)))
    p = expit(z)
    prediction = p >= np.asarray(thresholds)
    truth = y.astype(bool)
    exact = np.all(prediction == truth, axis=1)
    codes = y.astype(int) @ (1 << np.arange(6))
    tp = (prediction & truth).sum(axis=0)
    fp = (prediction & ~truth).sum(axis=0)
    fn = (~prediction & truth).sum(axis=0)
    return {"BCE": float(loss.mean()), "Brier": float(np.square(p - y).mean()),
        "macro_AP": float(np.mean([average_precision_score(y[:, j], p[:, j]) for j in range(6)])),
        "f1_exact": float(exact.mean()), "fixed05_exact": float(np.all((z >= 0) == truth, axis=1).mean()),
        "macro_F1": float(np.mean(2 * tp / np.maximum(2 * tp + fp + fn, 1))),
        "tail_exact": float(np.mean([exact[codes == code].mean() for code in tail]))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--expected-full-runs", type=int, default=60)
    parser.add_argument("--completed-groups-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    quota = read(HERE.parent / "2026-11-06-01-lt-abc/2026-11-06-lt-abc-quotas.json")
    sources = {base.sha(path / "manifest.json"): path for path in
               (root / "data", root / "balanced_data", root / "replication_data")}
    hash_cache = {}

    def digest(path):
        path = Path(path)
        if path not in hash_cache:
            hash_cache[path] = base.sha(path)
        return hash_cache[path]

    receipt_paths = [path for path in root.rglob("receipt.json") if "unique_fit_per_epoch" in read(path)]
    if args.completed_groups_only:
        receipt_paths = [path for path in receipt_paths
                         if read(path.parent.parent / "status.json")["status"] == "completed"]
    assert len(receipt_paths) == args.expected_full_runs, len(receipt_paths)
    checks, differences = [], []
    for receipt_path in sorted(receipt_paths):
        folder = receipt_path.parent
        models = folder.parent
        receipt = read(receipt_path)
        protocol = read(models / "protocol.json")
        source = sources[protocol["manifest_sha256"]]
        manifest = read(source / "manifest.json")
        assert receipt["protocol_sha256"] == digest(models / "protocol.json")
        assert receipt["checkpoint_sha256"] == digest(folder / "best_BCE.pth")
        for path, expected in protocol["source_sha256"].items():
            assert digest(path) == expected, path
        for name, expected in protocol["view_sha256"].items():
            assert digest(source / name) == expected, name
        for role in ("fit", "dev"):
            assert digest(manifest[f"{role}_path"]) == manifest[f"{role}_sha256"]
        assert receipt["epochs"] == 40 and receipt["updates"] == 10000
        assert receipt["unique_fit_per_epoch"] == 8000
        history = read(folder / "history.json")
        assert [row["epoch"] for row in history] == list(range(1, 41))
        generator = torch.Generator().manual_seed(receipt["seed"])
        expected_orders = [hashlib.sha256(torch.randperm(8000, generator=generator).numpy().tobytes()).hexdigest()
                           for _ in range(40)]
        assert receipt["order_sha256"] == expected_orders
        early_losses = np.array([row["early_BCE"] for row in history])
        best = int(early_losses.argmin())
        assert receipt["best_BCE_epoch"] == best + 1
        assert receipt["best_early_BCE"] == early_losses[best]
        for row in history:
            expected_lr = 1e-6 + (1e-4 - 1e-6) * (1 + np.cos(np.pi * (row["epoch"] - 1) / 39)) / 2
            np.testing.assert_allclose(row["lr"], expected_lr, rtol=1e-14)
            assert np.isfinite(row["train_objective"]) and np.isfinite(row["early_BCE"])
            assert row["train_objective"] == row["train_hard_BCE"]
        pairing = read(source / "fit_pairing_receipt.json")
        augmented = sum(row["augmented_presentations"] for row in history)
        gap15 = sum(row["gap15_presentations"] for row in history)
        assert augmented == (0 if receipt["arm"] == "B" else 20 * pairing["view_counts"][0])
        assert gap15 == (10 * pairing["view_counts"][1] if receipt["arm"] == "M" else 0)
        calibration = read(folder / "calibration.json")
        assert calibration["checkpoint_sha256"] == receipt["checkpoint_sha256"]
        checkpoint = torch.load(folder / "best_BCE.pth", map_location="cpu", weights_only=False)
        assert checkpoint["epoch"] == receipt["best_BCE_epoch"]
        np.testing.assert_allclose(checkpoint["early_BCE"], receipt["best_early_BCE"], atol=0, rtol=0)
        saved = np.load(folder / "evaluation_logits.npz")
        z, y = saved["logits"], saved["labels"]
        assert z.shape == y.shape == (4032, 6) and np.isfinite(z).all()
        np.testing.assert_array_equal(saved["indices"], manifest["development"])
        codes = y.astype(int) @ (1 << np.arange(6))
        np.testing.assert_array_equal(np.bincount(codes, minlength=64)[1:], np.full(63, 64))
        stored = list(csv.DictReader((models / "evaluation_metrics.csv").open(encoding="utf-8-sig")))
        for variant in ("raw", "temperature"):
            values = z if variant == "raw" else z / calibration["T"]
            actual = independent_metrics(values, y, calibration["thresholds"][variant], quota["tail_codes"])
            expected = next(row for row in stored if row["arm"] == receipt["arm"]
                            and int(row["seed"]) == receipt["seed"] and row["variant"] == variant)
            for metric, value in actual.items():
                error = abs(value - float(expected[metric]))
                assert error < 1e-12, (folder, variant, metric, error)
                differences.append(error)
        checks.append({"run": str(folder.relative_to(root)), "epochs": 40, "unique_fit_per_epoch": 8000,
            "identity_presentations": 320000, "augmented_presentations": augmented,
            "gap15_presentations": gap15, "best_BCE_epoch": best + 1,
            "source_data_checkpoint_hashes_verified": True, "all_epoch_orders_reconstructed": True})
    output = {"status": "passed", "source_sha256": base.sha(Path(__file__)), "full_network_runs": len(checks),
        "total_epochs": 40 * len(checks), "total_updates": 10000 * len(checks),
        "total_identity_presentations": 320000 * len(checks), "verified_unique_files": len(hash_cache),
        "independently_recomputed_scalar_metrics": len(differences),
        "maximum_metric_difference": max(differences), "runs": checks}
    base.write(args.output, output)
    print({key: value for key, value in output.items() if key != "runs"}, flush=True)


if __name__ == "__main__":
    main()
