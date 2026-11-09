"""Verify completed variable-budget training and frozen-head diagnostics."""
import csv
import hashlib
import json
import sys
from pathlib import Path

import torch
import numpy as np
import h5py

import training

HERE = Path(__file__).resolve().parent
sys.path.append(str(HERE.parent / "2026-11-08-01-overnight"))
from verify_overnight import independent_metrics

base = training.base
PREVIOUS = Path("E:/LJQ_code/LJQ_code_experiment_storage/overnight_v1/20261108_01")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def verify(root):
    sources = {base.sha(path / "manifest.json"): path for path in (PREVIOUS / "data", PREVIOUS / "replication_data")}
    for path in root.glob("*/manifest.json"):
        if "fit_A" in read(path):
            sources[base.sha(path)] = path.parent
    tail = read(HERE.parent / "2026-11-06-01-lt-abc/2026-11-06-lt-abc-quotas.json")["tail_codes"]
    records, errors, pending = [], [], []
    hash_cache = {}
    def digest(path):
        path = Path(path)
        if path not in hash_cache:
            hash_cache[path] = base.sha(path)
        return hash_cache[path]
    for receipt_path in sorted(root.rglob("receipt.json")):
        receipt = read(receipt_path)
        full = "unique_fit_per_epoch" in receipt
        if not full and "balanced" not in receipt:
            continue
        folder = receipt_path.parent
        parent = folder.parent
        if not (parent / "status.json").exists() or read(parent / "status.json")["status"] != "completed":
            pending.append({"path": str(folder.relative_to(root)), "kind": "full" if full else "head",
                            "epochs": receipt["epochs"], "status": "training_receipt_present_group_evaluation_pending"})
            continue
        protocol = read(parent / "protocol.json")
        source = sources[protocol["manifest_sha256"]]
        manifest = read(source / "manifest.json")
        for role in ("fit", "dev"):
            assert digest(manifest[f"{role}_path"]) == manifest[f"{role}_sha256"]
        for name, expected in protocol.get("view_sha256", {}).items():
            assert digest(source / name) == expected
        history = read(folder / "history.json")
        epochs = receipt["epochs"]
        assert len(history) == epochs and receipt["updates"] == epochs * 250
        generator = torch.Generator().manual_seed(receipt["seed"])
        if receipt.get("sampling"):
            sampler = read(parent / "sampler.json")
            assert digest(parent / "sampler.json") == protocol["sampler_sha256"]
            assert digest(folder / "sample_counts.npz") == receipt["sample_counts_sha256"]
            saved_counts = np.load(folder / "sample_counts.npz")
            probabilities = torch.tensor(sampler["probabilities"], dtype=torch.float64)
            np.testing.assert_array_equal(probabilities.numpy(), saved_counts["probabilities"])
            design = read(root / "sampling_design_audit.json")["results"]
            independent_hashes = {record["probability_sha256"] for record in design.values()}
            assert hashlib.sha256(probabilities.numpy().tobytes()).hexdigest() in independent_hashes
            expected_orders = []
            with h5py.File(source / "fit_gap10.mat", "r") as handle:
                positions10 = np.asarray(handle["positions"]).reshape(-1).astype(int)
            with h5py.File(source / "fit_gap15.mat", "r") as handle:
                positions15 = np.asarray(handle["positions"]).reshape(-1).astype(int)
            for index in range(epochs):
                order = torch.multinomial(probabilities, 8000, replacement=True, generator=generator)
                expected_orders.append(hashlib.sha256(order.numpy().tobytes()).hexdigest())
                counts = torch.bincount(order, minlength=8000).numpy()
                np.testing.assert_array_equal(counts, saved_counts["counts"][index])
                assert np.count_nonzero(counts) == receipt["unique_fit_per_epoch"][index]
                epoch = index + 1
                selected10 = positions10[(positions10 + epoch) % 2 == 0]
                selected15 = positions15[((positions15 + epoch) % 2 == 0) & (((epoch + positions15 % 2) // 2) % 2 == 0)]
                assert counts[selected10].sum() == history[index]["augmented_presentations"]
                assert counts[selected15].sum() == history[index]["gap15_presentations"]
        else:
            expected_orders = [hashlib.sha256(torch.randperm(8000, generator=generator).numpy().tobytes()).hexdigest()
                               for _ in range(epochs)]
        assert expected_orders == receipt["order_sha256"]
        checkpoint = folder / ("best_BCE.pth" if full else "best_head.pth")
        assert digest(checkpoint) == receipt["checkpoint_sha256"]
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        losses = [row["early_BCE"] for row in history]
        first_epoch = 1
        if "initial_early_BCE" in receipt:
            losses = [receipt["initial_early_BCE"]] + losses
            first_epoch = 0
        best_epoch = int(np.argmin(losses)) + first_epoch
        assert state["epoch"] == best_epoch
        np.testing.assert_allclose(state["early_BCE"], min(losses), rtol=0, atol=1e-12)
        if isinstance(protocol["source_sha256"], dict):
            for path, expected in protocol["source_sha256"].items():
                assert digest(path) == expected, path
        else:
            allowed_sources = [HERE.parent / "2026-11-08-01-overnight/run_generic_heads.py", HERE / "run_guarded_heads.py"]
            assert protocol["source_sha256"] in {digest(path) for path in allowed_sources}
        if full:
            assert receipt["protocol_sha256"] == digest(parent / "protocol.json")
            assert receipt.get("fit_identity_count", receipt["unique_fit_per_epoch"]) == 8000
            for row in history:
                expected_lr = 1e-6 + (1e-4 - 1e-6) * (1 + np.cos(np.pi * (row["epoch"] - 1) / (epochs - 1))) / 2
                np.testing.assert_allclose(row["lr"], expected_lr, rtol=1e-12, atol=1e-14)
            metric_path = parent / "evaluation_metrics.csv"
            metric_variant = "raw"
            arm = receipt["arm"]
        else:
            metric_path = parent / "metrics.csv"
            metric_variant = "balanced" if receipt["balanced"] else "hard"
            arm = folder.name.split("_")[0]
            pairing = read(source / "fit_pairing_receipt.json")
            assert sum(row["weak_presentations"] for row in history) == 20 * pairing["view_counts"][0]
            assert sum(row["gap15_presentations"] for row in history) == 10 * pairing["view_counts"][1]
        saved = np.load(folder / "evaluation_logits.npz")
        calibration = read(folder / "calibration.json")
        actual = independent_metrics(saved["logits"], saved["labels"], calibration["thresholds"]["raw"], tail)
        with metric_path.open(encoding="utf-8-sig") as handle:
            expected = next(row for row in csv.DictReader(handle) if int(row["seed"]) == receipt["seed"]
                            and row["arm"] == arm and row["variant"] == metric_variant)
        for key, value in actual.items():
            error = abs(value - float(expected[key]))
            assert error < 1e-12, (folder, key)
            errors.append(error)
        records.append({"path": str(folder.relative_to(root)), "kind": "full" if full else "head",
            "epochs": epochs, "updates": epochs * 250, "sample_presentations": epochs * 8000,
            "best_epoch": best_epoch, "source": str(source), "order_and_checkpoint_verified": True})
    ap_checks = []
    for parent in sorted(root.glob("ap_selection_e*")):
        if not (parent / "status.json").exists() or read(parent / "status.json")["status"] != "completed":
            continue
        protocol = read(parent / "protocol.json")
        assert digest(HERE / "evaluate_ap_checkpoint.py") == protocol["source_sha256"]
        with (parent / "evaluation_metrics.csv").open(encoding="utf-8-sig") as handle:
            expected_rows = list(csv.DictReader(handle))
        for name, selected in protocol["selected"].items():
            folder = parent / name
            assert digest(folder / "selected_model.pth") == selected["sha256"]
            saved = np.load(folder / "evaluation_logits.npz")
            calibration = read(folder / "calibration.json")
            seed = int(name.split("_")[1])
            for variant in ("raw", "temperature"):
                z = saved["logits"] / (calibration["T"] if variant == "temperature" else 1)
                values = independent_metrics(z, saved["labels"], calibration["thresholds"][variant], tail)
                expected = next(row for row in expected_rows if int(row["seed"]) == seed and row["variant"] == variant)
                for key, value in values.items():
                    error = abs(value - float(expected[key]))
                    assert error < 1e-12
                    errors.append(error)
            ap_checks.append({"path": str(folder.relative_to(root)), "selected_epoch": selected["epoch"]})
    result = {"status": "passed", "completed_runs": records, "AP_checkpoint_checks": ap_checks, "scalar_metrics": len(errors),
        "max_metric_error": max(errors, default=0), "full_runs": sum(row["kind"] == "full" for row in records + pending),
        "evaluated_full_runs": sum(row["kind"] == "full" for row in records), "pending_group_evaluation": pending,
        "head_fits": sum(row["kind"] == "head" for row in records + pending), "source_sha256": base.sha(Path(__file__))}
    base.write(root / "verification.json", result)
    return result


if __name__ == "__main__":
    root = Path("E:/LJQ_code/LJQ_code_experiment_storage/daytime_v1/20261109_01")
    result = verify(root)
    print({key: value for key, value in result.items() if key != "completed_runs"}, flush=True)
