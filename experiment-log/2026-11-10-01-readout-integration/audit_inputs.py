"""Bind source/model hashes and audit newly generated evaluation identities."""
import argparse
from common import *


def historical_pools():
    result = set()
    storage = ROOT.parent.parent
    for receipt in storage.rglob("generation_receipt.json"):
        record = read(receipt)
        path = Path(record.get("path", ""))
        if path.is_file() and path.suffix.lower() == ".mat":
            result.add(path.resolve())
    for filename in ("closedset_9class_train.mat", "closedset_9class_test.mat"):
        path = Path("E:/论文相关/干扰数据/复合干扰数据0716/九类/full_power_10_30dB_v1") / filename
        if path.is_file():
            result.add(path.resolve())
    return sorted(result)


def identity_arrays(path):
    with h5py.File(path, "r") as stream:
        noise = np.asarray(stream["receiver_noise_seed"]).reshape(-1).astype(np.int64)
        samples = (noise - 2 * 2246822519) % (1 << 32)
        active = np.asarray(stream["label_all"], dtype=bool)
        if "sample_seed" in stream:
            np.testing.assert_array_equal(samples, np.asarray(stream["sample_seed"]).reshape(-1))
    components = {(column + 1): (samples[active[column]] + 2246822519 + (column + 1) * 2654435761) % (1 << 32)
                  for column in range(len(active))}
    return samples, noise, components


def audit_pool(split, role, history):
    directory = ROOT / split / role
    pool = base.candidates(directory, "evaluation_pool", LABELS)
    samples, noise, components = identity_arrays(pool["path"])
    np.testing.assert_array_equal(samples, pool["seeds"])
    np.testing.assert_array_equal(noise, pool["noise"])
    registry = np.array([1, 2, 3, 4, 6, 8])
    for row, seeds in enumerate(pool["components"]):
        ids = registry[np.flatnonzero(pool["y"][row])]
        expected = (samples[row] + 2246822519 + ids * 2654435761) % (1 << 32)
        np.testing.assert_array_equal(seeds, expected)
    audit = []
    for path, historical in history.items():
        if path.resolve() == pool["path"].resolve():
            continue
        old_samples, old_noise, old_components = historical
        overlap = [len(np.intersect1d(samples, old_samples)), len(np.intersect1d(noise, old_noise))]
        component_overlap = sum(len(np.intersect1d(values, old_components.get(label, []))) for label, values in components.items())
        assert overlap == [0, 0] and component_overlap == 0, (str(path), overlap, component_overlap)
        audit.append({"path": str(path), "sample_overlap": 0, "noise_overlap": 0, "same_label_component_overlap": 0})
    indices = np.concatenate([np.flatnonzero(pool["codes"] == code) for code in range(1, 64)]).tolist()
    np.testing.assert_array_equal(np.bincount(pool["codes"], minlength=64)[1:], np.full(63, 64))
    manifest = {"path": str(pool["path"]), "sha256": pool["sha256"], "indices": indices, "role": role,
                "new_identities": True, "locked_test_used": False}
    if role == "confirmation":
        freeze = read(ROOT / "final_freeze.json")
        assert freeze["all_models_and_calibrations_frozen"]
        manifest["freeze_sha256"] = base.sha(ROOT / "final_freeze.json")
    write(directory / "manifest.json", manifest)
    write(directory / "preflight.json", {"passed": True, "identities": len(indices), "audit": audit,
          "method": "noise-derived sample/component identity mapping; all new actual component seeds verified; historical generator mapping previously audited",
          "new_generator_source_sha256": base.sha(HERE / "generate_evaluation_pool.m"), "time": stamp()})
    print("IDENTITY_AUDIT_PASSED", split, role, len(audit), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=["development", "confirmation"], default="development")
    args = parser.parse_args()
    if args.role == "development":
        sources = {str(path): base.sha(path) for path in HERE.glob("*.py")}
        sources.update({str(path): base.sha(path) for path in HERE.glob("*.m")})
        sources[str(HERE / "protocol.json")] = base.sha(HERE / "protocol.json")
        for directory in [SOURCE / "projects/CJ_Imbalance_0716/cjlt", SOURCE / "data_generation/0716/可配置组合干扰数据生成"]:
            sources.update({str(path): base.sha(path) for path in directory.rglob("*") if path.suffix in (".py", ".m")})
        for module in list(sys.modules.values()):
            path = Path(getattr(module, "__file__", "") or ".")
            if path.is_file() and path.suffix == ".py" and path.is_relative_to(SOURCE):
                sources[str(path)] = base.sha(path)
        source_lock = ROOT / "source_lock.json"
        if source_lock.exists():
            assert read(source_lock)["files"] == sources
        else:
            write(source_lock, {"files": sources, "remote_commit": CONFIG["remote_commit"], "time": stamp()})
        frozen = read(BASELINES / "final_freeze.json")
        models = [row for row in frozen["models"] if row["width"] == 32]
        assert len(models) == 10
        for row in models:
            folder = Path(row["folder"])
            assert base.sha(folder / "calibration.json") == row["calibration_sha256"]
            assert base.sha(folder / "receipt.json") == row["receipt_sha256"]
            assert base.sha(folder / "best_BCE.pth") == read(folder / "receipt.json")["checkpoint_sha256"]
        write(ROOT / "baseline_registry.json", {"models": models, "source_freeze_sha256": base.sha(BASELINES / "final_freeze.json")})
    history = {path: identity_arrays(path) for path in historical_pools()}
    for split in CONFIG["splits"]:
        audit_pool(split, args.role, history)


if __name__ == "__main__":
    main()
