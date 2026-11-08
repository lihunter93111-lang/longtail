"""Generate Python/PCG64 AMA pools with explicit independent roles."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "projects/CJ_Imbalance_0716"))
from cjlt.data.signal_simulator import Simulator, REGISTRY_IDS, rng, substream_seed
from cjlt.models.classwise_fusion import AMA_CLASSES


def compose_v3(simulator, active_codes, sample_seed, jnr_range=(10.0, 30.0)):
    components, jnrs, component_seeds = [], [], []
    for code in active_codes:
        seed = substream_seed(sample_seed, 1, REGISTRY_IDS[code])
        stream = rng(seed)
        jnr = float(stream.uniform(*jnr_range))
        raw, _ = getattr(simulator, "_" + code)(stream, simulator.physics[code])
        power = float(np.mean(np.abs(raw) ** 2))
        if not np.isfinite(power) or power <= 0:
            raise ValueError("Invalid component power")
        components.append(raw / np.sqrt(power) * np.sqrt(simulator.radar["noise_power"]) * 10 ** (jnr / 20))
        jnrs.append(jnr)
        component_seeds.append(seed)
    noise_seed = substream_seed(sample_seed, 2)
    stream = rng(noise_seed)
    noise = np.sqrt(simulator.radar["noise_power"] / 2) * (
        stream.standard_normal(simulator.length) + 1j * stream.standard_normal(simulator.length))
    clean = np.stack(components)
    power = 10 * np.log10(np.mean(np.abs(clean) ** 2, axis=1))
    return dict(iq=clean.sum(0) + noise, jnr=np.asarray(jnrs), power=power,
                relative=power - power.max(), noise_seed=noise_seed,
                component_seeds=component_seeds)


def write_pool(path, codes, base_seed):
    simulator = Simulator()
    count = len(codes)
    labels = ((np.asarray(codes)[:, None] >> np.arange(6)) & 1).astype(np.uint8)
    with h5py.File(path, "x") as stream:
        stream.attrs.update(classes_json=json.dumps(AMA_CLASSES), power_truth_schema="per-label-v1",
                            weak_gap_db=10.0, noise_power_db=0.0, generator="Python-PCG64-v3-equations",
                            physics_json=json.dumps(simulator.physics), component_jnr_db=json.dumps([10.0, 30.0]))
        signals = stream.create_dataset("iq_complex", (simulator.length, count), dtype="complex64",
                                        chunks=(simulator.length, 1), compression="lzf")
        stream.create_dataset("label_selected", data=labels.T)
        stream.create_dataset("sample_seed", data=np.arange(base_seed, base_seed + count, dtype=np.int64)[None])
        noise_ids = stream.create_dataset("receiver_noise_seed", (1, count), dtype="int64")
        fields = {key: stream.create_dataset(key, (6, count), dtype="float64", fillvalue=np.nan)
                  for key in ("component_effective_JNR_dB", "component_measured_power_dB", "relative_power_dB", "component_seeds")}
        weak = stream.create_dataset("weak_component_mask", (6, count), dtype="uint8")
        for row in range(count):
            active = np.flatnonzero(labels[row])
            values = compose_v3(simulator, [AMA_CLASSES[column] for column in active], base_seed + row)
            signals[:, row] = values["iq"]
            noise_ids[0, row] = values["noise_seed"]
            for key, value in zip(fields, (values["jnr"], values["power"], values["relative"], values["component_seeds"])):
                fields[key][active, row] = value
            weak[active, row] = (len(active) > 1) & (values["relative"] <= -10 + 1e-6)
    return labels.sum(0).astype(int).tolist()


def generate_dataset(output, train_counts, calibration=128, check=32, report=128, base_seed=1):
    output = Path(output)
    counts = np.asarray(train_counts, dtype=int)
    if counts.shape != (63,) or (counts <= 0).any() or min(calibration, check, report) < 1:
        raise ValueError("Declare positive quotas for all 63 combinations and every role")
    fit_codes = np.repeat(np.arange(1, 64), counts)
    per_combination = calibration + check + report
    dev_codes = np.repeat(np.arange(1, 64), per_combination)
    if base_seed < 0 or base_seed + len(fit_codes) + len(dev_codes) >= 2 ** 32:
        raise ValueError("Identity seeds must stay inside one PCG64 identity domain")
    output.mkdir(parents=True, exist_ok=False)
    fit_labels = write_pool(output / "train.h5", fit_codes, base_seed)
    write_pool(output / "development.h5", dev_codes, base_seed + len(fit_codes))
    roles = dict(fit=list(range(len(fit_codes))), calibration=[], check=[], report=[])
    for code in range(63):
        rows = np.arange(code * per_combination, (code + 1) * per_combination)
        roles["calibration"].extend(rows[:calibration].tolist())
        roles["check"].extend(rows[calibration:calibration + check].tolist())
        roles["report"].extend(rows[calibration + check:].tolist())
    (output / "split.json").write_text(json.dumps(roles, indent=2) + "\n", encoding="utf-8")
    files = {}
    for path in (output / "train.h5", output / "development.h5", output / "split.json"):
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1 << 20), b""):
                digest.update(chunk)
        files[path.name] = dict(bytes=path.stat().st_size, sha256=digest.hexdigest())
    manifest = dict(classes=list(AMA_CLASSES), base_seed=base_seed, train_counts=counts.tolist(),
                    train_label_counts=fit_labels, files=files,
                    generator="Python PCG64 physical port; not MATLAB random-stream parity")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--base-seed", required=True, type=int)
    parser.add_argument("--calibration", type=int, default=128)
    parser.add_argument("--check", type=int, default=32)
    parser.add_argument("--report", type=int, default=128)
    args = parser.parse_args()
    reference = json.loads(args.reference_manifest.read_text(encoding="utf-8"))
    if tuple(reference["classes"]) != AMA_CLASSES:
        raise ValueError("Reference manifest must declare the AMA label contract")
    counts = reference["files"]["train"]["combination_counts"]
    print(json.dumps(generate_dataset(args.output, counts, args.calibration, args.check,
                                      args.report, args.base_seed), ensure_ascii=False))


if __name__ == "__main__":
    main()
