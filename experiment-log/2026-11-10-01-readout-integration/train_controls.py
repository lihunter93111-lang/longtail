"""Uniform bootstrap and full-coverage combo weighting controls."""
import argparse
import random
from types import SimpleNamespace
from common import *


def import_trainer(filename, name):
    path = SOURCE / "experiment-log/2026-11-09-01-wide-followup" / filename
    specification = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=CONFIG["splits"], required=True)
    args = parser.parse_args()
    setup()
    import training_weighted as natural
    sampling = import_trainer("training_sampling.py", "integration_sampling_trainer")
    manifest = read(BASELINES / args.split / "data/manifest.json")
    x, y = base.load(manifest["fit_path"], manifest["fit_A"])
    early_x, early_y = base.load(manifest["dev_path"], manifest["early_stop"])
    cal_x, cal_y = base.load(manifest["dev_path"], manifest["calibration"])
    data_root = BASELINES / args.split / "data"
    view_hashes = read(BASELINES / args.split / "w32/protocol.json")["views"]
    for name, digest in view_hashes.items():
        assert base.sha(data_root / name) == digest
    weak10, positions10, _ = helper.load_views(data_root / "fit_gap10.mat")
    weak15, positions15, _ = helper.load_views(data_root / "fit_gap15.mat")
    codes = y.astype(int) @ (1 << np.arange(6))
    cardinality = y.sum(1).astype(int)
    combo_counts = np.bincount(codes, minlength=64)
    cardinality_counts = np.bincount(cardinality, minlength=7)
    combos_per_cardinality = np.bincount([int(c).bit_count() for c in range(1,64)], minlength=7)
    mixture = .25
    probabilities = (1-mixture)/8000 + mixture*cardinality_counts[cardinality]/(8000*combos_per_cardinality[cardinality]*combo_counts[codes])
    probabilities /= probabilities.sum()
    np.testing.assert_allclose(np.bincount(cardinality, weights=probabilities, minlength=7), cardinality_counts/8000)
    tensors = [x, torch.from_numpy(y), early_x, torch.from_numpy(early_y), weak10, torch.from_numpy(positions10),
               weak15, torch.from_numpy(positions15)]
    x, y_tensor, early_x, early_y_tensor, weak10, positions10, weak15, positions15 = [v.cuda() for v in tensors]
    sources = {str(Path(module.__file__)): base.sha(module.__file__) for module in (natural, sampling, helper, base)}
    sources[str(Path(__file__))] = base.sha(Path(__file__))
    sources[str(HERE / "05C_SAMPLING_MECHANISM_GOAL.txt")] = base.sha(HERE / "05C_SAMPLING_MECHANISM_GOAL.txt")
    for seed_index, pair_seed in enumerate(CONFIG["seeds"]):
        seed = 2300 + seed_index
        initial_hashes = []
        for group, trainer in (("weighted_expert", natural), ("bootstrap_expert", sampling)):
            check_deadline()
            if datetime.fromisoformat(CONFIG["training_cutoff"]).timestamp() - time.time() < 420:
                raise TimeoutError("Insufficient reserved time for a complete expert run")
            output = ROOT / args.split / "expert_training" / group
            output.mkdir(parents=True, exist_ok=True)
            objective_probabilities = probabilities if group == "weighted_expert" else np.full(8000, 1/8000)
            protocol = {"group": group, "mixture": mixture if group == "weighted_expert" else 0,
                        "seeds": list(range(2300,2305)), "main_seeds": CONFIG["seeds"], "epochs": 40,
                        "width": 32, "coverage": "M", "source_sha256": sources,
                        "manifest_sha256": base.sha(BASELINES / args.split / "data/manifest.json"),
                        "views": view_hashes, "probability_sha256": hashlib.sha256(objective_probabilities.tobytes()).hexdigest()}
            protocol_path = output / "protocol.json"
            if protocol_path.exists():
                assert read(protocol_path) == protocol
            else:
                write(protocol_path, protocol)
            write(output / "exposure_design.json", {"group":group, "objective_probabilities":objective_probabilities.tolist(),
                "expected_label_fraction":(objective_probabilities @ y).tolist(),
                "expected_unique_per_epoch":8000 if group=="weighted_expert" else float(np.sum(1-(1-objective_probabilities)**8000)),
                "objective_Kish_effective_sample_size":float(1/np.square(objective_probabilities).sum())})
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            model = base.build_backbone("dfms_iq", num_classes=6, width=32, dropout=.2).cuda()
            initial_hash = helper.model_hash(model)
            initial_hashes.append(initial_hash)
            options = SimpleNamespace(output=output, epochs=40,
                training_deadline=datetime.fromisoformat(CONFIG["training_cutoff"]).timestamp(),
                sample_probabilities=torch.from_numpy(objective_probabilities),
                sample_weights=torch.from_numpy((8000*probabilities).astype(np.float32)).cuda())
            receipt = trainer.train_one(options, model, x, y_tensor, early_x, early_y_tensor, seed, "M",
                initial_hash, base.sha(protocol_path), weak10, positions10, weak15, positions15)
            folder = output / f"M_{seed}"
            receipt["sampling"] = "uniform_with_replacement" if group=="bootstrap_expert" else "no_replacement_combo_weighted_loss"
            receipt["objective_mixture"] = 0 if group=="bootstrap_expert" else mixture
            write(folder / "receipt.json",receipt)
            ordinary = read(ROOT / args.split / "expert_training/ordinary_expert" / f"M_{seed}" / "receipt.json")
            assert ordinary["initial_state_sha256"] == initial_hash
            if group == "weighted_expert":
                assert ordinary["order_sha256"] == receipt["order_sha256"]
            state = torch.load(folder / "best_BCE.pth", map_location="cuda", weights_only=False)
            model.load_state_dict(state["model_state_dict"])
            model.eval()
            logits = base.predict(model, cal_x, "cuda").astype(float)
            thresholds = helper.fit_thresholds(logits, cal_y)
            destination = ROOT / args.split / "readouts" / f"{group}_{pair_seed}"
            destination.mkdir(exist_ok=True)
            bundle = ReadoutBundle().cuda().eval().requires_grad_(False)
            bundle.backbone.load_state_dict(model.state_dict())
            info = {"group": group, "seed": seed, "pair_seed": pair_seed, "receipt": receipt,
                    "protocol_sha256": base.sha(protocol_path), "time": stamp()}
            bundle.save(destination / "model.pth", thresholds, info)
            write(destination / "calibration.json", {"thresholds": thresholds, "original_thresholds": thresholds})
            write(destination / "receipt.json", {**info, "checkpoint_sha256": base.sha(destination / "model.pth")})
            np.savez(destination / "calibration_logits.npz", logits=logits, labels=cal_y)
            print("CONTROL_COMPLETE", args.split, group, seed, flush=True)
            del model, bundle
        assert initial_hashes[0] == initial_hashes[1]
    write(ROOT / args.split / "controls_status.json", {"status": "completed", "time": stamp(), "runs": 10})


if __name__ == "__main__":
    main()
