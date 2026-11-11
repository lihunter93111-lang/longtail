"""Matched frozen-backbone readouts with guarded epoch selection and replay."""
import argparse
import copy
import random
import traceback
from common import *


def train_stage(model, records, thresholds, targets, seed, directory):
    fit, check = records["fit"], records["check"]
    fit_reference = cached_logits(model, fit)
    check_reference = cached_logits(model, check)
    reference = evaluate(check, check_reference, thresholds)
    transform = FeatureTransform.fit(fit[model.variant], CONFIG["pca_components"], seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model.add_stage(targets, fit[model.variant].shape[1], transform)
    model.cuda().eval().requires_grad_(False)
    head = model.heads[-1].requires_grad_(True)
    transform = model.transforms[-1]
    with torch.no_grad():
        x = transform(torch.from_numpy(fit[model.variant]).cuda())
        check_x = transform(torch.from_numpy(check[model.variant]).cuda())
    truth = torch.from_numpy(fit["target"][:, targets]).cuda()
    frozen = torch.from_numpy(fit_reference[:, targets].astype(np.float32)).cuda()
    optimizer = torch.optim.AdamW(head.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"])
    generator = torch.Generator().manual_seed(seed + 10000)
    best = reference["exact"]
    best_epoch = 0
    best_state = copy.deepcopy(head.state_dict())
    initial_hash = helper.model_hash(head)
    history = []
    orders = []
    for epoch in range(1, CONFIG["epochs"] + 1):
        check_deadline()
        order = torch.randperm(len(x), generator=generator)
        assert len(torch.unique(order)) == 8000
        orders.append(hashlib.sha256(order.numpy().tobytes()).hexdigest())
        head.train()
        losses = []
        for index in order.cuda().split(CONFIG["batch"]):
            check_deadline()
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(frozen[index] + head(x[index]), truth[index])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), CONFIG["clip"], error_if_nonfinite=True)
            optimizer.step()
            losses.append(loss.detach())
        scheduler.step()
        head.eval()
        with torch.inference_mode():
            logits = check_reference.copy()
            logits[:, targets] += head(check_x).cpu().numpy()
        candidate = evaluate(check, logits, thresholds)
        guards = preserves_reference(candidate, reference, [LABELS[column] for column in targets])
        if guards and candidate["exact"] > best + 1e-12:
            best = candidate["exact"]
            best_epoch = epoch
            best_state = copy.deepcopy(head.state_dict())
        history.append({"epoch": epoch, "fit_loss": float(torch.stack(losses).mean()),
                        "check_exact": candidate["exact"], "check_BCE": candidate["raw_bce"],
                        "guards_passed": bool(guards), "order_sha256": orders[-1]})
        helper.atomic_torch_save({"head_state": head.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "order_rng": generator.get_state(),
            "best_state": best_state, "best_epoch": best_epoch, "history": history,
            "source_sha256": base.sha(Path(__file__))}, directory / f"stage_{len(model.stages)}_recovery.pth")
    head.load_state_dict(best_state)
    model.eval().requires_grad_(False)
    info = {"targets": targets, "best_epoch": best_epoch, "initial_state_sha256": initial_hash,
            "orders": orders, "history": history, "updates": CONFIG["epochs"] * 250,
            "fit_visits": CONFIG["epochs"] * 8000, "transform_sha256": helper.model_hash(transform)}
    write(directory / f"stage_{len(model.stages)}.json", info)
    print("READOUT_STAGE", directory.name, targets, "best_epoch", best_epoch, "check_exact", best, flush=True)
    return info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=CONFIG["splits"], required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=CONFIG["seeds"])
    parser.add_argument("--variants", nargs="+", default=CONFIG["variants"])
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    setup()
    paths = role_paths(args.split)
    models_root = ROOT / args.split / "readouts"
    models_root.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        check_deadline()
        original, calibration, baseline_receipt = load_baseline(args.split, seed)
        thresholds = calibration["thresholds"]["raw"]
        records = {role: cache_features(args.split, seed, role, original, path, indices)
                   for role, (path, indices) in paths.items()}
        saved_calibration = np.load(BASELINES / args.split / "w32" / f"M_{seed}" / "calibration_logits.npz")
        np.testing.assert_allclose(records["calibration"]["logits"], saved_calibration["logits"], atol=2e-5, rtol=1e-6)
        np.testing.assert_array_equal(expit(records["calibration"]["logits"].astype(float)) >= thresholds,
                                      expit(saved_calibration["logits"].astype(float)) >= thresholds)
        baseline_folder = models_root / f"baseline_{seed}"
        baseline_folder.mkdir(exist_ok=True)
        if not (baseline_folder / "model.pth").exists():
            original.save(baseline_folder / "model.pth", thresholds, {"baseline_receipt": baseline_receipt})
            write(baseline_folder / "calibration.json", {"thresholds": thresholds, "original_thresholds": thresholds})
        if args.smoke_only:
            inputs, _ = base.load(paths["calibration"][0], paths["calibration"][1][:63])
            with torch.inference_mode():
                np.testing.assert_allclose(original(inputs.cuda()).cpu().numpy(), records["calibration"]["logits"][:63], atol=2e-5, rtol=1e-6)
            write(ROOT / "smoke.json", {"passed": True, "seed": seed, "split": args.split,
                "feature_shapes": {v: list(records["fit"][v].shape) for v in CONFIG["variants"]},
                "parameters": sum(p.numel() for p in original.parameters()), "time": stamp()})
            print("SMOKE_PASSED", flush=True)
            return
        initial = copy.deepcopy(original.backbone.state_dict())
        stage_records = {}
        for variant in args.variants:
            check_deadline()
            folder = models_root / f"{variant}_{seed}"
            folder.mkdir(exist_ok=True)
            if (folder / "receipt.json").exists():
                receipt = read(folder / "receipt.json")
                assert receipt["source_sha256"] == base.sha(Path(__file__))
                assert receipt["checkpoint_sha256"] == base.sha(folder / "model.pth")
                stage_records[variant] = receipt["stages"]
                continue
            model = ReadoutBundle(variant).cuda().eval().requires_grad_(False)
            model.backbone.load_state_dict(initial)
            backbone_hash = helper.model_hash(model.backbone)
            started = time.time()
            stages = [train_stage(model, records, thresholds, targets, seed, folder) for targets in CONFIG["stages"]]
            assert helper.model_hash(model.backbone) == backbone_hash
            cal_z = cached_logits(model, records["calibration"])
            new_thresholds = helper.fit_thresholds(cal_z, records["calibration"]["target"])
            unchanged = [0, 1, 3]
            np.testing.assert_array_equal(cal_z[:, unchanged], records["calibration"]["logits"][:, unchanged])
            info = {"seed": seed, "split": args.split, "variant": variant, "stages": stages,
                    "source_sha256": base.sha(Path(__file__)), "protocol_sha256": base.sha(HERE / "protocol.json"),
                    "backbone_sha256": backbone_hash, "seconds": time.time() - started,
                    "completed_utc": stamp(), "role": "development_candidate"}
            model.save(folder / "model.pth", new_thresholds, info)
            loaded, _ = load_bundle(folder / "model.pth")
            inputs, _ = base.load(paths["calibration"][0], paths["calibration"][1])
            live = base.predict(loaded, inputs, "cuda").astype(float)
            np.testing.assert_allclose(live, cal_z, rtol=1e-5, atol=3e-5)
            mismatch = int(np.sum((expit(live) >= new_thresholds) != (expit(cal_z) >= new_thresholds)))
            assert mismatch == 0, (variant, seed, mismatch)
            write(folder / "calibration.json", {"thresholds": new_thresholds, "original_thresholds": thresholds,
                  "normal_forward_max_logit_error": float(np.max(abs(live-cal_z))), "binary_mismatches": mismatch})
            np.savez(folder / "calibration_logits.npz", logits=live, labels=records["calibration"]["target"])
            write(folder / "receipt.json", {**info, "checkpoint_sha256": base.sha(folder / "model.pth")})
            stage_records[variant] = stages
            print("READOUT_COMPLETE", args.split, seed, variant, flush=True)
            del model, loaded
        for stage_index in range(len(CONFIG["stages"])):
            values = [stage_records[v][stage_index] for v in args.variants]
            assert len({v["initial_state_sha256"] for v in values}) == 1
            assert all(v["orders"] == values[0]["orders"] for v in values)
        del records, original, initial
        torch.cuda.empty_cache()
    write(ROOT / args.split / "readout_status.json", {"status": "completed", "seeds": args.seeds,
          "variants": args.variants, "finished_utc": stamp()})


if __name__ == "__main__":
    try:
        main()
    except Exception:
        write(ROOT / f"failure_readouts_{time.time_ns()}.json", {"error": traceback.format_exc(), "time": stamp()})
        raise
