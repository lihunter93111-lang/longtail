"""Preserve the backbone M-view schedule while adapting frozen readouts."""
import copy
from common import *


def weak_features(split, seed, gap, model, fit):
    source = BASELINES / split / "data" / f"fit_gap{gap}.mat"
    expected = read(BASELINES / split / "w32/protocol.json")["views"][source.name]
    assert base.sha(source) == expected
    output = ROOT / split / "cache" / f"M_{seed}" / f"weak{gap}.npz"
    if output.exists():
        with np.load(output) as data:
            return dict(data)
    inputs, positions, targets = helper.load_views(source)
    records = collect(model, inputs)
    records.update(positions=positions, targets=targets, target=fit["target"][positions])
    assert (records["target"][np.arange(len(targets)), targets] == 1).all()
    with output.with_suffix(".tmp").open("wb") as stream:
        np.savez(stream, **records)
    output.with_suffix(".tmp").replace(output)
    write(output.with_suffix(".json"), {"source_sha256": expected, "sha256": base.sha(output), "time": stamp()})
    return records


def train_stage(model, fit, check, weak10, weak15, thresholds, targets, seed, folder):
    references = {"fit": cached_logits(model, fit), "weak10": cached_logits(model, weak10),
                  "weak15": cached_logits(model, weak15), "check": cached_logits(model, check)}
    reference_metrics = evaluate(check, references["check"], thresholds)
    transform = FeatureTransform.fit(fit[model.variant], CONFIG["pca_components"], seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model.add_stage(targets, fit[model.variant].shape[1], transform)
    model.cuda().eval().requires_grad_(False)
    transform = model.transforms[-1]
    head = model.heads[-1].requires_grad_(True)
    initial_hash = helper.model_hash(head)
    with torch.no_grad():
        x, x10, x15, check_x = [transform(torch.from_numpy(record[model.variant]).cuda())
                                for record in (fit, weak10, weak15, check)]
    y = torch.from_numpy(fit["target"][:, targets]).cuda()
    frozen, frozen10, frozen15 = [torch.from_numpy(references[role][:, targets].astype(np.float32)).cuda()
                                 for role in ("fit", "weak10", "weak15")]
    p10, p15 = [torch.from_numpy(record["positions"]).cuda() for record in (weak10, weak15)]
    optimizer = torch.optim.AdamW(head.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"])
    generator = torch.Generator().manual_seed(seed+10000)
    best, best_epoch = reference_metrics["exact"], 0
    best_state = copy.deepcopy(head.state_dict())
    history = []
    for epoch in range(1, CONFIG["epochs"]+1):
        check_deadline()
        selected10 = (p10 + epoch) % 2 == 0
        selected15 = ((p15 + epoch) % 2 == 0) & (((epoch + p15 % 2) // 2) % 2 == 0)
        epoch_x, epoch_reference = x.clone(), frozen.clone()
        epoch_x[p10[selected10]], epoch_reference[p10[selected10]] = x10[selected10], frozen10[selected10]
        epoch_x[p15[selected15]], epoch_reference[p15[selected15]] = x15[selected15], frozen15[selected15]
        order = torch.randperm(len(x), generator=generator)
        losses = []
        head.train()
        for indices in order.cuda().split(CONFIG["batch"]):
            check_deadline()
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(epoch_reference[indices] + head(epoch_x[indices]), y[indices])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), CONFIG["clip"], error_if_nonfinite=True)
            optimizer.step()
            losses.append(loss.detach())
        scheduler.step()
        head.eval()
        with torch.inference_mode():
            logits = references["check"].copy()
            logits[:, targets] += head(check_x).cpu().numpy()
        metrics = evaluate(check, logits, thresholds)
        guard = preserves_reference(metrics, reference_metrics, [LABELS[t] for t in targets])
        if guard and metrics["exact"] > best + 1e-12:
            best, best_epoch, best_state = metrics["exact"], epoch, copy.deepcopy(head.state_dict())
        history.append({"epoch": epoch, "fit_loss": float(torch.stack(losses).mean()), "check_exact": metrics["exact"],
                        "check_BCE": metrics["raw_bce"], "guard": bool(guard), "weak_visits": int(selected10.sum()),
                        "gap15_visits": int(selected15.sum()), "order_sha256": hashlib.sha256(order.numpy().tobytes()).hexdigest()})
        helper.atomic_torch_save({"head_state": head.state_dict(), "best_state": best_state, "best_epoch": best_epoch,
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "order_rng": generator.get_state(),
            "history": history}, folder / f"stage_{len(model.stages)}_recovery.pth")
    assert sum(row["weak_visits"] for row in history) == 12*len(p10)
    assert sum(row["gap15_visits"] for row in history) == 6*len(p15)
    head.load_state_dict(best_state)
    model.eval().requires_grad_(False)
    info = {"targets": targets, "best_epoch": best_epoch, "history": history, "initial_state_sha256": initial_hash,
            "updates": 6000, "fit_visits": 192000}
    write(folder / f"stage_{len(model.stages)}.json", info)
    print("COVERAGE_HEAD_STAGE", folder.name, targets, best_epoch, flush=True)
    return info


def main():
    setup()
    for split in CONFIG["splits"]:
        for seed in CONFIG["seeds"]:
            original, calibration, receipt = load_baseline(split, seed)
            thresholds = calibration["thresholds"]["raw"]
            records = {}
            for role in ("fit", "check", "calibration"):
                with np.load(ROOT / split / "cache" / f"M_{seed}" / f"{role}.npz") as data:
                    records[role] = dict(data)
            weak10 = weak_features(split, seed, 10, original, records["fit"])
            weak15 = weak_features(split, seed, 15, original, records["fit"])
            for variant in ("global_stats", "segment"):
                output = ROOT / split / "readouts" / f"{variant}_coverage_{seed}"
                output.mkdir(exist_ok=True)
                if (output / "receipt.json").exists():
                    assert read(output / "receipt.json")["source_sha256"] == base.sha(Path(__file__))
                    continue
                model = ReadoutBundle(variant).cuda().eval().requires_grad_(False)
                model.backbone.load_state_dict(original.backbone.state_dict())
                backbone_hash = helper.model_hash(model.backbone)
                stages = [train_stage(model, records["fit"], records["check"], weak10, weak15, thresholds,
                                      targets, seed, output) for targets in CONFIG["stages"]]
                assert helper.model_hash(model.backbone) == backbone_hash
                logits = cached_logits(model, records["calibration"])
                selected = helper.fit_thresholds(logits, records["calibration"]["target"])
                info = {"coverage": "M", "stages": stages, "source_sha256": base.sha(Path(__file__)),
                        "prompt_sha256": base.sha(HERE / "03B_COVERAGE_READOUT_GOAL.txt"), "time": stamp()}
                model.save(output / "model.pth", selected, info)
                write(output / "calibration.json", {"thresholds": selected, "original_thresholds": thresholds})
                write(output / "receipt.json", {**info, "checkpoint_sha256": base.sha(output / "model.pth")})
                print("COVERAGE_READOUT_COMPLETE", split, seed, variant, flush=True)
                del model
            del original, records, weak10, weak15
            torch.cuda.empty_cache()
    write(ROOT / "coverage_readout_status.json", {"status": "completed", "time": stamp()})


if __name__ == "__main__":
    main()
