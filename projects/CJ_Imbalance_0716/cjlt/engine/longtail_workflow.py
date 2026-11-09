"""Configuration-based baseline/readout jobs; no historical experiment orchestration."""
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import argparse
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from threadpoolctl import threadpool_limits
from cjlt.data.ama_hdf5 import load_roles, WeakViewDataset
from cjlt.data.label_sampling import draw_indices
from cjlt.models.classwise_fusion import ClasswiseReadoutModel, AMA_CLASSES
from cjlt.engine.longtail_evaluation import fit_thresholds
from cjlt.engine.model_bundle import load_bundle, save_bundle, file_sha256
from cjlt.engine.readout_trainer import collect_records, metrics, train_readout
from cjlt.engine.selective_fusion import train_fusion


def read_config(path):
    path = Path(path).resolve()
    if path.suffix == ".json":
        config = json.loads(path.read_text(encoding="utf-8"))
    else:
        import yaml
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Expected a configuration mapping")
    for key in ("train", "development", "split", "pairs", "weak_views"):
        value = config["data"].get(key)
        if value:
            value = Path(os.path.expandvars(value)).expanduser()
            config["data"][key] = str((value if value.is_absolute() else path.parent / value).resolve())
    for key in ("reference", "specialist", "initialization", "fmb_expert"):
        if config.get(key):
            value = Path(os.path.expandvars(config[key])).expanduser()
            config[key] = str((value if value.is_absolute() else path.parent / value).resolve())
    return config


def reference_model(config, seed):
    path = config.get("reference")
    if not path:
        raise ValueError("Set reference to a native model or an accepted legacy selection")
    path = path.format(seed=seed)
    if config.get("reference_format") == "backbone":
        if tuple(config["data"]["classes"]) != AMA_CLASSES:
            raise ValueError("Raw backbone import requires explicit AMA labels")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        state = payload.get("model_state_dict", payload.get("model", payload.get("state_dict", payload)))
        model = ClasswiseReadoutModel(backbone=config.get("backbone"))
        model.backbone.load_state_dict(state, strict=True)
        thresholds = config.get("reference_thresholds", [0.5] * 6)
    else:
        model, thresholds, _ = load_bundle(path, seed)
    for name in config.get("remove_readouts", []):
        if name not in model.readouts:
            raise ValueError("Cannot remove missing readout: {}".format(name))
        del model.readouts[name]
        del model.transforms[name]
        model.readout_options = [item for item in model.readout_options if item["name"] != name]
    return model.eval(), thresholds


def protocol_fingerprint(config, roles):
    """Bind role ordering and actual signal bytes, not just identity metadata."""
    paths = {dataset.path for dataset in roles.values()}
    for key in ("pairs", "weak_views"):
        if config["data"].get(key):
            paths.add(str(Path(config["data"][key]).resolve()))
    return dict(
        role_identity_sha256={name: hashlib.sha256(
            dataset.identities[dataset.indices].astype("<i8").tobytes()).hexdigest()
            for name, dataset in roles.items()},
        data_sha256=sorted(file_sha256(path) for path in paths),
    )


def verify_frozen(path, model_path, config, roles):
    frozen = json.loads(Path(path).read_text(encoding="utf-8"))
    if frozen.get("selected_without_report") is not True or tuple(frozen.get("classes", ())) != AMA_CLASSES:
        raise ValueError("Expected a frozen AMA selection made without report access")
    if frozen.get("model_sha256") != file_sha256(model_path):
        raise ValueError("Model differs from the frozen selection")
    for key, value in protocol_fingerprint(config, roles).items():
        if frozen.get(key) != value:
            raise ValueError("Evaluation data or identities differ from the frozen protocol")


def train_baseline(model, roles, options, seed, output, device, expert_path=None, weak_path=None, cache=False):
    model.to(device)
    epochs = int(options.get("epochs", 24))
    steps = int(options.get("steps_per_epoch", 250))
    batch_size = int(options.get("batch_size", 32))
    labels = roles["fit"].labels[roles["fit"].indices]
    prior = torch.from_numpy(np.clip(labels.mean(0), 1e-6, 1 - 1e-6)).to(device)
    mode = options.get("loss", "bce")
    if mode not in ("bce", "weighted_bce", "logit_adjustment"):
        raise ValueError("Unknown BCE variant")
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(options.get("lr", 1e-4)),
                                 weight_decay=float(options.get("weight_decay", 1e-4)))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    schedule = options.get("schedule", "cosine")
    if schedule not in ("cosine", "endpoint_cosine", "constant"):
        raise ValueError("Unknown learning-rate schedule")
    fit = WeakViewDataset(roles["fit"], weak_path, cache) if weak_path else roles["fit"]
    if weak_path and fit.parent_sha256 != file_sha256(roles["fit"].path):
        raise ValueError("Weak views belong to a different natural fit file")
    order_generator = torch.Generator().manual_seed(seed)
    best_score = -float("inf")
    best_state = None
    history = []
    recovery = output / "recovery.pt"
    if recovery.exists():
        saved = torch.load(recovery, map_location=device, weights_only=True)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        history, best_state, best_score = saved["history"], saved["best_state"], saved["best_score"]
        torch.set_rng_state(saved["torch_rng"].cpu())
        if str(device).startswith("cuda"):
            torch.cuda.set_rng_state(saved["cuda_rng"].cpu(), device)
        order_generator.set_state(saved["order_rng"].cpu())
    initial_state_sha = hashlib.sha256(b"".join(value.detach().cpu().numpy().tobytes()
                                      for value in model.state_dict().values())).hexdigest() if not history else saved["initial_state_sha256"]
    for epoch in range(len(history) + 1, epochs + 1):
        started = time.monotonic()
        if schedule != "cosine":
            high, low = float(options.get("lr", 1e-4)), float(options.get("lr_min", 1e-6))
            rate = high if schedule == "constant" else low + (high - low) * (1 + math.cos(math.pi * (epoch - 1) / max(epochs - 1, 1))) / 2
            for group in optimizer.param_groups:
                group["lr"] = rate
        if options.get("identity_order") == "fixed_generator":
            if options.get("sampler", "uniform") != "uniform" or steps * batch_size != len(labels):
                raise ValueError("Fixed identity order requires one natural fit presentation per epoch")
            order = torch.randperm(len(labels), generator=order_generator).numpy()
        else:
            order = draw_indices(labels, steps * batch_size, seed, epoch, options.get("sampler", "uniform"))
        if weak_path:
            fit.epoch = epoch
        model.train()
        losses = []
        for batch in DataLoader(Subset(fit, order.tolist()), batch_size=batch_size, shuffle=False):
            optimizer.zero_grad()
            logits = model(batch["signal"].to(device))
            if mode == "logit_adjustment":
                logits = logits + torch.logit(prior)
            pos_weight = (1 - prior) / prior if mode == "weighted_bce" else None
            loss = F.binary_cross_entropy_with_logits(logits, batch["target"].to(device), pos_weight=pos_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        if schedule == "cosine":
            scheduler.step()
        records = collect_records(model, roles["check"], device=device)
        measured = metrics(records, records["logits"], [0.5] * 6)
        key = options.get("selection_metric", "macro_aupr")
        if key not in ("macro_aupr", "exact", "raw_bce"):
            raise ValueError("Unknown check selection metric")
        score = measured[key] * (-1 if key == "raw_bce" else 1)
        if score > best_score:
            best_score, best_state = score, deepcopy(model.state_dict())
        history.append(dict(epoch=epoch, loss=float(np.mean(losses)), check_score=score,
                            check_exact=measured["exact"], check_raw_bce=measured["raw_bce"],
                            independent_fit_identities=int(len(np.unique(order))),
                            augmented_presentations=fit.augmented_count(order) if weak_path else 0,
                            order_sha256=hashlib.sha256(order.astype("<i8").tobytes()).hexdigest(),
                            lr=optimizer.param_groups[0]["lr"], seconds=time.monotonic() - started))
        (output / "history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
        torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                        best_state=best_state, best_score=best_score, history=history, torch_rng=torch.get_rng_state(),
                        cuda_rng=torch.cuda.get_rng_state(device) if str(device).startswith("cuda") else None,
                        order_rng=order_generator.get_state(), initial_state_sha256=initial_state_sha), recovery.with_suffix(".tmp"))
        recovery.with_suffix(".tmp").replace(recovery)
        print(json.dumps(dict(seed=seed, **history[-1])), flush=True)
    model.load_state_dict(best_state)
    if weak_path:
        fit.weak.close()
    if expert_path:
        expert, _, _ = load_bundle(expert_path.format(seed=seed), seed)
        if expert.backbone_options != model.backbone_options:
            raise ValueError("FMB expert must use the same backbone dimensions")
        model.expert = expert.backbone.to(device).eval().requires_grad_(False)
    calibration = collect_records(model, roles["calibration"], device=device)
    calibration_method = options.get("calibration", "exact_f1")
    if calibration_method not in ("exact_f1", "grid_f1"):
        raise ValueError("Unknown F1 calibration policy")
    thresholds = fit_thresholds(calibration["target"], calibration["logits"],
                                grid=np.linspace(0.05, 0.95, 181) if calibration_method == "grid_f1" else None)
    if expert_path:
        thresholds[0] = 0.5
    info = dict(seed=seed, kind="baseline", options=options, optimizer_steps=epochs * steps,
                history=history, initial_state_sha256=initial_state_sha, selection_role="check", calibration_role="calibration", report_used=False)
    save_bundle(output / "model.pt", model, thresholds, info)
    (output / "history.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    return info


def train_job(job):
    # PCA uses NumPy BLAS independently of Torch's intra-op thread setting.
    with threadpool_limits(limits=job[4], user_api="blas"):
        return _train_job(job)


def _train_job(job):
    kind, config, seed, output, threads, device = job
    if kind not in ("baseline", "readout", "fusion"):
        raise ValueError("Unknown training kind")
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    output = Path(output) / ("seed{}".format(seed))
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("Configuration changed; use a new output directory")
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    roles = load_roles(config["data"])
    try:
        options = config.get("training", {})
        if (output / "frozen.json").exists():
            verify_frozen(output / "frozen.json", output / "model.pt", config, roles)
            return dict(seed=seed, model=str(output / "model.pt"), already_complete=True)
        inputs = protocol_fingerprint(config, roles)
        input_path = output / "input_protocol.json"
        if input_path.exists() and json.loads(input_path.read_text()) != inputs:
            raise ValueError("Input data changed; refusing to resume an existing training run")
        input_path.write_text(json.dumps(inputs, indent=2) + "\n", encoding="utf-8")
        if min(int(options.get(key, default)) for key, default in
               (("epochs", 24), ("steps_per_epoch", 250), ("batch_size", 32))) < 1:
            raise ValueError("Positive training budgets are required")
        if kind == "fusion":
            info = train_fusion(config, roles, seed, output, device)
        elif kind == "readout":
            model, thresholds = reference_model(config, seed)
            combined = {**options, **config.get("readout", {})}
            info = train_readout(model, thresholds, roles, combined, seed, output, device, config["data"].get("pairs"))
        else:
            if config["data"].get("cache") == "memory":
                for role in ("fit", "check", "calibration"):
                    roles[role].cache_signals()
            model = ClasswiseReadoutModel(backbone=config.get("backbone"))
            if config.get("initialization"):
                payload = torch.load(config["initialization"].format(seed=seed), map_location="cpu", weights_only=True)
                state = payload.get("model_state_dict", payload.get("model", payload.get("state_dict", payload)))
                missing, extra = model.backbone.load_state_dict({k: v for k, v in state.items() if not k.startswith("classifier.")}, strict=False)
                if extra or any(not key.startswith("classifier.") for key in missing):
                    raise ValueError("Incompatible historical backbone initialization")
            info = train_baseline(model, roles, options, seed, output, device, config.get("fmb_expert"),
                                  config["data"].get("weak_views"), config["data"].get("cache") == "memory")
        # Freeze the portable model before an explicit report evaluation is allowed.
        if protocol_fingerprint(config, roles) != inputs:
            raise ValueError("Input data changed while training")
        frozen = dict(model_sha256=file_sha256(output / "model.pt"), classes=list(AMA_CLASSES),
                      seed=seed, selected_without_report=True,
                      **inputs)
        (output / "frozen.json").write_text(json.dumps(frozen, indent=2) + "\n", encoding="utf-8")
        return dict(seed=seed, model=str(output / "model.pt"), optimizer_steps=info["optimizer_steps"])
    finally:
        for dataset in roles.values():
            dataset.close()


def training_main(kind):
    parser = argparse.ArgumentParser(description="Train {} with an explicit shared data protocol".format(kind))
    if kind == "readout":
        parser.add_argument("--kind", choices=("readout", "fusion"), default=kind)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(2050, 2055)))
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()
    if args.jobs < 1 or args.threads < 1 or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Use positive concurrency and distinct model seeds")
    config = read_config(args.config)
    if args.reference:
        config["reference"] = str(args.reference.resolve())
    jobs = [(getattr(args, "kind", kind), config, seed, str(args.output.resolve()), args.threads, args.device) for seed in args.seeds]
    if args.jobs == 1:
        results = [train_job(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=args.jobs, mp_context=multiprocessing.get_context("spawn")) as pool:
            results = list(pool.map(train_job, jobs))
    print(json.dumps(results, ensure_ascii=False))


def evaluation_main():
    parser = argparse.ArgumentParser(description="Evaluate a frozen model or export historical components")
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=2050)
    parser.add_argument("--export-model", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--role", choices=("calibration", "check", "report"), default="check")
    parser.add_argument("--frozen", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    model, thresholds, metadata = load_bundle(args.model, args.seed)
    if args.export_model:
        save_bundle(args.export_model, model, thresholds, metadata)
    if args.config is None:
        if not args.export_model:
            parser.error("Supply --config for evaluation or --export-model for conversion")
        return
    if args.output is None:
        parser.error("Evaluation requires --output")
    config = read_config(args.config)
    roles = load_roles(config["data"])
    try:
        if args.role == "report":
            if args.frozen is None:
                raise ValueError("Report access requires a frozen model record")
            verify_frozen(args.frozen, args.model, config, roles)
        records = collect_records(model, roles[args.role], device=args.device)
        args.output.mkdir(parents=True, exist_ok=False)
        measured = metrics(records, records["logits"], thresholds)
        measured["at_0_5"] = metrics(records, records["logits"], [0.5] * 6)
        (args.output / "metrics.json").write_text(json.dumps(measured,
                                                            indent=2, allow_nan=False) + "\n", encoding="utf-8")
        np.savez_compressed(args.output / "predictions.npz", **records, thresholds=np.asarray(thresholds))
    finally:
        for dataset in roles.values():
            dataset.close()
