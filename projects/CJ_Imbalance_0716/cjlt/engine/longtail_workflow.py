"""Configuration-based baseline/readout jobs; no historical experiment orchestration."""
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import argparse
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import random
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from threadpoolctl import threadpool_limits
from cjlt.data.ama_hdf5 import load_roles
from cjlt.data.label_sampling import draw_indices
from cjlt.models.classwise_fusion import ClasswiseReadoutModel, AMA_CLASSES
from cjlt.engine.longtail_evaluation import fit_thresholds
from cjlt.engine.model_bundle import load_bundle, save_bundle, file_sha256
from cjlt.engine.readout_trainer import collect_records, metrics, train_readout


def read_config(path):
    path = Path(path).resolve()
    if path.suffix == ".json":
        config = json.loads(path.read_text(encoding="utf-8"))
    else:
        import yaml
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Expected a configuration mapping")
    for key in ("train", "development", "split", "pairs"):
        value = config["data"].get(key)
        if value:
            value = Path(os.path.expandvars(value)).expanduser()
            config["data"][key] = str((value if value.is_absolute() else path.parent / value).resolve())
    for key in ("reference", "initialization", "fmb_expert"):
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
    if config["data"].get("pairs"):
        paths.add(str(Path(config["data"]["pairs"]).resolve()))
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


def train_baseline(model, roles, options, seed, output, device, expert_path=None):
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
    best_score = -float("inf")
    best_state = None
    history = []
    for epoch in range(1, epochs + 1):
        order = draw_indices(labels, steps * batch_size, seed, epoch, options.get("sampler", "uniform"))
        model.train()
        losses = []
        for batch in DataLoader(Subset(roles["fit"], order.tolist()), batch_size=batch_size, shuffle=False):
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
                            check_exact=measured["exact"], independent_fit_identities=int(len(np.unique(order)))))
    model.load_state_dict(best_state)
    if expert_path:
        expert, _, _ = load_bundle(expert_path.format(seed=seed), seed)
        if expert.backbone_options != model.backbone_options:
            raise ValueError("FMB expert must use the same backbone dimensions")
        model.expert = expert.backbone.to(device).eval().requires_grad_(False)
    calibration = collect_records(model, roles["calibration"], device=device)
    thresholds = fit_thresholds(calibration["target"], calibration["logits"])
    if expert_path:
        thresholds[0] = 0.5
    info = dict(seed=seed, kind="baseline", options=options, optimizer_steps=epochs * steps,
                history=history, selection_role="check", calibration_role="calibration", report_used=False)
    save_bundle(output / "model.pt", model, thresholds, info)
    (output / "history.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    return info


def train_job(job):
    # PCA uses NumPy BLAS independently of Torch's intra-op thread setting.
    with threadpool_limits(limits=job[4], user_api="blas"):
        return _train_job(job)


def _train_job(job):
    kind, config, seed, output, threads, device = job
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    output = Path(output) / ("seed{}".format(seed))
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    roles = load_roles(config["data"])
    try:
        options = config.get("training", {})
        if min(int(options.get(key, default)) for key, default in
               (("epochs", 24), ("steps_per_epoch", 250), ("batch_size", 32))) < 1:
            raise ValueError("Positive training budgets are required")
        if kind == "readout":
            model, thresholds = reference_model(config, seed)
            combined = {**options, **config.get("readout", {})}
            info = train_readout(model, thresholds, roles, combined, seed, output, device, config["data"].get("pairs"))
        else:
            model = ClasswiseReadoutModel(backbone=config.get("backbone"))
            if config.get("initialization"):
                payload = torch.load(config["initialization"].format(seed=seed), map_location="cpu", weights_only=True)
                state = payload.get("model_state_dict", payload.get("model", payload.get("state_dict", payload)))
                missing, extra = model.backbone.load_state_dict({k: v for k, v in state.items() if not k.startswith("classifier.")}, strict=False)
                if extra or any(not key.startswith("classifier.") for key in missing):
                    raise ValueError("Incompatible historical backbone initialization")
            info = train_baseline(model, roles, options, seed, output, device, config.get("fmb_expert"))
        # Freeze the portable model before an explicit report evaluation is allowed.
        frozen = dict(model_sha256=file_sha256(output / "model.pt"), classes=list(AMA_CLASSES),
                      seed=seed, selected_without_report=True,
                      **protocol_fingerprint(config, roles))
        (output / "frozen.json").write_text(json.dumps(frozen, indent=2) + "\n", encoding="utf-8")
        return dict(seed=seed, model=str(output / "model.pt"), optimizer_steps=info["optimizer_steps"])
    finally:
        for dataset in roles.values():
            dataset.close()


def training_main(kind):
    parser = argparse.ArgumentParser(description="Train {} with an explicit shared data protocol".format(kind))
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
    jobs = [(kind, config, seed, str(args.output.resolve()), args.threads, args.device) for seed in args.seeds]
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
        np.savez_compressed(args.output / "predictions.npz", **records)
    finally:
        for dataset in roles.values():
            dataset.close()
