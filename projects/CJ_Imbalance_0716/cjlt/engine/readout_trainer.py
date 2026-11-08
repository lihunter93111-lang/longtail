"""Frozen-reference training shared by global and segmented residual heads."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import h5py
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from cjlt.data.ama_hdf5 import AMAHDF5Dataset
from cjlt.data.label_sampling import draw_indices, epoch_seed
from cjlt.models.classwise_fusion import AMA_CLASSES
from cjlt.models.segment_readout import segment_statistics
from cjlt.engine.feature_transform import FeatureTransform
from cjlt.engine.longtail_evaluation import evaluate_predictions, preserves_reference
from cjlt.engine.model_bundle import save_bundle


def collect_records(model, dataset, batch_size=64, device="cpu", features=False):
    model.eval().to(device)
    result = {}
    try:
        with torch.inference_mode():
            for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False):
                values = batch["signal"].to(device)
                if features:
                    logits, global_features, temporal = model(values, return_features=True)
                    items = dict(logits=logits, global_features=global_features,
                                 segment_features=segment_statistics(temporal))
                else:
                    items = dict(logits=model(values))
                items.update({key: value for key, value in batch.items() if key != "signal"})
                for key, value in items.items():
                    result.setdefault(key, []).append(value.detach().cpu().numpy())
    finally:
        dataset.close()
    return {key: np.concatenate(value) for key, value in result.items()}


def metrics(records, logits, thresholds):
    return evaluate_predictions(records["target"], logits, thresholds,
                                records.get("weak_mask"), records.get("relative_power_db"))


def train_readout(model, thresholds, roles, options, seed, output, device="cpu", pair_path=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    model.eval().requires_grad_(False)
    batch_size = int(options.get("batch_size", 32))
    epochs = int(options.get("epochs", 24))
    steps = int(options.get("steps_per_epoch", 250))
    fit = collect_records(model, roles["fit"], device=device, features=True)
    check = collect_records(model, roles["check"], device=device, features=True)
    targets = options.get("targets", ["SawFM", "MFT"])
    columns = [AMA_CLASSES.index(name) for name in targets]
    source = options.get("source", "segment")
    feature_key = "segment_features" if source == "segment" else "global_features"
    transform = FeatureTransform.fit(fit[feature_key], int(options.get("dimensions", 128)), seed)
    name = options.get("name", "readout")
    torch.manual_seed(seed)
    model.add_readout(name, targets, source, int(options.get("dimensions", 128)),
                      int(options.get("hidden", 32)), transform)
    head = model.readouts[name].to(device)
    head.requires_grad_(True)
    x = torch.from_numpy(transform.project(fit[feature_key])).to(device)
    cv = torch.from_numpy(transform.project(check[feature_key])).to(device)
    y = torch.from_numpy(fit["target"][:, columns]).to(device)
    base = torch.from_numpy(fit["logits"][:, columns]).to(device)
    pairs = None
    pair_views = int(options.get("pair_views", 0))
    if pair_views:
        if pair_views % 2 or pair_views >= batch_size or not pair_path:
            raise ValueError("Provide pair data and an even pair_views count below batch_size")
        paired_data = AMAHDF5Dataset(pair_path)
        for role in ("calibration", "check", "report"):
            dataset = roles[role]
            if np.intersect1d(paired_data.noise_ids, dataset.noise_ids[dataset.indices]).size:
                raise ValueError("Paired fit noise leaks into an evaluation role")
        pairs = collect_records(model, paired_data, device=device, features=True)
        # The newly attached branch is zero, so these are still frozen-reference logits.
        with h5py.File(pair_path, "r") as stream:
            target = np.asarray(stream["pair_target"]).reshape(-1)
            gap = np.asarray(stream["target_gap_bin"]).reshape(-1)
            side = np.asarray(stream["pair_side"]).reshape(-1)
        if any(len(field) != len(paired_data) for field in (target, gap, side)):
            raise ValueError("Pair metadata length mismatch")
        if len(side) % 2 or not np.array_equal(side, np.tile([0, 1], len(side) // 2)):
            raise ValueError("Pair views must be adjacent negative/positive rows")
        if not np.array_equal(paired_data.noise_ids[::2], paired_data.noise_ids[1::2]):
            raise ValueError("Each pair must share receiver noise")
        if not np.array_equal(target[::2], target[1::2]) or not np.array_equal(gap[::2], gap[1::2]):
            raise ValueError("Both views must declare the same target and power stratum")
        changed = paired_data.labels[1::2] - paired_data.labels[::2]
        expected = np.zeros_like(changed)
        expected[np.arange(len(changed)), target[::2]] = 1
        if not np.array_equal(changed, expected):
            raise ValueError("A pair must add exactly its declared target")
        strata = (target[::2] == 4).astype(int) * 3 + gap[::2]
        if set(target[::2]) != {2, 4} or not np.isin(gap, [0, 1, 2]).all():
            raise ValueError("Expected SawFM/MFT pairs in three declared power-gap strata")
        groups = [np.flatnonzero(strata == k) for k in range(6)]
        if not all(len(group) for group in groups):
            raise ValueError("All target/power strata must be represented")
        px = torch.from_numpy(transform.project(pairs[feature_key])).to(device)
        py = torch.from_numpy(pairs["target"][:, columns]).to(device)
        pb = torch.from_numpy(pairs["logits"][:, columns]).to(device)
    if str(device) == "cpu":
        torch.set_num_threads(int(options.get("head_threads", 1)))
    optimizer = torch.optim.AdamW(head.parameters(), lr=float(options.get("lr", 3e-4)),
                                 weight_decay=float(options.get("weight_decay", 1e-4)))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    reference = metrics(check, check["logits"], thresholds)
    best = reference["exact"]
    best_epoch = 0
    best_state = deepcopy(head.state_dict())
    history = []
    for epoch in range(1, epochs + 1):
        natural_count = batch_size - pair_views
        order = draw_indices(fit["target"], steps * natural_count, seed, epoch, stream=10)
        if pairs is not None:
            count = steps * (pair_views // 2)
            strata_order = (np.arange(count) + (epoch - 1) * count) % 6
            rng = np.random.default_rng(epoch_seed(seed, epoch, 31))
            rng.shuffle(strata_order)
            selected = np.array([rng.choice(groups[k]) for k in strata_order])
        head.train()
        losses = []
        for step, indices in enumerate(order.reshape(steps, natural_count)):
            xx, yy, bb = x[indices], y[indices], base[indices]
            if pairs is not None:
                chosen = selected[step * (pair_views // 2):(step + 1) * (pair_views // 2)]
                rows = np.column_stack((2 * chosen, 2 * chosen + 1)).reshape(-1)
                xx, yy, bb = [torch.cat((a, b[rows])) for a, b in ((xx, px), (yy, py), (bb, pb))]
            optimizer.zero_grad()
            loss = F.binary_cross_entropy_with_logits(bb + head(xx), yy)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 5, error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()
        head.eval()
        with torch.inference_mode():
            logits = check["logits"].copy()
            logits[:, columns] += head(cv).cpu().numpy()
        candidate = metrics(check, logits, thresholds)
        guarded = preserves_reference(candidate, reference, targets)
        if guarded and candidate["exact"] > best + 1e-12:
            best, best_epoch, best_state = candidate["exact"], epoch, deepcopy(head.state_dict())
        history.append(dict(epoch=epoch, loss=float(np.mean(losses)), check_exact=candidate["exact"],
                            guard_passed=guarded, natural_views=len(order), paired_views=steps * pair_views,
                            draw_sha256=hashlib.sha256(order.astype("<i8").tobytes()).hexdigest()))
    head.load_state_dict(best_state)
    model.eval().requires_grad_(False)
    info = dict(seed=seed, kind="frozen_readout", options=options, best_epoch=best_epoch,
                optimizer_steps=epochs * steps, check_exact=best, history=history,
                selection_role="check", decision="locked reference thresholds", report_used=False)
    save_bundle(output / "model.pt", model, thresholds, info)
    (output / "history.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    return info
