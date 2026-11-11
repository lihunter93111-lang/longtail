"""Shared numerical helpers and portable readout inference for this campaign."""
import hashlib
import importlib.util
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import numpy as np
import h5py
from scipy.special import expit

HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "protocol.json").read_text(encoding="utf-8"))
SOURCE = Path(CONFIG["remote_source"])
ROOT = Path(CONFIG["storage"])
BASELINES = Path(CONFIG["baseline_root"])
sys.path.insert(0, str(SOURCE / "projects/CJ_Imbalance_0716"))
spec = importlib.util.spec_from_file_location(
    "campaign_weak_helpers", SOURCE / "experiment-log/2026-11-06-05-weak-component-goal/run_weak_goal.py"
)
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
base = helper.base
from cjlt.models.segment_readout import ResidualReadout, segment_statistics
from cjlt.engine.feature_transform import FeatureTransform
from cjlt.engine.longtail_evaluation import evaluate_predictions, preserves_reference

LABELS = helper.LABELS
QUOTAS = json.loads((SOURCE / "experiment-log/2026-11-06-01-lt-abc/2026-11-06-lt-abc-quotas.json").read_text(encoding="utf-8"))


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for attempt in range(20):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(.1)


def stamp():
    return datetime.now(timezone.utc).isoformat()


def check_deadline(name="training_cutoff"):
    if time.time() >= datetime.fromisoformat(CONFIG[name]).timestamp():
        raise TimeoutError(f"Campaign {name} reached")


def setup():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert torch.cuda.is_available()


def metadata(path, indices):
    indices = np.asarray(indices, dtype=np.int64)
    with h5py.File(path, "r") as stream:
        labels = np.asarray(stream["known_label"], dtype=np.float32).T[indices]
        noise = np.asarray(stream["receiver_noise_seed"]).reshape(-1).astype(np.int64)[indices]
        refs = stream["component_nominal_JNR_dB"]
        powers = np.full(labels.shape, np.nan)
        for row, index in enumerate(indices):
            values = np.asarray(stream[refs[0, index]]).reshape(-1)
            powers[row, labels[row].astype(bool)] = values
    active_power = np.where(labels.astype(bool), powers, -np.inf)
    relative = powers - active_power.max(axis=1, keepdims=True)
    weak = labels.astype(bool) & (labels.sum(1, keepdims=True) > 1) & (relative <= -10 + 1e-6)
    return {"target": labels, "identity": noise, "power": powers,
            "relative_power_db": relative, "weak_mask": weak}


def role_paths(split):
    manifest = read(BASELINES / split / "data/manifest.json")
    return {
        "fit": (Path(manifest["fit_path"]), manifest["fit_A"]),
        "check": (Path(manifest["dev_path"]), manifest["early_stop"]),
        "calibration": (Path(manifest["dev_path"]), manifest["calibration"]),
    }


def evaluate(records, logits, thresholds):
    result = evaluate_predictions(records["target"], logits, thresholds,
                                  records["weak_mask"], records["relative_power_db"])
    values, _, _, _ = helper.metrics(logits, records["target"], thresholds, QUOTAS["tail_codes"])
    result.update(tail_exact=values["tail_exact"], ECE=values["ECE"],
                  confident_wrong_fraction=values["confident_wrong_fraction"])
    return result


class ReadoutBundle(torch.nn.Module):
    def __init__(self, variant="baseline", stages=(), width=32):
        super().__init__()
        self.variant = variant
        self.width = width
        self.backbone = base.build_backbone("dfms_iq", num_classes=6, width=width, dropout=.2)
        self.heads = torch.nn.ModuleList()
        self.transforms = torch.nn.ModuleList()
        self.stages = []
        for stage in stages:
            self.add_stage(stage["targets"], stage["inputs"])

    def add_stage(self, targets, inputs, transform=None):
        self.transforms.append(transform if transform is not None else FeatureTransform(inputs, CONFIG["pca_components"]))
        self.heads.append(ResidualReadout(CONFIG["pca_components"], CONFIG["hidden"], len(targets)))
        self.stages.append({"targets": list(targets), "inputs": inputs})

    def feature_branches(self, values):
        temporal = self.backbone.forward_temporal_features(values)
        projected = self.backbone.feature_projection(self.backbone.global_pool(temporal))
        logits = self.backbone.forward_classifier(projected)
        return logits, {"projected_global": projected,
                        "global_stats": segment_statistics(temporal, (1,)),
                        "segment": segment_statistics(temporal)}

    def forward(self, values):
        logits, features = self.feature_branches(values)
        if self.stages:
            chosen = features[self.variant]
            for stage, head, transform in zip(self.stages, self.heads, self.transforms):
                logits[:, stage["targets"]] += head(transform(chosen))
        return logits

    def save(self, path, thresholds, info):
        helper.atomic_torch_save({"format": "w32-readout-integration-v1", "variant": self.variant,
            "stages": self.stages, "width": self.width, "model_state_dict": self.state_dict(),
            "thresholds": list(thresholds), "info": info}, Path(path))


def load_bundle(path):
    state = torch.load(path, map_location="cpu", weights_only=False)
    model = ReadoutBundle(state["variant"], state["stages"], state["width"])
    model.load_state_dict(state["model_state_dict"])
    return model.cuda().eval().requires_grad_(False), state


def load_baseline(split, seed):
    folder = BASELINES / split / "w32" / f"M_{seed}"
    receipt = read(folder / "receipt.json")
    assert base.sha(folder / "best_BCE.pth") == receipt["checkpoint_sha256"]
    model = ReadoutBundle().cuda().eval().requires_grad_(False)
    state = torch.load(folder / "best_BCE.pth", map_location="cuda", weights_only=False)
    model.backbone.load_state_dict(state["model_state_dict"])
    return model, read(folder / "calibration.json"), receipt


def collect(model, inputs):
    model.eval()
    chunks = {name: [] for name in ("logits", "projected_global", "global_stats", "segment")}
    with torch.inference_mode():
        for batch in inputs.split(64):
            z, branches = model.feature_branches(batch.cuda())
            chunks["logits"].append(z.cpu().numpy())
            for name, values in branches.items():
                chunks[name].append(values.cpu().numpy())
    return {name: np.concatenate(values) for name, values in chunks.items()}


def cache_features(split, seed, role, model, path, indices):
    directory = ROOT / split / "cache" / f"M_{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    cache = directory / f"{role}.npz"
    receipt = cache.with_suffix(".json")
    identity_hash = hashlib.sha256(np.asarray(indices, dtype="<i8").tobytes()).hexdigest()
    binding = {"data_path": str(path), "data_sha256": base.sha(path), "indices_sha256": identity_hash,
               "backbone_sha256": base.sha(BASELINES / split / "w32" / f"M_{seed}" / "best_BCE.pth")}
    if cache.exists() and receipt.exists():
        saved = read(receipt)
        assert saved["binding"] == binding and saved["sha256"] == base.sha(cache)
        with np.load(cache, allow_pickle=False) as data:
            return dict(data)
    inputs, labels = base.load(path, indices)
    records = metadata(path, indices)
    np.testing.assert_array_equal(records["target"], labels)
    records.update(collect(model, inputs))
    temporary = cache.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, **records)
    temporary.replace(cache)
    write(receipt, {"binding": binding, "sha256": base.sha(cache), "created_utc": stamp()})
    return records


def cached_logits(model, records):
    logits = records["logits"].astype(np.float64).copy()
    if model.stages:
        values = torch.from_numpy(records[model.variant]).cuda()
        with torch.inference_mode():
            for stage, head, transform in zip(model.stages, model.heads, model.transforms):
                logits[:, stage["targets"]] += head(transform(values)).cpu().numpy()
    return logits
