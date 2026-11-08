"""Portable complete-model checkpoints and explicit historical-model import."""
import hashlib
import json
from pathlib import Path
import torch
from cjlt.models.classwise_fusion import ClasswiseReadoutModel, AMA_CLASSES
from cjlt.engine.feature_transform import FeatureTransform

FORMAT = "cjlt-classwise-readout-v1"


def file_sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def save_bundle(path, model, thresholds, metadata=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(format=FORMAT, classes=list(AMA_CLASSES), architecture=model.architecture(),
                   state_dict={key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
                   thresholds=[float(value) for value in thresholds], metadata=metadata or {})
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _checked_path(spec, parent):
    path = Path(spec["path"])
    if not path.is_absolute():
        path = parent / path
    if spec.get("sha256") and file_sha256(path) != spec["sha256"]:
        raise ValueError("Historical component SHA mismatch: {}".format(path))
    return path


def _state(path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    return payload.get("model", payload.get("model_state_dict", payload.get("state_dict", payload))), payload


def import_legacy_selection(path, seed):
    """Import the accepted component graph once; future reloads need only one bundle."""
    path = Path(path)
    selected = json.loads(path.read_text(encoding="utf-8"))
    if not selected.get("formal_report_acceptance"):
        raise ValueError("The supplied selection did not pass formal report acceptance")
    row = next((row for row in selected["models"] if row["seed"] == seed), None)
    if row is None or "isrj_head" not in row:
        raise ValueError("Expected an accepted ISRJ+SawFM/MFT component selection for this seed")
    segment = row["current_reference"]
    refs = segment["references"]
    weights, base = _state(_checked_path(refs["new"]["B"], path.parent))
    options = base["config"]["model"]["kwargs"]
    if any(options.get(key, False) for key in ("anti_aliased_downsampling",)):
        raise ValueError("Historical import currently requires the default DFMS backbone")
    model = ClasswiseReadoutModel(backbone={key: options[key] for key in ("width", "dropout")},
                                 expert=True, linear=True, local=True, paired=True)
    model.backbone.load_state_dict(weights, strict=True)
    model.expert.load_state_dict(_state(_checked_path(refs["FMB"], path.parent))[0], strict=True)
    linear = _state(_checked_path(refs["new"]["linear"], path.parent))[0]
    model.linear.load_state_dict({key.removeprefix("linear."): value for key, value in linear.items()}, strict=True)
    model.local.load_state_dict(_state(_checked_path(refs["new"]["local"], path.parent))[0], strict=True)
    model.paired.load_state_dict(_state(_checked_path(refs["target_head"], path.parent))[0], strict=True)
    for name, targets, head, transform in [
        ("sawmft", ["SawFM", "MFT"], segment["residual_head"], segment["transform"]),
        ("isrj", ["ISRJ"], row["isrj_head"], row["feature_transform"]),
    ]:
        projection = FeatureTransform.from_npz(_checked_path(transform, path.parent))
        model.add_readout(name, targets, transform=projection)
        model.readouts[name].load_state_dict(_state(_checked_path(head, path.parent))[0], strict=True)
    return model.eval(), row["thresholds"], dict(imported_selection_sha256=file_sha256(path), seed=seed)


def load_bundle(path, seed=2050):
    path = Path(path)
    if path.suffix == ".json":
        return import_legacy_selection(path, seed)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format") != FORMAT or tuple(payload.get("classes", ())) != AMA_CLASSES:
        raise ValueError("Expected a native AMA-six-label model bundle; use explicit legacy import otherwise")
    model = ClasswiseReadoutModel(**payload["architecture"])
    model.load_state_dict(payload["state_dict"], strict=True)
    thresholds = payload["thresholds"]
    if len(thresholds) != 6 or any(not 0 <= value <= 1 for value in thresholds):
        raise ValueError("Invalid frozen thresholds")
    return model.eval(), thresholds, payload["metadata"]
