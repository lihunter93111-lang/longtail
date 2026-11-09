"""Conservative, label-local fusion for frozen AMA model bundles.

The reference route remains the default prediction.  A frozen specialist is
allowed to correct only declared target labels through an uncertainty gate;
the gate is deliberately low-capacity so it can be fitted on an independent
identity pool without becoming a third classifier.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F

from cjlt.models.classwise_fusion import AMA_CLASSES, FrozenLogitFusion


def _as_logits(values, name):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(AMA_CLASSES) or not np.isfinite(values).all():
        raise ValueError("{} must be finite [N, 6] logits".format(name))
    return values


def _columns(targets):
    names = tuple(targets)
    if not names or len(set(names)) != len(names) or any(name not in AMA_CLASSES for name in names):
        raise ValueError("targets must be a non-empty unique AMA label subset")
    return names, tuple(AMA_CLASSES.index(name) for name in names)


def _threshold_logits(thresholds):
    values = np.asarray(thresholds, dtype=np.float64)
    if values.shape != (len(AMA_CLASSES),) or np.any((values < 0) | (values > 1)):
        raise ValueError("expected six probability thresholds")
    values = np.clip(values, 1e-5, 1.0 - 1e-5)
    return np.log(values / (1.0 - values))


def _fit(model, inputs, target, seed, epochs, learning_rate, gate_penalty=0.0, anchor_penalty=0.0):
    if int(epochs) < 1 or not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("Use a positive fusion training budget and learning rate")
    if not np.isfinite(target).all() or not np.isin(target, [0, 1]).all():
        raise ValueError("Fusion targets must be binary")
    torch.manual_seed(int(seed))
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate), weight_decay=1e-4)
    x = torch.as_tensor(inputs, dtype=torch.float32)
    y = torch.as_tensor(target, dtype=torch.float32)
    for _ in range(int(epochs)):
        optimizer.zero_grad(set_to_none=True)
        output, gate, reference = model(x)
        loss = F.binary_cross_entropy_with_logits(output, y)
        if gate_penalty:
            loss = loss + float(gate_penalty) * gate.mean()
        if anchor_penalty:
            loss = loss + float(anchor_penalty) * F.binary_cross_entropy_with_logits(
                output, torch.sigmoid(reference).detach()
            )
        loss.backward()
        optimizer.step()
    model.eval()
    return model


class _StaticAffine(torch.nn.Module):
    """A necessary low-capacity baseline: one affine blend per label."""

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 1)
        with torch.no_grad():
            self.linear.weight.copy_(torch.tensor([[1.0, 0.0]]))
            self.linear.bias.zero_()

    def forward(self, inputs):
        reference = inputs[:, :1]
        return self.linear(inputs[:, :2]), torch.ones_like(reference), reference


class _SelectiveGate(torch.nn.Module):
    """Reference logit plus a sparse, confidence-aware specialist residual."""

    def __init__(self):
        super().__init__()
        self.specialist_scale = torch.nn.Parameter(torch.tensor(1.0))
        self.specialist_bias = torch.nn.Parameter(torch.tensor(0.0))
        self.gate = torch.nn.Linear(5, 1)
        torch.nn.init.zeros_(self.gate.weight)
        torch.nn.init.constant_(self.gate.bias, -2.5)

    def forward(self, inputs):
        reference, specialist, reference_margin, specialist_margin = (
            inputs[:, 0:1], inputs[:, 1:2], inputs[:, 2:3], inputs[:, 3:4]
        )
        features = torch.cat((reference, specialist, reference - specialist,
                              reference_margin.abs(), specialist_margin.abs()), dim=1)
        gate = torch.sigmoid(self.gate(features))
        aligned_specialist = self.specialist_scale * specialist + self.specialist_bias
        return reference + gate * (aligned_specialist - reference), gate, reference


def fit_affine_fusion(reference_logits, specialist_logits, target, targets=AMA_CLASSES, seed=2050,
                      epochs=300, learning_rate=0.04):
    """Fit a static per-label blend; exposed separately to force target labels."""
    reference = _as_logits(reference_logits, "reference_logits")
    specialist = _as_logits(specialist_logits, "specialist_logits")
    target = np.asarray(target, dtype=np.float64)
    names, columns = _columns(targets)
    if reference.shape != specialist.shape or target.shape != reference.shape:
        raise ValueError("reference, specialist, and target shapes must match")
    weights, biases = [], []
    for offset, column in enumerate(columns):
        model = _fit(_StaticAffine(), np.column_stack((reference[:, column], specialist[:, column])),
                     target[:, column:column + 1], int(seed) + offset, epochs, learning_rate)
        weights.append(float(model.linear.weight.detach()[0, 0]))
        weights.append(float(model.linear.weight.detach()[0, 1]))
        biases.append(float(model.linear.bias.detach()[0]))
    return dict(kind="affine", targets=list(names), weights=weights, biases=biases,
                epochs=int(epochs), learning_rate=float(learning_rate))


def fit_selective_fusion(reference_logits, specialist_logits, target, reference_thresholds,
                          specialist_thresholds, targets=("SawFM", "MFT", "ISRJ"), seed=2050,
                          epochs=500, learning_rate=0.03, gate_penalty=0.012, anchor_penalty=0.02):
    """Fit one conservative residual gate for each target label."""
    reference = _as_logits(reference_logits, "reference_logits")
    specialist = _as_logits(specialist_logits, "specialist_logits")
    target = np.asarray(target, dtype=np.float64)
    names, columns = _columns(targets)
    if reference.shape != specialist.shape or target.shape != reference.shape:
        raise ValueError("reference, specialist, and target shapes must match")
    reference_cut = _threshold_logits(reference_thresholds)
    specialist_cut = _threshold_logits(specialist_thresholds)
    states = []
    for offset, column in enumerate(columns):
        inputs = np.column_stack((reference[:, column], specialist[:, column],
                                  reference[:, column] - reference_cut[column],
                                  specialist[:, column] - specialist_cut[column]))
        model = _fit(_SelectiveGate(), inputs, target[:, column:column + 1], int(seed) + offset,
                     epochs, learning_rate, gate_penalty, anchor_penalty)
        states.append({key: value.detach().cpu().numpy().tolist() for key, value in model.state_dict().items()})
    return dict(kind="selective", targets=list(names), states=states, epochs=int(epochs),
                learning_rate=float(learning_rate), gate_penalty=float(gate_penalty),
                anchor_penalty=float(anchor_penalty), reference_thresholds=np.asarray(reference_thresholds).tolist(),
                specialist_thresholds=np.asarray(specialist_thresholds).tolist())


def apply_fusion(reference_logits, specialist_logits, state):
    """Return fused logits and one gate value per label (zero means reference only)."""
    reference = _as_logits(reference_logits, "reference_logits")
    specialist = _as_logits(specialist_logits, "specialist_logits")
    if reference.shape != specialist.shape:
        raise ValueError("reference and specialist logits must have matching shape")
    names, columns = _columns(state["targets"])
    fused = reference.copy()
    gates = np.zeros_like(reference)
    kind = state.get("kind")
    if kind == "affine":
        weights = np.asarray(state["weights"], dtype=np.float64).reshape(len(columns), 2)
        biases = np.asarray(state["biases"], dtype=np.float64)
        for row, column in enumerate(columns):
            fused[:, column] = weights[row, 0] * reference[:, column] + weights[row, 1] * specialist[:, column] + biases[row]
            gates[:, column] = 1.0
    elif kind == "selective":
        reference_cut = _threshold_logits(state["reference_thresholds"])
        specialist_cut = _threshold_logits(state["specialist_thresholds"])
        for row, column in enumerate(columns):
            inputs = np.column_stack((reference[:, column], specialist[:, column],
                                      reference[:, column] - reference_cut[column],
                                      specialist[:, column] - specialist_cut[column]))
            model = _SelectiveGate()
            model.load_state_dict({key: torch.as_tensor(value) for key, value in state["states"][row].items()})
            with torch.inference_mode():
                output, gate, _ = model(torch.as_tensor(inputs, dtype=torch.float32))
            fused[:, column] = output.numpy().reshape(-1)
            gates[:, column] = gate.numpy().reshape(-1)
    else:
        raise ValueError("Unknown fusion kind: {}".format(kind))
    return fused, gates


def train_fusion(config, roles, seed, output, device="cpu"):
    """Fit on fit, select thresholds on calibration, and never read report signals."""
    from cjlt.engine.longtail_evaluation import fit_thresholds
    from cjlt.engine.model_bundle import file_sha256, load_bundle, save_bundle
    from cjlt.engine.readout_trainer import collect_records, metrics

    options = config.get("fusion", {})
    kind = options.get("kind", "affine")
    if kind not in ("reference", "affine", "selective"):
        raise ValueError("Unknown fusion kind")
    paths = {key: Path(config[key].format(seed=seed)) for key in
             (("reference",) if kind == "reference" else ("reference", "specialist"))}
    hashes = {key: file_sha256(path) for key, path in paths.items()}
    reference, reference_thresholds, _ = load_bundle(paths["reference"], seed)
    reference.eval().requires_grad_(False)
    calibration = collect_records(reference, roles["calibration"], device=device)
    grid = np.linspace(0.05, 0.95, 181)
    thresholds = fit_thresholds(calibration["target"], calibration["logits"], grid=grid)
    reference_metrics = metrics(calibration, calibration["logits"], thresholds)
    state = dict(kind="reference", targets=[])
    model, measured = reference, reference_metrics
    if kind != "reference":
        specialist, specialist_thresholds, _ = load_bundle(paths["specialist"], seed)
        specialist.eval().requires_grad_(False)
        fit = collect_records(reference, roles["fit"], device=device)
        other_fit = collect_records(specialist, roles["fit"], device=device)
        other_calibration = collect_records(specialist, roles["calibration"], device=device)
        for left, right in ((fit, other_fit), (calibration, other_calibration)):
            for key in ("identity", "target", "weak_mask", "relative_power_db"):
                if key in left or key in right:
                    if key not in left or key not in right or not np.array_equal(left[key], right[key], equal_nan=True):
                        raise ValueError("Fusion inputs disagree for {}".format(key))
        targets = options.get("targets", list(AMA_CLASSES) if kind == "affine" else ["SawFM", "MFT", "ISRJ"])
        arguments = {key: options[key] for key in ("epochs", "learning_rate") if key in options}
        if kind == "affine":
            state = fit_affine_fusion(fit["logits"], other_fit["logits"], fit["target"],
                                      targets=targets, seed=seed, **arguments)
        else:
            arguments.update({key: options[key] for key in ("gate_penalty", "anchor_penalty") if key in options})
            state = fit_selective_fusion(fit["logits"], other_fit["logits"], fit["target"],
                                         reference_thresholds, specialist_thresholds,
                                         targets=targets, seed=seed, **arguments)
        logits, _ = apply_fusion(calibration["logits"], other_calibration["logits"], state)
        slack = float(options.get("fpr_slack", 0.0025))
        if not np.isfinite(slack) or slack < 0:
            raise ValueError("FPR slack must be finite and nonnegative")
        caps = [reference_metrics["per_label"][name]["negative_fpr"] + slack for name in AMA_CLASSES]
        thresholds = fit_thresholds(calibration["target"], logits, grid=grid, fpr_caps=caps)
        measured = metrics(calibration, logits, thresholds)
        model = FrozenLogitFusion(reference, specialist, state)
    slack = float(options.get("fpr_slack", 0.0025))
    guards = dict(
        fpr=all(measured["per_label"][name]["negative_fpr"] <= reference_metrics["per_label"][name]["negative_fpr"] + slack + 1e-12
                for name in AMA_CLASSES),
        fmb_f1=measured["per_label"]["FMB"]["f1"] >= reference_metrics["per_label"]["FMB"]["f1"] - 0.002,
    )
    weak_labels = [name for name in ("SawFM", "MFT", "ISRJ")
                   if reference_metrics["per_label"][name].get("weak_recall") is not None]
    guards["tail_weak_recall"] = bool(not weak_labels or np.mean([
        measured["per_label"][name]["weak_recall"] - reference_metrics["per_label"][name]["weak_recall"]
        for name in weak_labels]) >= -0.005)
    guards["passed"] = all(guards.values())
    if hashes != {key: file_sha256(path) for key, path in paths.items()}:
        raise ValueError("Source models changed during fusion fitting")
    info = dict(seed=seed, kind="frozen_fusion", fusion_state=state,
                optimizer_steps=int(state.get("epochs", 0)) * len(state["targets"]),
                source_model_sha256=hashes, calibration=measured, calibration_guard=guards,
                guard_policy="diagnostic; no automatic rejection or fallback", report_used=False)
    save_bundle(Path(output) / "model.pt", model, thresholds, info)
    (Path(output) / "history.json").write_text(json.dumps(info, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return info
