"""DFMS backbone plus explicit replacement and additive label branches."""
from copy import deepcopy
import torch
from torch import nn
from cjlt.models.backbone_adapter import build_backbone
from cjlt.models.segment_readout import ResidualReadout, segment_statistics
from cjlt.engine.feature_transform import FeatureTransform

AMA_CLASSES = ("FMB", "AMA", "SawFM", "SinFM", "MFT", "ISRJ")


class ClasswiseReadoutModel(nn.Module):
    def __init__(self, backbone=None, expert=False, linear=False, local=False,
                 paired=False, readouts=()):
        super().__init__()
        self.backbone_options = dict(backbone or {"width": 16, "dropout": 0.2})
        self.backbone = build_backbone("dfms_iq", num_classes=6, **self.backbone_options)
        dimensions = self.backbone.feature_dim
        self.expert = build_backbone("dfms_iq", num_classes=6, **self.backbone_options) if expert else None
        self.linear = nn.Linear(dimensions, 5) if linear else None
        self.local = ResidualReadout(dimensions, outputs=3) if local else None
        self.paired = ResidualReadout(dimensions, outputs=2) if paired else None
        self.readouts = nn.ModuleDict()
        self.transforms = nn.ModuleDict()
        self.readout_options = []
        for options in readouts:
            self.add_readout(**options)

    def add_readout(self, name, targets, source="segment", dimensions=128, hidden=32,
                    transform=None):
        if name in self.readouts or source not in ("global", "segment"):
            raise ValueError("Duplicate readout or unknown feature source")
        columns = [AMA_CLASSES.index(label) for label in targets]
        if not columns or len(set(columns)) != len(columns):
            raise ValueError("Readout targets must be distinct known labels")
        if self.expert is not None and 0 in columns:
            raise ValueError("Disable the replacing FMB expert before adding an FMB residual")
        inputs = self.backbone.feature_dim * (14 if source == "segment" else 1)
        self.transforms[name] = transform if transform is not None else FeatureTransform(inputs, dimensions)
        self.readouts[name] = ResidualReadout(dimensions, hidden, len(columns))
        self.readout_options.append(dict(name=name, targets=list(targets), source=source,
                                         dimensions=dimensions, hidden=hidden))

    def architecture(self):
        return dict(backbone=self.backbone_options, expert=self.expert is not None,
                    linear=self.linear is not None, local=self.local is not None,
                    paired=self.paired is not None, readouts=self.readout_options)

    def features(self, values):
        temporal = self.backbone.forward_temporal_features(values)
        global_features = self.backbone.feature_projection(self.backbone.global_pool(temporal))
        return global_features, temporal

    def classify_features(self, global_features, segmented=None, expert_logits=None):
        """Apply the same frozen decision graph to live or cached features."""
        logits = self.backbone.forward_classifier(global_features).clone()
        if self.linear is not None:
            logits[:, 1:] = self.linear(global_features)
        if self.local is not None:
            logits[:, [2, 4, 5]] += self.local(global_features)
        if self.paired is not None:
            # The paired SawFM/MFT head replaces the old local rows.
            if self.linear is None:
                raise ValueError("Paired residuals require the five-label linear head")
            logits[:, [2, 4]] = self.linear(global_features)[:, [1, 3]] + self.paired(global_features)
        for options in self.readout_options:
            name = options["name"]
            if options["source"] == "segment":
                if segmented is None:
                    raise ValueError("Segmented readouts require temporal statistics")
                features = segmented
            else:
                features = global_features
            columns = [AMA_CLASSES.index(label) for label in options["targets"]]
            logits[:, columns] += self.readouts[name](self.transforms[name](features))
        if self.expert is not None:
            if expert_logits is None:
                raise ValueError("The FMB expert requires its own logits")
            logits[:, 0] = expert_logits[:, 0]
        return logits

    def forward(self, values, return_features=False):
        global_features, temporal = self.features(values)
        segmented = segment_statistics(temporal) if any(
            item["source"] == "segment" for item in self.readout_options) else None
        expert_logits = self.expert(values) if self.expert is not None else None
        logits = self.classify_features(global_features, segmented, expert_logits)
        if return_features:
            return logits, global_features, temporal
        return logits


class FrozenLogitFusion(nn.Module):
    """Portable R/W inference; both complete branches are stored in the bundle."""

    def __init__(self, reference, specialist, fusion_state):
        super().__init__()
        if not all(isinstance(model, ClasswiseReadoutModel) for model in (reference, specialist)):
            raise ValueError("Fusion requires two classwise model bundles")
        self.reference = reference.eval().requires_grad_(False)
        self.specialist = specialist.eval().requires_grad_(False)
        self.fusion_state = deepcopy(fusion_state)
        targets = fusion_state["targets"]
        if not targets or len(set(targets)) != len(targets) or any(name not in AMA_CLASSES for name in targets):
            raise ValueError("Expected distinct AMA target labels")
        if fusion_state["kind"] not in ("affine", "selective"):
            raise ValueError("Expected affine or selective fusion")

    def architecture(self):
        return dict(reference=self.reference.architecture(), specialist=self.specialist.architecture(),
                    fusion_state=self.fusion_state)

    def fuse_logits(self, reference, specialist):
        # Match the original offline calculation: affine arithmetic is float64,
        # while the trained gate uses float32 inputs and weights.
        reference, specialist = reference.double(), specialist.double()
        fused = reference.clone()
        state = self.fusion_state
        for row, name in enumerate(state["targets"]):
            column = AMA_CLASSES.index(name)
            left, right = reference[:, column], specialist[:, column]
            if state["kind"] == "affine":
                alpha, beta = state["weights"][2 * row:2 * row + 2]
                fused[:, column] = alpha * left + beta * right + state["biases"][row]
            else:
                cuts = []
                for key in ("reference_thresholds", "specialist_thresholds"):
                    threshold = min(max(state[key][column], 1e-5), 1 - 1e-5)
                    cuts.append(reference.new_tensor(threshold).logit())
                inputs = torch.stack((left, right, left - cuts[0], right - cuts[1]), dim=1).float()
                a, b, ma, mb = inputs.unbind(1)
                features = torch.stack((a, b, a - b, ma.abs(), mb.abs()), dim=1)
                weights = {key: features.new_tensor(value) for key, value in state["states"][row].items()}
                gate = nn.functional.linear(features, weights["gate.weight"], weights["gate.bias"]).sigmoid().squeeze(1)
                aligned = weights["specialist_scale"] * b + weights["specialist_bias"]
                fused[:, column] = a + gate * (aligned - a)
        return fused

    def forward(self, values):
        return self.fuse_logits(self.reference(values), self.specialist(values))
