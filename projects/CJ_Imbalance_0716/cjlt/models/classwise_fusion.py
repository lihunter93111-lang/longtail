"""DFMS backbone plus explicit replacement and additive label branches."""
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
