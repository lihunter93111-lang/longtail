"""Portable affine fusion with per-label fallback and shared-backbone execution."""
from common import *


def architecture(model):
    shapes = []
    for head in model.heads:
        first = next(layer for layer in head.modules() if isinstance(layer,torch.nn.Linear))
        shapes.append({"dimensions":first.in_features,"hidden":first.out_features})
    return {"variant": model.variant, "stages": model.stages, "width": model.width,"head_shapes":shapes}


def make_readout(options):
    if "head_shapes" not in options:
        return ReadoutBundle(options["variant"], options["stages"], options["width"])
    assert len(options["stages"])==len(options["head_shapes"])
    model = ReadoutBundle(options["variant"],width=options["width"])
    for stage,shape in zip(options["stages"],options["head_shapes"]):
        model.transforms.append(FeatureTransform(stage["inputs"],shape["dimensions"]))
        model.heads.append(ResidualReadout(shape["dimensions"],shape["hidden"],len(stage["targets"])))
        model.stages.append(stage)
    return model


class StaticFusion(torch.nn.Module):
    def __init__(self, reference, specialist, state, mode="two_model", shared_backbone=False):
        super().__init__()
        self.reference = reference
        self.specialist = specialist
        self.fusion_state = state
        self.mode = mode
        self.shared_backbone = shared_backbone
        if shared_backbone:
            assert specialist is not None and not reference.stages
            self.reference.backbone = self.specialist.backbone

    def branches(self, values):
        if self.shared_backbone:
            base_logits, features = self.specialist.feature_branches(values)
            right = base_logits.clone()
            for stage, head, transform in zip(self.specialist.stages, self.specialist.heads, self.specialist.transforms):
                right[:, stage["targets"]] += head(transform(features[self.specialist.variant]))
            return base_logits, right
        left = self.reference(values) if self.reference.stages else self.reference.backbone(values)
        if self.specialist is None:
            return left, torch.zeros_like(left)
        right = self.specialist(values) if self.specialist.stages else self.specialist.backbone(values)
        return left, right

    def fuse(self, reference, specialist):
        left, right = reference.double(), specialist.double()
        if self.mode == "specialist_only":
            left, right = right, torch.zeros_like(right)
        result = reference.double().clone()
        state = self.fusion_state
        for row, label in enumerate(state["targets"]):
            column = LABELS.index(label)
            if column in state.get("fallback_columns", []):
                continue
            a,b = state["weights"][2*row:2*row+2]
            result[:, column] = a*left[:, column] + b*right[:, column] + state["biases"][row]
        return result

    def forward(self, values):
        return self.fuse(*self.branches(values))

    def save(self, path, thresholds, info):
        helper.atomic_torch_save({"format": "w32_static_fusion_v1", "reference": architecture(self.reference),
            "specialist": architecture(self.specialist) if self.specialist is not None else None,
            "fusion_state": self.fusion_state, "mode": self.mode, "shared_backbone": self.shared_backbone,
            "model_state_dict": self.state_dict(), "thresholds": list(thresholds), "info": info}, Path(path))


def load_fusion(path):
    state = torch.load(path, map_location="cpu", weights_only=False)
    reference = make_readout(state["reference"])
    specialist = make_readout(state["specialist"]) if state["specialist"] is not None else None
    if state["shared_backbone"]:
        for key,value in state["model_state_dict"].items():
            if key.startswith("reference.backbone."):
                other = key.replace("reference.backbone.", "specialist.backbone.", 1)
                torch.testing.assert_close(value, state["model_state_dict"][other], rtol=0, atol=0)
    model = StaticFusion(reference, specialist, state["fusion_state"], state["mode"], state["shared_backbone"])
    model.load_state_dict(state["model_state_dict"])
    return model.cuda().eval().requires_grad_(False), state
