"""Training-only normalization/PCA with portable tensor state."""
import numpy as np
import torch
from torch import nn
from sklearn.utils.extmath import randomized_svd


class FeatureTransform(nn.Module):
    def __init__(self, dimensions, outputs):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dimensions))
        self.register_buffer("scale", torch.ones(dimensions))
        self.register_buffer("center", torch.zeros(dimensions))
        self.register_buffer("components", torch.eye(outputs, dimensions))

    def forward(self, values):
        return ((values - self.mean) / self.scale - self.center) @ self.components.T

    def project(self, values):
        state = {key: value.detach().cpu().numpy() for key, value in self.state_dict().items()}
        return np.ascontiguousarray(((values - state["mean"]) / state["scale"] - state["center"])
                                   @ state["components"].T, dtype=np.float32)

    @classmethod
    def fit(cls, values, dimensions=128, seed=2050, role="fit"):
        if role != "fit":
            raise ValueError("Normalization/PCA may only be fitted on fit identities")
        values = np.asarray(values, dtype=np.float32)
        if values.ndim != 2 or min(values.shape) < dimensions or not np.isfinite(values).all():
            raise ValueError("Insufficient or non-finite training features")
        mean = values.mean(0, dtype=np.float64).astype(np.float32)
        scale = np.maximum(values.std(0, dtype=np.float64), 1e-5).astype(np.float32)
        normalized = np.ascontiguousarray((values - mean) / scale)
        center = normalized.mean(0, dtype=np.float64).astype(np.float32)
        _, _, components = randomized_svd(normalized - center, n_components=dimensions,
                                         n_iter=4, random_state=seed, flip_sign=True)
        model = cls(values.shape[1], dimensions)
        model.load_state_dict({key: torch.from_numpy(value.copy()) for key, value in
                               dict(mean=mean, scale=scale, center=center,
                                    components=components.astype(np.float32)).items()})
        return model

    @classmethod
    def from_npz(cls, path):
        with np.load(path, allow_pickle=False) as saved:
            model = cls(len(saved["mean"]), saved["components"].shape[0])
            model.load_state_dict({key: torch.from_numpy(saved[key].copy()) for key in
                                   ("mean", "scale", "center", "components")})
        return model
