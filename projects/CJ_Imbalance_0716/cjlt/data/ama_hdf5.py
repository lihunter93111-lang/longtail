"""Explicit AMA label contract, lazy complex I/Q loading and disjoint roles."""
import json
import os
from pathlib import Path
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from cjlt.models.classwise_fusion import AMA_CLASSES


class AMAHDF5Dataset(Dataset):
    def __init__(self, path, indices=None, classes=AMA_CLASSES):
        if tuple(classes) != AMA_CLASSES:
            raise ValueError("This adapter requires the explicit AMA six-label order, not VFT")
        self.path = str(Path(path).resolve())
        with h5py.File(self.path, "r") as stream:
            declared = stream.attrs.get("classes_json")
            if declared is not None and tuple(json.loads(declared)) != AMA_CLASSES:
                raise ValueError("HDF5 label contract differs from configuration")
            self.labels = np.asarray(stream["label_selected"]).T.astype(np.float32)
            self.noise_ids = np.asarray(stream["receiver_noise_seed"]).reshape(-1).astype(np.int64)
            identity_key = "sample_id" if "sample_id" in stream else "receiver_noise_seed"
            self.identities = np.asarray(stream[identity_key]).reshape(-1).astype(np.int64)
            self.length, count = stream["iq_complex"].shape
            self.weak_mask = self.relative_power_db = None
            if stream.attrs.get("power_truth_schema") == "per-label-v1":
                self.relative_power_db = np.asarray(stream["relative_power_dB"]).T
                self.weak_mask = np.asarray(stream["weak_component_mask"]).T.astype(bool)
                active = self.labels.astype(bool)
                weak_gap = float(stream.attrs["weak_gap_db"])
                if self.relative_power_db.shape != active.shape or self.weak_mask.shape != active.shape:
                    raise ValueError("Power truth shape mismatch")
                if not np.isfinite(self.relative_power_db[active]).all() or not np.isnan(self.relative_power_db[~active]).all():
                    raise ValueError("Power truth requires finite active and NaN absent labels")
                expected = active & (active.sum(1, keepdims=True) > 1) & (self.relative_power_db <= -weak_gap + 1e-6)
                if not np.array_equal(expected, self.weak_mask):
                    raise ValueError("Weak mask is inconsistent with per-component power")
        if self.labels.shape != (count, 6) or not np.isin(self.labels, [0, 1]).all() or not (self.labels.sum(1) > 0).all():
            raise ValueError("Expected nonempty six-label binary targets")
        if len(self.identities) != count or len(self.noise_ids) != count or len(set(self.identities)) != count:
            raise ValueError("Invalid or duplicated sample identities")
        self.indices = np.arange(count) if indices is None else np.asarray(indices, dtype=np.int64)
        if self.indices.ndim != 1 or not len(self.indices) or len(np.unique(self.indices)) != len(self.indices):
            raise ValueError("Role indices must be nonempty and unique")
        if self.indices.min() < 0 or self.indices.max() >= count:
            raise ValueError("Role indices exceed data size")
        self._handle = None
        self._pid = None

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, row):
        index = int(self.indices[row])
        if self._handle is None or self._pid != os.getpid():
            self.close()
            self._handle = h5py.File(self.path, "r")
            self._pid = os.getpid()
        value = self._handle["iq_complex"][:, index]
        if value.dtype.fields:
            signal = np.stack((value["real"], value["imag"]))
        elif np.iscomplexobj(value):
            signal = np.stack((value.real, value.imag))
        else:
            raise ValueError("iq_complex must contain complex or real/imag compound data")
        signal = np.asarray(signal, dtype=np.float32)
        signal /= np.sqrt(np.mean(signal[0] ** 2 + signal[1] ** 2) + 1e-12)
        if not np.isfinite(signal).all():
            raise ValueError("Non-finite I/Q")
        result = dict(signal=torch.from_numpy(np.ascontiguousarray(signal)),
                      target=torch.from_numpy(self.labels[index].copy()), identity=int(self.identities[index]))
        if self.weak_mask is not None:
            result.update(weak_mask=torch.from_numpy(self.weak_mask[index].copy()),
                          relative_power_db=torch.from_numpy(self.relative_power_db[index].copy()))
        return result

    def close(self):
        if self._handle is not None:
            self._handle.close()
        self._handle = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_handle"] = None
        state["_pid"] = None
        return state


def load_roles(data):
    if tuple(data.get("classes", ())) != AMA_CLASSES:
        raise ValueError("Declare data.classes explicitly in AMA order")
    split = json.loads(Path(data["split"]).read_text(encoding="utf-8"))
    split = split.get("roles", split)
    roles = {}
    try:
        for role in ("fit", "calibration", "check", "report"):
            source = data["train"] if role == "fit" else data["development"]
            key = "development_test" if role == "report" and "development_test" in split else role
            roles[role] = AMAHDF5Dataset(source, split[key], data["classes"])
        for number, (name, left) in enumerate(roles.items()):
            for other, right in list(roles.items())[number + 1:]:
                for field in ("identities", "noise_ids"):
                    a = getattr(left, field)[left.indices]
                    b = getattr(right, field)[right.indices]
                    if np.intersect1d(a, b).size:
                        raise ValueError("Identity leakage between {} and {}".format(name, other))
    except Exception:
        for dataset in roles.values():
            dataset.close()
        raise
    return roles
