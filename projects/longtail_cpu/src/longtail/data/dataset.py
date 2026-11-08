"""Worker-safe lazy HDF5 loading, independent of preprocessing and models."""
import os
from functools import lru_cache
from pathlib import Path
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from .preprocess import transform
from .power_truth import read_power_truth


def read_metadata(path):
    with h5py.File(path, 'r') as f:
        labels = np.asarray(f['label_selected']).T
        if 'sample_id' in f:
            if 'pair_id' not in f or f.attrs.get('identity_schema') != 'one-bit-pairs-v1':
                raise ValueError('Explicit paired sample IDs require a declared pair identity schema')
            seeds = np.asarray(f['sample_id']).reshape(-1).astype(np.int64)
            if f['pair_id'].size != len(seeds) or f['receiver_noise_seed'].size != len(seeds):
                raise ValueError('Paired identity metadata length mismatch')
        else:
            seeds = np.asarray(f['receiver_noise_seed']).reshape(-1).astype(np.int64)
        signal_shape = f['iq_complex'].shape
    if labels.ndim != 2 or labels.shape[1] != 6 or not np.isin(labels, [0, 1]).all() or not (labels.sum(1) > 0).all():
        raise ValueError('Expected six-label nonempty binary targets')
    if len(labels) != len(seeds) or signal_shape[1] != len(labels) or len(np.unique(seeds)) != len(seeds):
        raise ValueError('Invalid sample identities or signal shape')
    return labels.astype(np.uint8), seeds, signal_shape[0]


@lru_cache(maxsize=2)
def _cached_signals(path, modified_ns, size):
    # Decode compressed column chunks once, sharing read-only raw arrays across roles.
    with h5py.File(path, 'r') as f:
        raw = np.asarray(f['iq_complex'])
    if raw.dtype.fields:
        values = np.stack([raw['real'].T, raw['imag'].T], axis=1)
    elif np.iscomplexobj(raw):
        values = np.stack([raw.real.T, raw.imag.T], axis=1)
    else:
        raise TypeError('Expected complex I/Q storage')
    values = np.ascontiguousarray(values, dtype=np.float32)
    values.flags.writeable = False
    return values


class IQDataset(Dataset):
    def __init__(self, path, indices, preprocess, cache="none"):
        self.path = str(Path(path).resolve())
        self.labels, self.seeds, self.length = read_metadata(path)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.preprocess = preprocess
        self.power_truth = read_power_truth(self.path, self.labels)
        self._handle = None
        self._pid = None
        stat = Path(self.path).stat()
        if cache not in ('memory', 'none'):
            raise ValueError('Unknown dataset cache mode')
        self._signals = _cached_signals(self.path, stat.st_mtime_ns, stat.st_size) if cache == 'memory' else None
        if self.indices.ndim != 1 or len(self.indices) == 0 or self.indices.min() < 0 or self.indices.max() >= len(self.labels):
            raise ValueError('Invalid sample indices')

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        index = int(self.indices[position])
        if self._signals is not None:
            values = self._signals[index]
        else:
            if self._handle is None or self._pid != os.getpid():
                self.close()
                self._handle = h5py.File(self.path, 'r')
                self._pid = os.getpid()
            raw = self._handle['iq_complex'][:, index]
            if raw.dtype.fields:
                values = np.stack([raw['real'], raw['imag']])
            elif np.iscomplexobj(raw):
                values = np.stack([raw.real, raw.imag])
            else:
                raise TypeError('iq_complex must be complex or have real/imag fields')
        result = {'signal': torch.from_numpy(transform(values, self.preprocess)),
                'target': torch.from_numpy(self.labels[index].astype(np.float32)),
                'index': index, 'identity': int(self.seeds[index])}
        if self.power_truth is not None:
            result.update({key: torch.from_numpy(value[index].copy()) for key, value in self.power_truth.items()})
        return result

    def close(self):
        if self._handle is not None:
            self._handle.close()
        self._handle = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_handle'] = None
        state['_pid'] = None
        return state
