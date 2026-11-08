from copy import deepcopy
import json
import h5py
import numpy as np
import pytest
import torch
from longtail.config import load_config
from longtail.io import sha256, write_json


@pytest.fixture(autouse=True)
def cpu_threads():
    torch.set_num_threads(2)


@pytest.fixture
def experiment(tmp_path):
    classes = ['FMB', 'AMA', 'SawFM', 'SinFM', 'MFT', 'ISRJ']
    files = {}
    for role, repeat, offset in [('train', 4, 10000), ('development', 3, 20000)]:
        codes = np.repeat(np.arange(1, 64), repeat)
        y = ((codes[:, None] >> np.arange(6)) & 1).astype(np.uint8)
        rng = np.random.default_rng(offset)
        z = np.empty((64, len(y)), dtype=[('real', '<f8'), ('imag', '<f8')])
        z['real'] = rng.normal(size=z.shape)
        z['imag'] = rng.normal(size=z.shape)
        path = tmp_path / (role + '.mat')
        with h5py.File(path, 'w') as f:
            f['iq_complex'] = z
            f['label_selected'] = y.T
            f['receiver_noise_seed'] = np.arange(len(y))[None, :] + offset
        files[role] = {'path': path.name, 'samples': len(y), 'bytes': path.stat().st_size,
                       'sha256': sha256(path), 'label_counts': y.sum(0).tolist(),
                       'combination_counts': [repeat] * 63}
    manifest = {'dataset_id': 'fixture', 'classes': classes, 'files': files}
    write_json(tmp_path / 'dataset.json', manifest)
    config = deepcopy(load_config('configs/baseline.json'))
    config['data']['manifest'] = str(tmp_path / 'dataset.json')
    config['data']['per_combination'] = {'calibration': 1, 'check': 1, 'development_test': 1}
    config['dataloader'].update(batch_size=16, workers=0, pin_memory=False)
    config['model']['kwargs']['width'] = 4
    config['train'].update(epochs=1, device='cpu')
    config['run']['output_dir'] = str(tmp_path / 'run')
    write_json(tmp_path / 'config.json', config)
    return tmp_path, config, manifest
