"""DataLoader construction and deterministic worker seeding."""
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from .dataset import IQDataset


def seed_worker(_):
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)


def make_dataset(manifest, root, split, role, config):
    from pathlib import Path
    key = 'train' if role == 'fit' else 'development'
    return IQDataset(Path(root) / manifest['files'][key]['path'], split['roles'][role], config['preprocess'], cache=config['cache'])


def make_loader(dataset, config, *, training=False, sampler=None, batch_size=None, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size or config['batch_size'],
                      shuffle=training and sampler is None, sampler=sampler,
                      num_workers=config['workers'], pin_memory=config['pin_memory'],
                      persistent_workers=config['persistent_workers'] and config['workers'] > 0,
                      worker_init_fn=seed_worker, generator=generator)
