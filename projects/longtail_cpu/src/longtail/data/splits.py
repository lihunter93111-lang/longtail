"""One identity-safe, combination-stratified split shared by train and eval."""
from pathlib import Path
import numpy as np
from .dataset import read_metadata
from ..io import fingerprint, sha256


def codes(labels):
    return labels.astype(np.int64) @ (1 << np.arange(labels.shape[1]))


def build_split(manifest, root, config):
    train_path = Path(root) / manifest['files']['train']['path']
    dev_path = Path(root) / manifest['files']['development']['path']
    train_y, train_id, _ = read_metadata(train_path)
    dev_y, dev_id, _ = read_metadata(dev_path)
    if np.intersect1d(train_id, dev_id).size:
        raise ValueError('Train and development identities overlap')
    if len(manifest['classes']) != train_y.shape[1]:
        raise ValueError('Dataset label contract mismatch')
    actual = {}
    for name, path in [('train', train_path), ('development', dev_path)]:
        actual[name] = sha256(path) if config['verify_hashes'] else manifest['files'][name]['sha256']
        if actual[name] != manifest['files'][name]['sha256']:
            raise ValueError(f'{name} hash mismatch')
    grouped = codes(dev_y)
    rng = np.random.default_rng(config['split_seed'])
    roles = {key: [] for key in ('calibration', 'check', 'development_test')}
    for code in range(1, 64):
        pool = np.flatnonzero(grouped == code)
        required = sum(config['per_combination'].values())
        if len(pool) != required:
            raise ValueError(f'Combination {code}: expected {required} development samples, found {len(pool)}')
        rng.shuffle(pool)
        offset = 0
        for role in roles:
            count = config['per_combination'][role]
            roles[role].extend(pool[offset:offset + count].tolist())
            offset += count
    result = {'classes': manifest['classes'], 'dataset_id': manifest['dataset_id'],
              'source_sha256': actual, 'split_seed': config['split_seed'],
              'roles': {'fit': list(range(len(train_y))), **{k: sorted(v) for k, v in roles.items()}},
              'fit_label_counts': train_y.sum(0).astype(int).tolist(),
              'fit_combination_counts': np.bincount(codes(train_y), minlength=64)[1:].tolist(),
              'train_development_identity_overlap': 0}
    result['sha256'] = fingerprint(result)
    return result


def validate_split(split, manifest):
    payload = {k: v for k, v in split.items() if k != 'sha256'}
    if fingerprint(payload) != split['sha256']:
        raise ValueError('Saved split manifest was modified')
    if split['dataset_id'] != manifest['dataset_id'] or split['classes'] != manifest['classes'] or split['source_sha256'] != {k: v['sha256'] for k, v in manifest['files'].items()}:
        raise ValueError('Checkpoint split belongs to another dataset')
    dev_roles = [split['roles'][key] for key in ('calibration', 'check', 'development_test')]
    joined = [x for role in dev_roles for x in role]
    if len(set(joined)) != len(joined):
        raise ValueError('Development roles overlap')
