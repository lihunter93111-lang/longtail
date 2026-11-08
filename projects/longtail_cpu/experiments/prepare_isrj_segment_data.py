"""Fresh natural identities for isolated ISRJ readout validation."""
import argparse
from copy import deepcopy
import os
from pathlib import Path

import h5py
import numpy as np

from longtail.data import v3_fresh
from longtail.data.signals import DEFAULT_PHYSICS
from longtail.io import read_json, write_json, sha256, fingerprint
from prepare_staged_validation_data import save_blocks
from prepare_replication_temporal_data import distribution_worker, file_record

ROOT = Path(__file__).resolve().parents[1]


def prepare(output, previous, workers):
    if output.exists():
        raise FileExistsError('Use a fresh data directory')
    prior = read_json(previous/'plan.json')
    source_path = Path(prior['data']['manifest'])
    source = read_json(source_path)
    output.mkdir(parents=True)
    definitions = {'natural': (4260000001, 18144, (10., 30.)),
                   'ood_wide': (4261000001, 4032, (10., 40.))}
    plan = {'schema': 'isrj-segment-data-v1', 'definitions': definitions,
            'classes': source['classes'], 'physics': DEFAULT_PHYSICS,
            'source_manifest': str(source_path), 'source_sha256': sha256(source_path),
            'per_combination': {'calibration': 128, 'check': 32, 'development_test': 128},
            'fit': 'unchanged natural32k; no new paired training in this feature-only experiment',
            'matlab_bitwise_parity': False,
            'generator_sources': {str(p.relative_to(ROOT)): sha256(p) for p in [Path(__file__),
                ROOT/'src/longtail/data/signals.py', ROOT/'src/longtail/data/v3_fresh.py',
                ROOT/'experiments/prepare_staged_validation_data.py', ROOT/'experiments/prepare_replication_temporal_data.py']}}
    plan['sha256'] = fingerprint(plan)
    write_json(output/'generation_plan.json', plan)
    files, noise = {}, []
    for tag, (first, count, jnr) in definitions.items():
        seeds = first+np.arange(count, dtype=np.uint64)
        assert seeds.max() < 2**32
        codes = np.repeat(np.arange(1, 64), count//63)
        tasks = [(codes[i:i+64], seeds[i:i+64], source['classes'], jnr) for i in range(0, count, 64)]
        path = output/(tag+'.h5')
        save_blocks(path, tasks, v3_fresh.initialize, distribution_worker,
                    DEFAULT_PHYSICS, count, workers=workers)
        files[tag] = {**file_record(path), 'path': str(path)}
        with h5py.File(path) as f:
            noise.extend(f['receiver_noise_seed'][:].ravel().tolist())
    assert len(np.unique(noise)) == len(noise)
    checked = []
    for directory in [ROOT.parent/'datasets/longtail', ROOT/'data', ROOT/'backup']:
        for path in sorted(directory.rglob('*')):
            if path.suffix not in ['.h5', '.mat'] or output in path.parents or not h5py.is_hdf5(path):
                continue
            with h5py.File(path) as f:
                if 'receiver_noise_seed' not in f:
                    continue
                assert not np.intersect1d(noise, f['receiver_noise_seed'][:].ravel()).size, str(path)
            checked.append(str(path.resolve()))
    os.link(source_path.parent/source['files']['train']['path'], output/'train.h5')
    manifest = deepcopy(source)
    manifest.update(dataset_id='ama-v3-isrj-segment-20261105', generation_plan_sha256=plan['sha256'])
    manifest['files']['train']['path'] = 'train.h5'
    manifest['files']['development'] = {**files['natural'], 'path': 'natural.h5'}
    write_json(output/'dataset.json', manifest)
    write_json(output/'generation_complete.json', {'status': 'completed', 'manifest': str(output/'dataset.json'),
        'files': files, 'new_background_count': len(noise), 'identity_overlap_existing': 0,
        'checked_identity_files': checked, 'plan_sha256': plan['sha256']})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--previous', required=True, type=Path)
    p.add_argument('--workers', type=int, default=8)
    a = p.parse_args()
    prepare(a.output.resolve(), a.previous.resolve(), a.workers)
