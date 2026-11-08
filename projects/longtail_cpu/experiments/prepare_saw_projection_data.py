"""Fresh identities and parameter-shift reports for SawFM projection validation."""
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


def definitions():
    result = {}
    for name, seed, count, jnr in [('natural', 4270000001, 18144, (10., 30.)),
                                  ('ood_wide', 4271000001, 4032, (10., 40.)),
                                  ('arrival_ood', 4272000001, 4032, (10., 30.)),
                                  ('saw_period_ood', 4273000001, 4032, (10., 30.))]:
        physics = deepcopy(DEFAULT_PHYSICS)
        if name == 'arrival_ood':
            physics['arrival_delay_s'] = [50e-6, 90e-6]
        if name == 'saw_period_ood':
            physics['SawFM']['sweep_period_s'] = [80e-6, 120e-6]
        result[name] = {'seed': seed, 'samples': count, 'jnr': jnr, 'physics': physics}
    return result


def prepare(output, previous, workers):
    if output.exists():
        raise FileExistsError('Use a new data directory')
    prior = read_json(previous/'plan.json')
    source_path = Path(prior['data']['manifest'])
    source = read_json(source_path)
    output.mkdir(parents=True)
    plan = {'schema': 'saw-projection-v1', 'definitions': definitions(), 'classes': source['classes'],
        'source_manifest': str(source_path), 'source_sha256': sha256(source_path),
        'per_combination': {'calibration': 128, 'check': 32, 'development_test': 128},
        'fit': 'unchanged natural32k; original10--30dB component JNR',
        'parameter_shift_reports': 'diagnostic only, never used for checkpoint or candidate selection',
        'arrival_convention': 'existing simulator circular np.roll, not a measured propagation channel',
        'matlab_bitwise_parity': False,
        'generator_sources': {str(p.relative_to(ROOT)): sha256(p) for p in [Path(__file__),
            ROOT/'src/longtail/data/signals.py', ROOT/'src/longtail/data/v3_fresh.py',
            ROOT/'experiments/prepare_staged_validation_data.py', ROOT/'experiments/prepare_replication_temporal_data.py']}}
    plan['sha256'] = fingerprint(plan)
    write_json(output/'generation_plan.json', plan)
    files, noise = {}, []
    for name, info in plan['definitions'].items():
        n = info['samples']
        seeds = info['seed']+np.arange(n, dtype=np.uint64)
        assert seeds.max() < 2**32
        codes = np.repeat(np.arange(1, 64), n//63)
        tasks = [(codes[i:i+64], seeds[i:i+64], source['classes'], info['jnr']) for i in range(0, n, 64)]
        path = output/(name+'.h5')
        save_blocks(path, tasks, v3_fresh.initialize, distribution_worker, info['physics'], n, workers=workers)
        files[name] = {**file_record(path), 'path': str(path), 'physics': info['physics'], 'component_jnr_db': info['jnr']}
        with h5py.File(path) as f:
            noise.extend(f['receiver_noise_seed'][:].ravel().tolist())
    assert len(np.unique(noise)) == len(noise)
    checked = []
    for folder in [ROOT.parent/'datasets/longtail', ROOT/'data', ROOT/'backup']:
        for path in sorted(folder.rglob('*')):
            if path.suffix not in ['.h5', '.mat'] or output in path.parents or not h5py.is_hdf5(path):
                continue
            with h5py.File(path) as f:
                if 'receiver_noise_seed' not in f:
                    continue
                assert not np.intersect1d(noise, f['receiver_noise_seed'][:].ravel()).size, str(path)
            checked.append(str(path.resolve()))
    os.link(source_path.parent/source['files']['train']['path'], output/'train.h5')
    manifest = deepcopy(source)
    manifest.update(dataset_id='ama-v3-saw-projection-20261107', generation_plan_sha256=plan['sha256'])
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
