"""Fresh evaluation identities for frozen segment-statistics probes."""
import argparse
from copy import deepcopy
import os
from pathlib import Path

import h5py
import numpy as np

from longtail.data import targeted_pairs, v3_fresh
from longtail.data.signals import DEFAULT_PHYSICS
from longtail.io import fingerprint, read_json, sha256, write_json
from prepare_staged_validation_data import save_blocks
from prepare_replication_temporal_data import distribution_worker, file_record

ROOT = Path(__file__).resolve().parents[1]


def prepare(output, previous, workers):
    if output.exists():
        raise FileExistsError('Use a fresh directory')
    prior = read_json(previous/'plan.json')
    source_path = Path(prior['data']['manifests']['targeted'])
    source = read_json(source_path)
    old_pairs = read_json(prior['data']['pair_manifest'])
    classes = source['classes']
    output.mkdir(parents=True)
    definitions = {'natural': (4240000001, 18144, (10., 30.)),
                   'ood_wide': (4241000001, 4032, (10., 40.))}
    tasks = []
    for edge in old_pairs['edges']:
        if edge['role'] != 'development_test':
            continue
        for gap in range(3):
            tasks.append((edge['background_code'], edge['target'], gap,
                          4250000001+len(tasks)*targeted_pairs.MAX_ATTEMPTS,
                          32, edge['edge_id'], classes))
    assert max(t[3]+targeted_pairs.MAX_ATTEMPTS for t in tasks) < 2**32
    plan = {'schema': 'segment-statistics-data-v1', 'natural': definitions,
            'pair_tasks': tasks, 'classes': classes, 'physics': DEFAULT_PHYSICS,
            'per_combination': {'calibration': 128, 'check': 32, 'development_test': 128},
            'fit': 'exact same natural32k and2736 conditional pairs as previous suite',
            'source_manifest': str(source_path), 'source_sha256': sha256(source_path),
            'generator_sources': {str(p.relative_to(ROOT)): sha256(p) for p in [
                Path(__file__), ROOT/'src/longtail/data/signals.py', ROOT/'src/longtail/data/v3_fresh.py',
                ROOT/'src/longtail/data/one_bit.py', ROOT/'src/longtail/data/targeted_pairs.py']},
            'matlab_bitwise_parity': False}
    plan['sha256'] = fingerprint(plan)
    write_json(output/'generation_plan.json', plan)
    files, new_noise = {}, []
    for tag, (first, count, jnr) in definitions.items():
        path = output/(tag+'.h5')
        codes = np.repeat(np.arange(1, 64), count//63)
        seeds = first+np.arange(count, dtype=np.uint64)
        blocks = [(codes[i:i+64], seeds[i:i+64], classes, jnr) for i in range(0, count, 64)]
        save_blocks(path, blocks, v3_fresh.initialize, distribution_worker,
                    DEFAULT_PHYSICS, count, workers=workers)
        files[tag] = {**file_record(path), 'path': str(path)}
        with h5py.File(path) as f:
            new_noise.extend(f['receiver_noise_seed'][:].ravel().tolist())
    pair_path = output/'pair_report.h5'
    save_blocks(pair_path, tasks, targeted_pairs.initialize, targeted_pairs.conditional_worker,
                DEFAULT_PHYSICS, 2304, paired=True, workers=workers)
    files['pair_report'] = {**file_record(pair_path), 'path': str(pair_path)}
    with h5py.File(pair_path) as f:
        y = f['label_selected'][:].T
        assert np.all((y[::2] ^ y[1::2]).sum(1) == 1)
        assert f['raw_difference_relative_error'][:].max() < 1e-12
        new_noise.extend(f['receiver_noise_seed'][0, ::2].tolist())
    assert len(np.unique(new_noise)) == len(new_noise)
    checked = []
    for directory in [ROOT.parent/'datasets/longtail', ROOT/'data', ROOT/'backup']:
        for path in sorted(directory.rglob('*')):
            if path.suffix not in ['.mat', '.h5'] or output in path.parents or not h5py.is_hdf5(path):
                continue
            with h5py.File(path) as f:
                if 'receiver_noise_seed' not in f:
                    continue
                assert not np.intersect1d(new_noise, f['receiver_noise_seed'][:].ravel()).size, str(path)
            checked.append(str(path.resolve()))
    os.link(source_path.parent/source['files']['train']['path'], output/'train.h5')
    manifest = deepcopy(source)
    manifest.update(dataset_id='ama-v3-segment-statistics-20261105', generation_plan_sha256=plan['sha256'])
    manifest['files']['train']['path'] = 'train.h5'
    manifest['files']['development'] = {**files['natural'], 'path': 'natural.h5'}
    write_json(output/'dataset.json', manifest)
    write_json(output/'generation_complete.json', {
        'status': 'completed', 'manifest': str(output/'dataset.json'), 'files': files,
        'pair_fit': old_pairs['files']['fit'], 'pair_edges': old_pairs['edges'],
        'new_background_count': len(new_noise), 'identity_overlap_existing': 0,
        'checked_identity_files': checked, 'plan_sha256': plan['sha256']})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--previous', required=True, type=Path)
    p.add_argument('--workers', type=int, default=8)
    a = p.parse_args()
    prepare(a.output.resolve(), a.previous.resolve(), a.workers)
