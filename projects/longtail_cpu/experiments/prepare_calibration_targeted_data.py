"""Larger fresh calibration pools and gap-stratified, identity-safe one-bit pairs."""
import argparse
from copy import deepcopy
from pathlib import Path
import os

import h5py
import numpy as np

from longtail.data import v3_fresh, targeted_pairs
from longtail.data.signals import DEFAULT_PHYSICS
from longtail.data.dataset import read_metadata
from longtail.io import read_json, write_json, sha256, fingerprint
from prepare_staged_validation_data import save_blocks
from prepare_replication_temporal_data import distribution_worker, file_record

ROOT = Path(__file__).resolve().parents[1]


def prepare(output, source, workers=8):
    if output.exists():
        raise FileExistsError('Use a new data directory')
    original = read_json(source)
    classes = original['classes']
    physics = deepcopy(DEFAULT_PHYSICS)
    roles = {r: [] for r in ['fit', 'check', 'development_test']}
    edges = []
    cell = 0
    for target in [2, 4]:
        backgrounds = [c for c in range(1, 64) if not (c >> target) & 1]
        for j, background in enumerate(np.random.default_rng(6100+target).permutation(backgrounds)):
            role = 'fit' if j < 19 else 'check' if j < 25 else 'development_test'
            count = 24 if role == 'fit' else 16 if role == 'check' else 32
            edge = target*64+int(background)
            edges.append({'target': target, 'background_code': int(background), 'edge_id': edge, 'role': role})
            for gap_bin in range(3):
                roles[role].append((int(background), target, gap_bin, 4200000001+cell*targeted_pairs.MAX_ATTEMPTS, count, edge, classes))
                cell += 1
    output.mkdir(parents=True)
    plan = {'schema': 'large-calibration-targeted-pairs-v1', 'physics': physics, 'source_manifest': str(source),
            'source_manifest_sha256': sha256(source), 'classes': classes,
            'natural': {'calibration': {'first_seed': 4150000001, 'samples': 18144},
                        'targeted': {'first_seed': 4160000001, 'samples': 18144},
                        'ood_wide': {'first_seed': 4163000001, 'samples': 4032}},
            'per_combination': {'calibration': 128, 'check': 32, 'development_test': 128},
            'small_calibration_blocks': 'eight disjoint16-per-combination blocks; block0 is predeclared1008 control',
            'gap_bins': targeted_pairs.GAP_BINS, 'pair_edges': edges,
            'pair_sampling': 'Reject identities until original independent10--30dB target gap falls in bin; never rescale components.',
            'max_attempts_per_cell': targeted_pairs.MAX_ATTEMPTS,
            'generator_sha256': {str(p.relative_to(ROOT)): sha256(p) for p in [Path(__file__), ROOT/'src/longtail/data/targeted_pairs.py', ROOT/'src/longtail/data/one_bit.py', ROOT/'src/longtail/data/v3_fresh.py', ROOT/'src/longtail/data/signals.py']},
            'matlab_bitwise_parity': False, 'pair_holdout': '19/6/6 background edges; the frozen natural backbone has seen all63 combinations'}
    plan['sha256'] = fingerprint(plan)
    write_json(output/'generation_plan.json', plan)
    manifests, all_new_noise, shifts = {}, [], {}
    for tag, info in plan['natural'].items():
        folder = output/tag
        folder.mkdir()
        count = info['samples']
        codes = np.repeat(np.arange(1, 64), count//63)
        seeds = info['first_seed']+np.arange(count, dtype=np.uint64)
        tasks = [(codes[i:i+64], seeds[i:i+64], classes, (10., 40.) if tag == 'ood_wide' else (10., 30.)) for i in range(0, count, 64)]
        file = folder/'development.h5'
        save_blocks(file, tasks, v3_fresh.initialize, distribution_worker, physics, count, workers=workers)
        record = file_record(file)
        with h5py.File(file) as f:
            all_new_noise.extend(f['receiver_noise_seed'][:].ravel().tolist())
        if tag == 'ood_wide':
            shifts[tag] = {**record, 'path': str(file), 'component_jnr_db': [10., 40.]}
            continue
        os.link(source.parent/original['files']['train']['path'], folder/'train.h5')
        manifest = deepcopy(original)
        manifest.update(dataset_id='ama-v3-large-calibration-'+tag+'-20261104', generation_plan_sha256=plan['sha256'])
        manifest['files']['train']['path'] = 'train.h5'
        manifest['files']['development'] = record
        write_json(folder/'dataset.json', manifest)
        manifests[tag] = str(folder/'dataset.json')
    folder = output/'targeted_pairs'
    folder.mkdir()
    pair_files, audits = {}, {}
    for role, tasks in roles.items():
        total = 2*sum(task[4] for task in tasks)
        file = folder/(role+'.h5')
        save_blocks(file, tasks, targeted_pairs.initialize, targeted_pairs.conditional_worker, physics, total, paired=True, workers=workers)
        rec = file_record(file)
        with h5py.File(file) as f:
            y = f['label_selected'][:].T
            target = f['pair_target'][:].ravel().astype(int)
            bins = f['target_gap_bin'][:].ravel()
            gap = -f['relative_power_dB'][:].T[np.arange(total), target]
            assert np.all((y[::2] ^ y[1::2]).sum(1) == 1)
            assert np.all(y[np.arange(0,total,2), target[::2]] == 0)
            assert np.all(y[np.arange(1,total,2), target[1::2]] == 1)
            assert np.array_equal(f['receiver_noise_seed'][:, ::2], f['receiver_noise_seed'][:, 1::2])
            assert f['raw_difference_relative_error'][:].max() < 1e-12
            assert len(np.unique(f['sample_id'][:])) == total
            for b, (lo, hi) in enumerate(targeted_pairs.GAP_BINS):
                values = gap[1::2][bins[1::2] == b]
                assert np.all((values >= lo) & (values < hi))
            all_new_noise.extend(f['receiver_noise_seed'][0, ::2].tolist())
            audits[role] = {'pairs': total//2, 'one_bit_verified': True, 'gap_bins_verified': True,
                'target_gap_bin_counts': {f'{c}_{b}': int(np.sum((target[1::2] == c) & (bins[1::2] == b))) for c in [2,4] for b in range(3)},
                'max_rejection_attempts': int(f['rejection_attempts'][:].max())}
        pair_files[role] = {**rec, 'path': str(file), 'pairs': total//2}
    assert len(np.unique(all_new_noise)) == len(all_new_noise)
    checked = []
    for directory in [ROOT.parent/'datasets/longtail', ROOT/'data', ROOT/'backup']:
        for path in sorted(directory.rglob('*')):
            if path.suffix not in ['.h5', '.mat'] or output in path.parents or not h5py.is_hdf5(path):
                continue
            with h5py.File(path) as f:
                if 'receiver_noise_seed' not in f:
                    continue
                assert not np.intersect1d(all_new_noise, f['receiver_noise_seed'][:].ravel()).size, str(path)
            checked.append(str(path.resolve()))
    pair_manifest = {'files': pair_files, 'edges': edges, 'gap_bins': targeted_pairs.GAP_BINS,
                     'generation_plan_sha256': plan['sha256'], 'audits': audits}
    write_json(folder/'manifest.json', pair_manifest)
    write_json(output/'generation_complete.json', {'status': 'completed', 'manifests': manifests,
               'pair_manifest': str(folder/'manifest.json'), 'shifts': shifts,
               'new_background_count': len(all_new_noise), 'checked_identity_files': checked,
               'identity_overlap_existing': 0, 'new_backgrounds_mutually_disjoint': True, 'pair_audits': audits,
               'plan_sha256': plan['sha256']})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--workers', type=int, default=8)
    a = p.parse_args()
    prepare(a.output.resolve(), a.source.resolve(), a.workers)
