"""Independent identities for pair replication, temporal heads and fresh retraining."""
import argparse
from copy import deepcopy
from pathlib import Path
import os
import shutil

import h5py
import numpy as np

from longtail.data import one_bit, v3_fresh
from longtail.data.dataset import read_metadata
from longtail.data.power_truth import read_power_truth
from longtail.data.signals import DEFAULT_PHYSICS, substream_seed
from longtail.io import read_json, write_json, sha256, fingerprint
from prepare_staged_validation_data import save_blocks

ROOT = Path(__file__).resolve().parents[1]


def distribution_worker(task):
    codes, seeds, classes, jnr_range = task
    block = v3_fresh.chunk_worker((codes, seeds, classes))
    if tuple(jnr_range) == (10., 30.):
        return block
    for i, (code, seed) in enumerate(zip(codes, seeds)):
        active = np.flatnonzero((int(code) >> np.arange(6)) & 1)
        sample = v3_fresh.compose_v3(v3_fresh._SIM, [classes[c] for c in active], int(seed), jnr_range)
        block['iq_complex'][:, i] = sample['iq']
        for key, field in [('component_effective_JNR_dB', 'jnr'), ('component_measured_power_dB', 'power'),
                           ('relative_power_dB', 'relative'), ('component_seeds', 'component_seeds')]:
            block[key][:, i] = np.nan
            block[key][active, i] = sample[field]
        block['weak_component_mask'][:, i] = 0
        block['weak_component_mask'][active, i] = (len(active) > 1) & (sample['relative'] <= -10 + 1e-6)
    return block


def file_record(path):
    y, _, _ = read_metadata(path)
    read_power_truth(path, y)
    return {'path': path.name, 'sha256': sha256(path), 'bytes': path.stat().st_size,
            'samples': len(y), 'label_counts': y.sum(0).astype(int).tolist(),
            'combination_counts': np.bincount(y.astype(int) @ (1 << np.arange(6)), minlength=64)[1:].tolist()}


def prepare(output, source_manifest, workers=8):
    if output.exists():
        raise FileExistsError('Use a fresh output directory')
    source = read_json(source_manifest)
    classes = source['classes']
    definitions = {}
    for tag, base in [('repA', 4001000000), ('repB', 4002000000), ('temporal', 4003000000), ('retrain', 4011000000)]:
        definitions[tag] = {'codes': np.repeat(np.arange(1, 64), 96),
                            'seeds': base + np.arange(1, 6049, dtype=np.uint64), 'jnr': (10., 30.)}
    for tag, base, jnr in [('ood_low', 4012000000, (0., 20.)), ('ood_wide', 4013000000, (10., 40.))]:
        definitions[tag] = {'codes': np.repeat(np.arange(1, 64), 64),
                            'seeds': base + np.arange(1, 4033, dtype=np.uint64), 'jnr': jnr}
    definitions['new_train'] = {'codes': np.repeat(np.arange(1, 64), source['files']['train']['combination_counts']),
                                'seeds': 4010000000 + np.arange(1, 32001, dtype=np.uint64), 'jnr': (10., 30.)}
    assert len(definitions['new_train']['codes']) == 32000
    pair_definitions = {}
    for tag, base in [('repA', 4101000000), ('repB', 4102000000), ('retrain', 4103000000)]:
        roles = {r: [] for r in ['fit', 'check', 'development_test']}
        edges = []
        seed = base + 1
        for target in [0, 2, 4, 5]:
            backgrounds = np.array([c for c in range(1, 64) if not (c >> target) & 1])
            for j, background in enumerate(np.random.default_rng(6100 + target).permutation(backgrounds)):
                role = 'fit' if j < 19 else 'check' if j < 25 else 'development_test'
                count = 64 if role == 'development_test' else 32
                edge = target * 64 + int(background)
                edges.append({'target': target, 'background_code': int(background), 'edge_id': edge, 'role': role, 'pairs': count})
                for _ in range(count):
                    roles[role].append((int(background), target, seed, edge))
                    seed += 1
        pair_definitions[tag] = (roles, edges)
    new_seeds = [d['seeds'] for d in definitions.values()]
    new_seeds += [np.array([r[2] for rows in roles.values() for r in rows], dtype=np.uint64)
                  for roles, _ in pair_definitions.values()]
    all_seeds = np.concatenate(new_seeds)
    assert len(np.unique(all_seeds)) == len(all_seeds) and all_seeds.max() < 2**32
    noise = np.array([substream_seed(int(s), 2) for s in all_seeds])
    assert len(np.unique(noise)) == len(noise)
    checked = []
    for directory in [ROOT.parent / 'datasets/longtail', ROOT / 'data', ROOT / 'backup']:
        for path in sorted(directory.rglob('*')):
            if path.suffix not in ['.h5', '.mat'] or not h5py.is_hdf5(path):
                continue
            with h5py.File(path) as f:
                if 'receiver_noise_seed' not in f:
                    continue
                assert not np.intersect1d(noise, f['receiver_noise_seed'][:].ravel()).size, str(path)
            checked.append(str(path.resolve()))
    output.mkdir(parents=True)
    plan = {'schema': 'replication-temporal-data-v1', 'classes': classes, 'physics': deepcopy(DEFAULT_PHYSICS),
            'source_manifest': str(source_manifest), 'source_manifest_sha256': sha256(source_manifest),
            'source_code_sha256': {str(p.relative_to(ROOT)): sha256(p) for p in [Path(__file__), ROOT/'src/longtail/data/v3_fresh.py', ROOT/'src/longtail/data/one_bit.py', ROOT/'src/longtail/data/signals.py']},
            'definitions': {k: {'count': len(d['seeds']), 'first_seed': int(d['seeds'][0]), 'last_seed': int(d['seeds'][-1]), 'jnr': d['jnr']} for k, d in definitions.items()},
            'pair_first_seeds': {'repA': 4101000001, 'repB': 4102000001, 'retrain': 4103000001},
            'identity_overlap_existing': 0, 'checked_identity_files': sorted(set(checked)),
            'new_backgrounds': len(all_seeds), 'matlab_bitwise_parity': False,
            'limitations': 'Same Python physical simulator; OOD shifts component power only. All63 combinations occur in natural fit. Pair edges fixed across replications; identities change.'}
    plan['sha256'] = fingerprint(plan)
    write_json(output/'generation_plan.json', plan)
    manifests, pair_manifests, shifts = {}, {}, {}
    for tag, definition in definitions.items():
        folder = output/tag
        folder.mkdir()
        role = 'train' if tag == 'new_train' else 'development'
        file = folder/(role+'.h5')
        codes, seeds = definition['codes'], definition['seeds']
        tasks = [(codes[i:i+64], seeds[i:i+64], classes, definition['jnr']) for i in range(0, len(seeds), 64)]
        save_blocks(file, tasks, v3_fresh.initialize, distribution_worker, plan['physics'], len(seeds), workers=workers)
        record = file_record(file)
        if tag == 'new_train':
            new_train = file
            continue
        if tag.startswith('ood_'):
            shifts[tag] = {**record, 'path': str(file), 'component_jnr_db': definition['jnr']}
            continue
        # new_train is generated last in insertion order; defer retrain train link.
        if tag != 'retrain':
            train = source_manifest.parent/source['files']['train']['path']
            try:
                os.link(train, folder/'train.h5')
            except OSError:
                shutil.copyfile(train, folder/'train.h5')
        manifest = deepcopy(source)
        manifest.update(dataset_id='ama-v3-replication-'+tag+'-20261104', generation_plan_sha256=plan['sha256'])
        manifest['files']['development'] = record
        if tag != 'retrain':
            manifest['files']['train']['path'] = 'train.h5'
            write_json(folder/'dataset.json', manifest)
            manifests[tag] = str(folder/'dataset.json')
        else:
            retrain_manifest = manifest
    target = output/'retrain/train.h5'
    os.link(new_train, target)
    retrain_manifest['files']['train'] = file_record(target)
    write_json(output/'retrain/dataset.json', retrain_manifest)
    manifests['retrain'] = str(output/'retrain/dataset.json')
    for tag, (roles, edges) in pair_definitions.items():
        folder = output/('pairs_'+tag)
        folder.mkdir()
        files = {}
        for role, rows in roles.items():
            array = np.asarray(rows, np.int64)
            tasks = [(array[i:i+64, 0], array[i:i+64, 1], array[i:i+64, 2], array[i:i+64, 3], classes) for i in range(0, len(rows), 64)]
            file = folder/(role+'.h5')
            save_blocks(file, tasks, one_bit.initialize, one_bit.chunk_worker, plan['physics'], 2*len(rows), paired=True, workers=workers)
            rec = file_record(file)
            with h5py.File(file) as f:
                y = f['label_selected'][:].T
                assert np.all((y[::2] ^ y[1::2]).sum(1) == 1)
                assert np.array_equal(f['receiver_noise_seed'][:, ::2], f['receiver_noise_seed'][:, 1::2])
                assert f['raw_difference_relative_error'][:].max() < 1e-12
            files[role] = {**rec, 'path': str(file), 'pairs': len(rows)}
        manifest = {'files': files, 'edges': edges, 'generation_plan_sha256': plan['sha256']}
        write_json(folder/'manifest.json', manifest)
        pair_manifests[tag] = str(folder/'manifest.json')
    write_json(output/'generation_complete.json', {'status': 'completed', 'manifests': manifests, 'pair_manifests': pair_manifests,
               'shifts': shifts, 'plan_sha256': plan['sha256'], 'new_identity_overlap_existing': 0, 'checked_identity_files': len(set(checked))})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--workers', type=int, default=8)
    a = p.parse_args()
    prepare(a.output.resolve(), a.source.resolve(), a.workers)
