"""Preserve fit/calibration/check, replace only consumed reporting identities."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import hashlib
import multiprocessing as mp
import os
from pathlib import Path
import shutil

import h5py
import numpy as np

from longtail.data.dataset import read_metadata
from longtail.data.manage import verify
from longtail.data.signals import substream_seed
from longtail.data.splits import build_split, codes
from longtail.data.v3_fresh import initialize, chunk_worker
from longtail.io import read_json, write_json, sha256, fingerprint

ROOT = Path(__file__).resolve().parents[1]


def prepare(previous_suite, output, base_seed=3500000000, workers=8, dataset_id='ama-sixclass-v3-cpu-optimization-20261103'):
    previous = read_json(previous_suite / 'plan.json')
    source_run = Path(previous['jobs'][0]['output'])
    config = read_json(source_run / 'config.json')
    split = read_json(source_run / 'split.json')
    manifest_path = Path(config['data']['manifest'])
    source_root = manifest_path.parent
    manifest = read_json(manifest_path)
    source_generation = read_json(source_root / 'generation_plan.json')
    if output.exists():
        raise FileExistsError('Use a new report-pool directory')
    verify(source_root)
    report_indices = np.array(split['roles']['development_test'])
    labels, old_ids, _ = read_metadata(source_root / manifest['files']['development']['path'])
    report_codes = codes(labels[report_indices])
    seeds = base_seed + np.arange(1, len(report_indices) + 1, dtype=np.uint64)
    if seeds.max() >= 2 ** 32:
        raise ValueError('Seed overflow')
    new_ids = np.array([substream_seed(int(s), 2) for s in seeds])
    assert len(np.unique(new_ids)) == len(new_ids)
    checked = []
    candidates = set(Path(p) for p in source_generation.get('existing_identity_files_checked', source_generation.get('checked_identity_files', [])))
    for directory in [source_root.parent, ROOT / 'data', ROOT / 'backup']:
        candidates.update(p for p in directory.rglob('*') if p.suffix in ('.mat', '.h5'))
    for path in sorted(candidates):
        if not path.is_file() or not h5py.is_hdf5(path):
            continue
        with h5py.File(path, 'r') as f:
            if 'receiver_noise_seed' not in f:
                continue
            existing = np.asarray(f['receiver_noise_seed']).ravel().astype(np.uint64)
        if np.intersect1d(existing, new_ids).size:
            raise ValueError('New report identity overlap: ' + str(path))
        checked.append(str(path.resolve()))
    plan = {'schema': 'v3-replace-consumed-report-v1', 'base_seed': base_seed,
            'source_dataset_manifest': str(manifest_path), 'source_manifest_sha256': sha256(manifest_path),
            'previous_suite': str(previous_suite), 'source_split_sha256': split['sha256'],
            'preserved_roles': ['fit', 'calibration', 'check'], 'replaced_role': 'development_test',
            'new_report_samples': len(seeds), 'new_report_per_combination': 64,
            'report_indices': report_indices.tolist(), 'physics': source_generation['physics'],
            'rng': 'NumPy PCG64; original v3 physical equations, independent 10--30 dB JNR',
            'matlab_bitwise_parity': False, 'checked_identity_files': checked,
            'new_report_overlap_existing': 0, 'generator_sha256': sha256(Path(__file__)),
            'physics_port_sha256': sha256(ROOT / 'src/longtail/data/signals.py'),
            'composition_sha256': sha256(ROOT / 'src/longtail/data/v3_fresh.py')}
    plan['sha256'] = fingerprint(plan)
    output.mkdir(parents=True)
    write_json(output / 'generation_plan.json', plan)
    source_train = source_root / manifest['files']['train']['path']
    try:
        os.link(source_train, output / 'train.h5')
    except OSError:
        shutil.copyfile(source_train, output / 'train.h5')
    temporary = output / 'development.partial.h5'
    source_dev = source_root / manifest['files']['development']['path']
    shutil.copyfile(source_dev, temporary)
    tasks = [(report_codes[i:i + 64], seeds[i:i + 64], manifest['classes']) for i in range(0, len(seeds), 64)]
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn'),
                             initializer=initialize, initargs=(plan['physics'],)) as pool, h5py.File(temporary, 'r+') as f:
        for chunk, block in enumerate(pool.map(chunk_worker, tasks)):
            indices = report_indices[chunk * 64:(chunk + 1) * 64]
            for name, values in block.items():
                f[name][:, indices] = values
            if chunk % 8 == 0 or chunk == len(tasks) - 1:
                print({'new_report_generated': min((chunk + 1) * 64, len(seeds)), 'total': len(seeds)}, flush=True)
        f.attrs['generation_plan_sha256'] = plan['sha256']
        f.attrs['generator'] = 'v3-pcg64-preserve-development-replace-report-v1'
    temporary.replace(output / 'development.h5')
    preserved = {}
    with h5py.File(source_dev, 'r') as old, h5py.File(output / 'development.h5', 'r') as new:
        for role in ['calibration', 'check']:
            idx = np.array(split['roles'][role])
            preserved[role] = {}
            for name in old:
                old_digest = hashlib.sha256(np.asarray(old[name][:, idx]).tobytes()).hexdigest()
                new_digest = hashlib.sha256(np.asarray(new[name][:, idx]).tobytes()).hexdigest()
                assert old_digest == new_digest, (role, name)
                preserved[role][name] = new_digest
        jnr = np.asarray(new['component_effective_JNR_dB'][:, report_indices])
        active_jnr = jnr[np.isfinite(jnr)]
        assert (active_jnr >= 10).all() and (active_jnr <= 30).all()
        jnr_audit = {'minimum': float(active_jnr.min()), 'maximum': float(active_jnr.max()),
                     'mean': float(active_jnr.mean()), 'std': float(active_jnr.std()),
                     'active_components': len(active_jnr)}
    result = deepcopy(manifest)
    result['dataset_id'] = dataset_id
    result['generation_plan_sha256'] = plan['sha256']
    result['development_role'] = 'Existing calibration/check; 4032 fresh reporting identities deferred until all fixed jobs and decisions complete.'
    result['source_manifest_sha256'] = sha256(manifest_path)
    for role in ['train', 'development']:
        path = output / (role + '.h5')
        result['files'][role].update(path=path.name, sha256=sha256(path), bytes=path.stat().st_size)
    assert result['files']['train']['sha256'] == manifest['files']['train']['sha256']
    write_json(output / 'dataset.json', result)
    new_split = build_split(result, output, config['data'])
    assert new_split['roles'] == split['roles']
    _, ids, _ = read_metadata(output / 'development.h5')
    np.testing.assert_array_equal(ids[report_indices], new_ids)
    for role in ['calibration', 'check']:
        np.testing.assert_array_equal(ids[split['roles'][role]], old_ids[split['roles'][role]])
    verification = verify(output)
    write_json(output / 'generation_complete.json', {'status': 'completed', 'verification': verification,
        'preserved_role_array_sha256': preserved, 'new_report_identity_overlap_existing': 0,
        'new_report_identity_files_checked': len(checked), 'report_jnr_audit': jnr_audit,
        'new_split_sha256': new_split['sha256'], 'matlab_bitwise_parity': False})
    print({'status': 'completed', 'output': str(output), 'identity_files_checked': len(checked)}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous-suite', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--base-seed', type=int, default=3500000000)
    args = parser.parse_args()
    prepare(args.previous_suite.resolve(), args.output.resolve(), base_seed=args.base_seed, workers=args.workers)
