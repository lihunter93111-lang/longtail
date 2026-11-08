"""Plan and generate fresh six-class paired-power data with NumPy."""
import argparse
from pathlib import Path
import numpy as np
import h5py
from ..config import REPO_ROOT, merge
from ..io import read_json, write_json, fingerprint, sha256
from .manage import CLASSES
from .signals import DEFAULT_PHYSICS


def plan(config_path):
    config = read_json(config_path)
    required = {'base_seed', 'label_count_source', 'gaps_db', 'nominal_jnr_db', 'weak_gap_db', 'output_dir'}
    if not required.issubset(config) or set(config)-required-{'physics'} or not config['gaps_db'] or len(set(config['gaps_db'])) != len(config['gaps_db']) or min(config['gaps_db']) < 0 or config['weak_gap_db'] <= 0:
        raise ValueError('Invalid generation configuration')
    source = read_json(REPO_ROOT / config['label_count_source'])
    if source['classes'] != CLASSES:
        raise ValueError('AMA six-class count source required')
    counts = source['files']['train']['combination_counts']
    if len(counts) != 63 or min(counts) < 1 or max(counts) >= 10000:
        raise ValueError('All 63 combinations need 1..9999 identities')
    development = source['files']['development']['combination_counts']
    if len(development) != 63 or min(development) < 1 or max(development) >= 10000:
        raise ValueError('Invalid development counts')
    profiles = [{'name': 'label_if3', 'counts_train': counts}]
    labels = (np.arange(1, 64)[:, None] >> np.arange(6)) & 1
    summaries = {}
    for profile in profiles:
        vector = np.asarray(profile['counts_train'])
        marginal = vector @ labels
        summaries[profile['name']] = {'samples': int(vector.sum()), 'label_counts': marginal.tolist(),
            'label_if': float(marginal.max() / marginal.min()), 'combination_cf': float(vector.max() / vector.min())}
    intervals = []
    for role, offset in [('train', 0), ('development', 10000000)]:
        vector = counts if role == 'train' else development
        values = np.concatenate([config['base_seed'] + offset + code * 10000 + np.arange(1, n + 1) for code, n in enumerate(vector, 1)])
        if values.min() < 0 or values.max() > 2 ** 32 - 1:
            raise ValueError('Sample seeds exceed uint32')
        intervals.append((values + 2 * 2246822519) % (2 ** 32))
    if np.intersect1d(*intervals).size:
        raise ValueError('Generation roles overlap')
    # Check existing snapshots, including backup, without loading any waveform.
    existing_root = (REPO_ROOT / source.get('local_root', '../datasets/longtail')).resolve()
    checked = 0
    for path in (p for p in existing_root.rglob('*') if p.suffix in ('.mat', '.h5')):
        try:
            with h5py.File(path, 'r') as f:
                if 'receiver_noise_seed' not in f: continue
                old = np.asarray(f['receiver_noise_seed']).reshape(-1).astype(np.uint64)
        except OSError:
            continue
        checked += 1
        if any(np.intersect1d(old, values).size for values in intervals):
            raise ValueError(f'New identities overlap existing source: {path}')
    result = {**config, 'classes': CLASSES, 'profiles': profiles, 'counts_development': development,
              'distribution_summary': summaries,
              'count_source_sha256': sha256(REPO_ROOT / config['label_count_source']),
              'simulator_sha256': sha256(Path(__file__).with_name('signals.py')),
              'writer_sha256': sha256(Path(__file__).with_name('generate_files.py')),
              'generator': 'numpy-pcg64-v1', 'physics': merge(DEFAULT_PHYSICS, config.get('physics', {})),
              'output_dir': str((REPO_ROOT / config['output_dir']).resolve()),
              'schema': 'paired-power-generation-plan-v2', 'status': 'planned_not_generated',
              'identity_rule': 'base + role_offset(0/10000000) + combination_code*10000 + repetition',
              'existing_identity_files_checked': checked,
              'weak_definition': 'active and cardinality>1 and measured_power-max_active_power<=-weak_gap_db',
              'pairing': 'Same component seeds, noise seed and non-power parameters across gaps'}
    result['sha256'] = fingerprint(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=REPO_ROOT / 'generation/joint.json')
    parser.add_argument('--plan', type=Path, default=REPO_ROOT / 'data/joint_generation_plan.json')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    result = plan(args.config)
    write_json(args.plan, result)
    if args.execute:
        from .generate_files import generate
        generate(result)
    print({'status': result['status'] if not args.execute else 'raw_generated_requires_prepare_and_verify', 'plan': str(args.plan)})


if __name__ == '__main__':
    main()
