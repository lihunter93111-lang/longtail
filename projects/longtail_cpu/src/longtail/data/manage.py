"""Organize and verify working copies while preserving all original versions."""
import argparse
from pathlib import Path
import shutil
import h5py
import numpy as np
from .dataset import read_metadata
from .power_truth import read_power_truth
from ..io import read_json, write_json, sha256

CLASSES = ['FMB', 'AMA', 'SawFM', 'SinFM', 'MFT', 'ISRJ']
V3 = 'six_class_pure_label_longtail_v3_IF3_CF8_N8000_AMA'



def verify(root):
    root = Path(root).resolve()
    manifest = read_json(root / 'dataset.json')
    if manifest['classes'] != CLASSES:
        raise ValueError('Expected the AMA six-class contract')
    if set(manifest['files']) != {'train', 'development'}:
        raise ValueError('Expected train and development file records')
    joint = manifest.get('power_truth_schema') == 'per-label-v1'
    if not joint and sha256(root / 'ama_v3/source_audit.json') != manifest['source_audit_sha256']:
        raise ValueError('Source audit hash mismatch')
    identities = {}
    results = {}
    reference_labels = None
    for role, record in manifest['files'].items():
        relative = Path(record['path'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Unsafe dataset path')
        path = root / relative
        if path.stat().st_size != record['bytes'] or sha256(path) != record['sha256']:
            raise ValueError(f'Hash mismatch: {path}')
        y, seeds, length = read_metadata(path)
        if joint:
            if read_power_truth(path, y) is None:
                raise ValueError('Missing per-label power truth')
        counts = np.bincount(y.astype(np.int64) @ (1 << np.arange(6)), minlength=64)[1:]
        if y.shape != (record['samples'], 6) or length != manifest.get('signal_length', 3000) or (counts == 0).any():
            raise ValueError(f'Dataset contract mismatch: {role}')
        if y.sum(0).tolist() != record['label_counts'] or counts.tolist() != record['combination_counts']:
            raise ValueError(f'Distribution mismatch: {role}')
        with h5py.File(path, 'r') as f:
            dtype = f['iq_complex'].dtype
            if list(dtype.names or ()) != ['real', 'imag'] and not np.issubdtype(dtype, np.complexfloating):
                raise ValueError('Unexpected I/Q storage type')
        identities[role] = seeds
        if role == 'development':
            reference_labels = y
        results[role] = {'samples': len(y), 'sha256': record['sha256'], 'combinations': int((counts > 0).sum())}
    if np.intersect1d(identities['train'], identities['development']).size:
        raise ValueError('Dataset identity leakage')
    for name, record in manifest.get('evaluation_views', {}).items():
        relative = Path(record['path'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Unsafe view path')
        path = root / relative
        if path.stat().st_size != record['bytes'] or sha256(path) != record['sha256']:
            raise ValueError('View hash mismatch')
        y, ids, length = read_metadata(path)
        if length != manifest.get('signal_length', 3000) or not np.array_equal(y, reference_labels) or not np.array_equal(ids, identities['development']):
            raise ValueError('View row labels or identities differ')
        if read_power_truth(path, y) is None:
            raise ValueError('Missing view power truth')
    return {'verified': True, 'files': results, 'evaluation_views': len(manifest.get('evaluation_views', {})),
            'train_development_identity_overlap': 0}


def organize(source_root, output_root):
    source_root, output_root = Path(source_root).resolve(), Path(output_root).resolve()
    if (output_root / 'dataset.json').exists():
        verify(output_root)
        return read_json(output_root / 'dataset.json')
    original = source_root / V3
    audit = read_json(original / 'audit_summary.json')
    if audit['selected_types'] != CLASSES:
        raise ValueError('v3 source audit label contract mismatch')
    destination = output_root / 'ama_v3'
    destination.mkdir(parents=True, exist_ok=True)
    records = {}
    for role, filename, target in [('train', 'closedset_6class_train.mat', 'train.mat'),
                                   ('development', 'closedset_6class_test.mat', 'development.mat')]:
        source = original / filename
        expected = audit['files'][filename]['sha256']
        if sha256(source) != expected:
            raise ValueError('Source v3 integrity check failed')
        path = destination / target
        if path.exists():
            if sha256(path) != expected:
                raise FileExistsError(f'Refusing to replace existing data: {path}')
        else:
            temporary = path.with_suffix('.partial')
            shutil.copyfile(source, temporary)
            if sha256(temporary) != expected:
                raise ValueError('Copy integrity check failed')
            temporary.replace(path)
        y, _, _ = read_metadata(path)
        counts = np.bincount(y.astype(np.int64) @ (1 << np.arange(6)), minlength=64)[1:]
        records[role] = {'path': f'ama_v3/{target}', 'sha256': expected, 'bytes': path.stat().st_size,
                         'samples': len(y), 'label_counts': y.sum(0).astype(int).tolist(),
                         'combination_counts': counts.tolist(), 'original_file': f'{V3}/{filename}'}
    shutil.copyfile(original / 'audit_summary.json', destination / 'source_audit.json')
    manifest = {'schema_version': 1, 'dataset_id': 'ama-sixclass-v3', 'classes': CLASSES,
                'signal_length': 3000, 'files': records,
                'development_role': 'Consumed development pool, not an untouched locked test',
                'source': 'https://modelscope.cn/datasets/fulanya/longtail',
                'source_audit_sha256': sha256(destination / 'source_audit.json')}
    write_json(output_root / 'dataset.json', manifest)
    verify(output_root)
    return manifest


def backup_unused(root, inventory):
    """Move the original multi-version snapshot aside; never delete data."""
    root = Path(root).resolve()
    verify(root)
    snapshot = read_json(inventory)
    backup = root / 'backup'
    if backup.is_symlink():
        raise ValueError('Backup directory must not be a symlink')
    names = ['closedset_6class', 'power_sweep_9level_v1',
             'six_class_pure_label_longtail_v1_IF3_CF8_N8000',
             'six_class_pure_label_longtail_v2_IF3_CF8_N8000_AMA', V3,
             'README.md', '.gitattributes']
    # Validate the complete plan before moving any directory.
    for name in names:
        source, target = root / name, backup / name
        if source.exists() and target.exists():
            raise FileExistsError(f'Both source and backup exist: {name}')
        if source.is_symlink():
            raise ValueError(f'Refusing to move unexpected symlink: {source}')
    for record in snapshot['files']:
        relative = Path(record['path'])
        if relative.is_absolute() or '..' in relative.parts or relative.parts[0] not in names:
            raise ValueError('Unsafe snapshot path')
        source = root / relative
        path = source if source.exists() else backup / relative
        if not path.is_file() or path.stat().st_size != record['bytes']:
            raise ValueError(f'Original snapshot is incomplete: {relative}')
    backup.mkdir(exist_ok=True)
    moved = []
    for name in names:
        source, target = root / name, backup / name
        if source.exists():
            before = source.stat()
            source.rename(target)
            after = target.stat()
            if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
                raise RuntimeError('Move did not preserve the original filesystem object')
            moved.append(name)
    for record in snapshot['files']:
        path = backup / record['path']
        if not path.is_file() or path.stat().st_size != record['bytes']:
            raise ValueError(f'Backup verification failed: {path}')
    saved_inventory = backup / 'source_snapshot.json'
    if not saved_inventory.exists():
        write_json(saved_inventory, snapshot)
    result = {'moved_paths': moved, 'backup_directory': str(backup),
              'original_files_preserved': len(snapshot['files']),
              'original_bytes_preserved': sum(r['bytes'] for r in snapshot['files']),
              'deletion_performed': False, 'active_dataset': verify(root)}
    receipt = root / 'backup.json'
    if moved or not receipt.exists():
        write_json(receipt, result)
    return read_json(receipt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prepare = sub.add_parser('prepare')
    prepare.add_argument('--source-root', type=Path, required=True)
    prepare.add_argument('--output-root', type=Path, required=True)
    for name in ['verify']:
        item = sub.add_parser(name)
        item.add_argument('--root', type=Path, required=True)
    archive = sub.add_parser('backup-unused')
    archive.add_argument('--root', type=Path, required=True)
    archive.add_argument('--inventory', type=Path, required=True)
    joint = sub.add_parser('prepare-joint')
    joint.add_argument('--raw-root', type=Path, required=True)
    joint.add_argument('--output-root', type=Path, required=True)
    joint.add_argument('--profile', choices=['label_if3'], default='label_if3')
    joint.add_argument('--train-gaps', type=float, nargs='+', default=[0, 6, 12, 18, 24])
    args = parser.parse_args()
    if args.command == 'prepare':
        result = organize(args.source_root, args.output_root)
    elif args.command == 'prepare-joint':
        from .joint import prepare_joint
        result = prepare_joint(args.raw_root, args.output_root, args.profile, args.train_gaps)
    elif args.command == 'backup-unused':
        result = backup_unused(args.root, args.inventory)
    else:
        result = verify(args.root)
    print(result)


if __name__ == '__main__':
    main()
