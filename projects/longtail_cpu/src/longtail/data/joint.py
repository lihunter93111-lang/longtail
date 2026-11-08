"""Build identity-unique training data and paired evaluation views from new raw files."""
from pathlib import Path
from contextlib import ExitStack
import shutil
import h5py
import numpy as np
from ..io import read_json, write_json, sha256
from .dataset import read_metadata
from .power_truth import read_power_truth
from .manage import CLASSES

FIELDS = ['iq_complex', 'label_selected', 'receiver_noise_seed', 'component_effective_JNR_dB',
          'component_measured_power_dB', 'relative_power_dB', 'weak_component_mask']


def _mix(paths, destination):
    metadata = [read_metadata(path) for path in paths]
    labels, ids, length = metadata[0]
    truths = [read_power_truth(p, m[0]) for p, m in zip(paths, metadata)]
    if any(t is None for t in truths):
        raise ValueError('New raw data must provide verified per-label power truth')
    if any(not np.array_equal(labels, m[0]) or not np.array_equal(ids, m[1]) or length != m[2] for m in metadata[1:]):
        raise ValueError('Power views must align by row, labels and identity')
    codes = labels.astype(np.int64) @ (1 << np.arange(6))
    selected = np.empty(len(labels), dtype=int)
    for code in np.unique(codes):
        positions = np.flatnonzero(codes == code)
        # A complete rotation of the attenuated role before switching gaps
        # avoids aliasing repetition%cardinality with repetition%num_views.
        cardinality = int(labels[positions[0]].sum())
        selected[positions] = (np.arange(len(positions)) // cardinality) % len(paths)
    temporary = destination.with_suffix('.partial')
    with h5py.File(temporary, 'w') as out:
        with h5py.File(paths[0], 'r') as first:
            for key in FIELDS:
                d = first[key]
                out.create_dataset(key, shape=d.shape, dtype=d.dtype,
                                   chunks=(d.shape[0], min(d.shape[1], 64)), compression='lzf')
            for key, value in first.attrs.items():
                out.attrs[key] = value
        with ExitStack() as stack:
            handles = [stack.enter_context(h5py.File(path, 'r')) for path in paths]
            for start in range(0, len(labels), 64):
                stop = min(start+64, len(labels))
                choices = selected[start:stop]
                for key in FIELDS:
                    block = np.empty((out[key].shape[0], stop-start), dtype=out[key].dtype)
                    for view, source in enumerate(handles):
                        mask = choices == view
                        if mask.any():
                            block[:, mask] = source[key][:, start:stop][:, mask]
                    out[key][:, start:stop] = block
    temporary.replace(destination)
    read_power_truth(destination, labels)
    return labels, ids


def _record(path, root):
    y, _, _ = read_metadata(path)
    codes = y.astype(np.int64) @ (1 << np.arange(6))
    return {'path': str(path.relative_to(root)), 'sha256': sha256(path), 'bytes': path.stat().st_size,
            'samples': len(y), 'label_counts': y.sum(0).astype(int).tolist(),
            'combination_counts': np.bincount(codes, minlength=64)[1:].tolist()}


def prepare_joint(raw_root, output_root, profile='label_if3', train_gaps=(0, 6, 12, 18, 24)):
    raw_root, output_root = Path(raw_root).resolve(), Path(output_root).resolve()
    receipt = read_json(raw_root / 'generation_complete.json')
    if receipt['status'] != 'completed':
        raise ValueError('Raw generation is incomplete')
    if profile != 'label_if3' or not train_gaps or len(set(train_gaps)) != len(train_gaps):
        raise ValueError('Invalid frequency profile or training gaps')
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError('Use an empty destination for a new experiment condition')
    output_root.mkdir(parents=True, exist_ok=True)
    sources = list((raw_root / profile).glob('gap_*/development.h5')) + list((raw_root / profile).glob('gap_*/development.mat'))
    sources.sort(key=lambda path: float(path.parent.name.removeprefix('gap_')))
    if not sources:
        raise ValueError('Missing development power views')
    files = {}
    role_ids, role_labels = {}, {}
    for role in ['train', 'development']:
        paths = list(sources) if role == 'development' else []
        if role == 'train':
            for gap in train_gaps:
                path = raw_root / profile / f'gap_{gap:g}' / f'{role}.h5'
                paths.append(path if path.exists() else path.with_suffix('.mat'))
        target = output_root / f'{role}.h5'
        role_labels[role], role_ids[role] = _mix(paths, target)
        files[role] = _record(target, output_root)
    if np.intersect1d(role_ids['train'], role_ids['development']).size:
        raise ValueError('Train/development identities overlap')
    expected_length = read_metadata(output_root / files['development']['path'])[2]
    views = {}
    for source in sources:
        y, identity, length = read_metadata(source)
        if length != expected_length or not np.array_equal(identity, role_ids['development']) or not np.array_equal(y, role_labels['development']):
            raise ValueError('Evaluation view labels or identities are misaligned')
        if read_power_truth(source, y) is None:
            raise ValueError('Evaluation views require verified power truth')
        target = output_root / 'views' / source.parent.name / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        views[source.parent.name] = _record(target, output_root)
    manifest = {'schema_version': 2, 'dataset_id': f'joint-ama-v1-{profile}-gaps-{list(train_gaps)}',
                'signal_length': read_metadata(output_root / files['train']['path'])[2],
                'classes': CLASSES, 'files': files, 'evaluation_views': views,
                'power_truth_schema': 'per-label-v1', 'raw_plan_sha256': receipt['plan_sha256'],
                'frequency_profile': profile, 'train_gaps_db': list(train_gaps),
                'development_gaps_db': [float(path.parent.name.removeprefix('gap_')) for path in sources],
                'identity_protocol': 'One view per train identity; shared role indices across evaluation views'}
    write_json(output_root / 'dataset.json', manifest)
    from .manage import verify
    verify(output_root)
    return manifest
