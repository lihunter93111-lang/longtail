"""Build 32k paired training views and a new, deferred reporting identity pool."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import h5py
import numpy as np

from prepare_combo_report_pool import prepare as prepare_report
from longtail.data.dataset import read_metadata
from longtail.data.manage import verify
from longtail.data.signals import substream_seed
from longtail.data.splits import codes
from longtail.data.weak_validation import initialize, chunk_worker, TARGET_LABELS
from longtail.io import read_json, write_json, sha256, fingerprint


def link(source, target):
    try:
        os.link(source, target)
    except OSError:
        shutil.copyfile(source, target)


def prepare(previous_suite, output, workers=8):
    if output.exists():
        raise FileExistsError('Use a new data directory')
    output.mkdir(parents=True)
    evaluation = output / 'evaluation'
    prepare_report(previous_suite, evaluation, base_seed=3600000000, workers=workers,
                   dataset_id='ama-sixclass-v3-weak-confirmation-20261103')
    evaluation_manifest = read_json(evaluation / 'dataset.json')
    reference_plan = read_json(evaluation / 'generation_plan.json')
    original_manifest_path = Path(reference_plan['source_dataset_manifest'])
    original_manifest = read_json(original_manifest_path)
    original_train = original_manifest_path.parent / original_manifest['files']['train']['path']
    labels, retained_ids, _ = read_metadata(original_train)
    old_codes = codes(labels)
    with h5py.File(original_train) as f:
        old_seeds = f['sample_seed'][:].ravel()
    row_codes, row_seeds, targets, retained_positions = [], [], [], []
    next_new_seed = 3700000001
    for code in range(1, 64):
        old = np.flatnonzero(old_codes == code)
        active_targets = [c for c in TARGET_LABELS if (code >> c) & 1]
        eligible = code.bit_count() > 1 and active_targets
        seeds = np.r_[old_seeds[old], np.arange(next_new_seed, next_new_seed + 3 * len(old), dtype=np.uint64)]
        next_new_seed += 3 * len(old)
        retained_positions.extend(range(len(row_codes), len(row_codes) + len(old)))
        row_codes.extend([code] * len(seeds)); row_seeds.extend(seeds.tolist())
        targets.extend([active_targets[(i // 2) % len(active_targets)] if eligible and i % 2 == 0 else -1 for i in range(len(seeds))])
    row_codes, row_seeds, targets = np.array(row_codes), np.array(row_seeds, dtype=np.uint64), np.array(targets)
    assert len(row_codes) == 32000 and len(retained_positions) == 8000
    new_seeds = row_seeds[~np.isin(np.arange(32000), retained_positions)]
    new_ids = np.array([substream_seed(int(s), 2) for s in new_seeds])
    checked = sorted(set(reference_plan['checked_identity_files'] + [str(evaluation / 'development.h5')]))
    for path in checked:
        with h5py.File(path) as f:
            assert not np.intersect1d(new_ids, f['receiver_noise_seed'][:].ravel()).size, path
    generation = {'schema': 'paired-weak-fit-validation-v1', 'source_manifest': str(original_manifest_path),
        'source_manifest_sha256': sha256(original_manifest_path), 'evaluation_manifest': str(evaluation / 'dataset.json'),
        'physics': reference_plan['physics'], 'classes': evaluation_manifest['classes'],
        'retained_training_identities': 8000, 'additional_training_identities': 24000, 'training_pool_size': 32000,
        'new_training_base_seed': 3700000001, 'new_training_identity_files_checked': checked,
        'new_training_identity_overlap_existing': 0, 'combo_counts_multiplier': 4,
        'paired_views': 'B and C have identical component/noise seeds and labels; only targeted JNR differs.',
        'hard_targets': ['SawFM', 'MFT', 'ISRJ'], 'hard_rule': 'Exactly half of each eligible multi-label combination; rotate active target labels; conditional Uniform[10,30] with target <= max-10dB.',
        'hard_rows': int((targets >= 0).sum()), 'target_row_counts': {evaluation_manifest['classes'][c]: int((targets == c).sum()) for c in TARGET_LABELS},
        'D_data': 'Exactly the same manifest and files as C; loss is the only change.',
        'matlab_bitwise_parity': False, 'generator_sha256': sha256(Path(__file__)),
        'physics_sha256': sha256(Path(__file__).resolve().parents[1] / 'src/longtail/data/weak_validation.py')}
    generation['sha256'] = fingerprint(generation)
    write_json(output / 'generation_plan.json', generation)
    directories = [output / 'B_diverse', output / 'C_weak']
    for d in directories:
        d.mkdir(); link(evaluation / 'development.h5', d / 'development.h5')
    tasks = [(row_codes[i:i+64], row_seeds[i:i+64], targets[i:i+64], evaluation_manifest['classes']) for i in range(0,32000,64)]
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn'), initializer=initialize,
                             initargs=(generation['physics'],)) as pool, \
         h5py.File(directories[0] / 'train.partial.h5', 'x') as natural, \
         h5py.File(directories[1] / 'train.partial.h5', 'x') as hard:
        for f in [natural, hard]:
            f.attrs.update(power_truth_schema='per-label-v1', weak_gap_db=10., noise_power_db=0.,
                           generation_plan_sha256=generation['sha256'], generator='v3-paired-power-python-pcg64-v1')
        for chunk, views in enumerate(pool.map(chunk_worker, tasks)):
            start, stop = chunk * 64, min((chunk+1)*64,32000)
            for f, block in zip([natural, hard], views):
                for name, values in block.items():
                    if name not in f:
                        f.create_dataset(name, shape=(values.shape[0],32000), dtype=values.dtype,
                                         chunks=(values.shape[0],64), compression='lzf')
                    f[name][:,start:stop] = values
            if chunk % 32 == 0 or stop == 32000:
                print({'paired_training_rows': stop, 'total': 32000}, flush=True)
    records, audits = {}, {}
    for variant, d in zip(['B_diverse','C_weak'], directories):
        (d / 'train.partial.h5').replace(d / 'train.h5')
        y, ids, length = read_metadata(d / 'train.h5')
        with h5py.File(d / 'train.h5') as f:
            weak = f['weak_component_mask'][:].T.astype(bool)
            jnr = f['component_effective_JNR_dB'][:].T
            hard_target = f['hard_target_label'][:].ravel()
        np.testing.assert_array_equal(y.sum(0), labels.sum(0) * 4)
        assert len(np.unique(ids)) == 32000 and length == 3000
        active = np.isfinite(jnr); assert ((jnr[active] >= 10) & (jnr[active] <= 30)).all()
        if variant == 'C_weak':
            rows = np.flatnonzero(hard_target >= 0)
            assert weak[rows, hard_target[rows]].all()
        with h5py.File(evaluation / 'development.h5') as f:
            assert not np.intersect1d(ids, f['receiver_noise_seed'][:].ravel()).size
        record = {'path': 'train.h5', 'samples':32000, 'bytes':(d/'train.h5').stat().st_size,
                  'sha256':sha256(d/'train.h5'), 'label_counts':y.sum(0).tolist(),
                  'combination_counts':np.bincount(codes(y),minlength=64)[1:].tolist()}
        manifest = deepcopy(evaluation_manifest)
        manifest.update(dataset_id='ama-sixclass-weak-validation-'+variant+'-20261103',
                        generation_plan_sha256=generation['sha256'], training_role=variant)
        manifest['development_power_distribution'] = 'Independent Uniform[10,30] JNR per component'
        manifest['training_power_distribution'] = ('Independent Uniform[10,30] JNR per component' if variant == 'B_diverse'
            else 'Conditional weak-target augmentation for half of eligible rows, all JNR within [10,30]')
        if variant == 'C_weak':
            manifest.pop('independent_component_jnr_db', None)
            manifest['component_jnr_db_range'] = [10,30]
        manifest['files']['train'] = record
        write_json(d/'dataset.json',manifest); verification = verify(d)
        audits[variant] = {'verification':verification, 'weak_positive_count':weak.sum(0).tolist(),
                          'weak_positive_fraction':float(weak.sum()/y.sum()), 'hard_rows':int((hard_target>=0).sum())}
        records[variant] = str(d/'dataset.json')
    with h5py.File(original_train) as old, h5py.File(directories[0]/'train.h5') as new:
        for i in range(0,8000,64):
            original_indices = np.arange(i,min(i+64,8000))
            new_indices = retained_positions[i:i+64]
            for name in old:
                np.testing.assert_array_equal(old[name][:,original_indices],new[name][:,new_indices])
    with h5py.File(directories[0]/'train.h5') as b, h5py.File(directories[1]/'train.h5') as c:
        for name in ['sample_seed','receiver_noise_seed','component_seeds','label_selected']:
            np.testing.assert_array_equal(b[name][:],c[name][:])
    write_json(output/'generation_complete.json', {'status':'completed','manifests':records,
        'evaluation_manifest':str(evaluation/'dataset.json'),'audits':audits,
        'retained_8000_all_arrays_verified':True,'B_C_latents_and_noise_seeds_identical':True,
        'new_training_overlap_existing':0,'training_development_overlap':0,
        'report_identity_audit':read_json(evaluation/'generation_complete.json')})
    print(json.dumps({'status':'completed','root':str(output),'audits':audits},ensure_ascii=False),flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--previous-suite',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path);p.add_argument('--workers',type=int,default=8)
    a=p.parse_args();prepare(a.previous_suite.resolve(),a.output.resolve(),a.workers)
