"""Synthetic contract checks; these fixtures are not physics experiment data."""
from copy import deepcopy
import h5py
import numpy as np
import pytest
import torch
from longtail.config import load_config, resolve_config
from longtail.data.dataset import read_metadata, IQDataset
from longtail.data.joint import prepare_joint
from longtail.data.manage import verify
from longtail.data.power_truth import read_power_truth
from longtail.engine.checkpoint import load_checkpoint
from longtail.evaluate import run as evaluate
from longtail.io import read_json, write_json, sha256
from longtail.models.factory import build_model
from longtail.train import run
from longtail.weak_metrics import weak_metrics


def add_truth(path, gap):
    y, _, _ = read_metadata(path)
    power = np.where(y, 30., np.nan)
    for row, active in enumerate(y):
        slots = np.flatnonzero(active)
        if len(slots) > 1:
            power[row, slots[row % len(slots)]] -= gap
    relative = power - np.nanmax(power, axis=1, keepdims=True)
    weak = y.astype(bool) & (relative <= -10 + 1e-6)
    with h5py.File(path, 'r+') as f:
        f.attrs['power_truth_schema'] = np.bytes_('per-label-v1')
        f.attrs['weak_gap_db'] = 10.
        for key, value in [('component_effective_JNR_dB', power), ('component_measured_power_dB', power),
                           ('relative_power_dB', relative), ('weak_component_mask', weak)]:
            f[key] = value.T
    return weak


@pytest.fixture
def joint_raw(experiment):
    root, config, _ = experiment
    import shutil
    raw = root / 'raw'
    for gap in (0, 12, 18, 24):
        directory = raw / 'label_if3' / f'gap_{gap}'
        directory.mkdir(parents=True)
        for role in ('train', 'development'):
            path = directory / f'{role}.mat'
            shutil.copyfile(root / f'{role}.mat', path)
            add_truth(path, gap)
    write_json(raw / 'generation_complete.json', {'status': 'completed', 'plan_sha256': 'synthetic-fixture'})
    return root, config, raw


def test_nominal_mean_is_never_weak_truth(experiment):
    root, config, _ = experiment
    path = root / 'train.mat'
    y, _, _ = read_metadata(path)
    with h5py.File(path, 'r+') as f:
        f['component_nominal_JNR_dB'] = np.full((1, len(y)), 30.)
    assert read_power_truth(path, y) is None
    dataset = IQDataset(path, [0], config['dataloader']['preprocess'])
    assert 'weak_mask' not in dataset[0]
    assert weak_metrics(y, np.zeros_like(y), list('abcdef'), [0.5]*6)['available'] is False


def test_power_truth_is_per_label_and_recomputed(experiment):
    root, config, _ = experiment
    path = root / 'train.mat'
    y, _, _ = read_metadata(path)
    expected = add_truth(path, 12)
    truth = read_power_truth(path, y)
    np.testing.assert_array_equal(truth['weak_mask'], expected)
    assert not expected[y.sum(1) == 1].any()
    dataset = IQDataset(path, [9], config['dataloader']['preprocess'])
    assert dataset[0]['weak_mask'].shape == (6,)
    dataset.close()
    with h5py.File(path, 'r+') as f:
        weak = f['weak_component_mask']
        weak[0, 0] = True
    with pytest.raises(ValueError, match='Weak mask'):
        read_power_truth(path, y)


def test_wrong_power_shapes_and_absent_values_are_rejected(experiment):
    root, _, _ = experiment
    path = root / 'train.mat'
    y, _, _ = read_metadata(path)
    add_truth(path, 0)
    with h5py.File(path, 'r+') as f:
        f['component_measured_power_dB'][1, 0] = 0  # label absent
    with pytest.raises(ValueError, match='absent'):
        read_power_truth(path, y)
    with h5py.File(path, 'r+') as f:
        del f['component_measured_power_dB']
        f['component_measured_power_dB'] = np.zeros((1, len(y)))
    with pytest.raises(ValueError, match='shape'):
        read_power_truth(path, y)


def test_weak_ranking_excludes_strong_positives():
    y = np.array([[1], [1], [0]])
    weak = np.array([[1], [0], [0]])
    z = np.array([[2.], [-100.], [0.]])
    metric = weak_metrics(y, z, ['A'], [0.5], weak, np.array([[-12.], [0.], [np.nan]]), [10])
    assert metric['weak_macro_aupr'] == 1
    assert metric['weak_macro_recall'] == 1
    assert metric['per_label']['A']['negative_fpr'] == 1
    assert metric['tail_weak_positive_count'] == 0
    assert metric['tail_labels'] == []  # No invented tail when all frequencies tie.
    assert sum(b['positive_count'] for b in metric['power_gap_bins']) == 2
    with pytest.raises(ValueError, match='subset'):
        weak_metrics(y, z, ['A'], [0.5], np.ones_like(y))


def test_compact_configs_replace_model_kwargs_and_derive_class_count(tmp_path):
    baseline = load_config('configs/baseline.json')
    assert 'kwargs' not in read_json('configs/baseline.json')['model']
    assert baseline['model']['kwargs']['width'] == 16
    assert baseline['model']['kwargs']['phase_lags'] == [1, 4, 16]
    assert baseline['dataloader']['preprocess']['normalization'] == 'complex_rms'
    write_json(tmp_path / 'parent.json', {**read_json('configs/baseline.json'),
        'model': {'entrypoint': 'longtail.models.dfms:build_model', 'kwargs': {'width': 4}}})
    write_json(tmp_path / 'child.json', {'extends': 'parent.json',
        'model': {'entrypoint': 'longtail.models.transformer:build_model'}})
    config = resolve_config(tmp_path / 'child.json')
    assert 'width' not in config['model']['kwargs']
    assert config['model']['kwargs']['dimension'] == 64
    model = build_model(config['model'], num_classes=6)
    assert not any(isinstance(module, torch.nn.Conv1d) for module in model.modules())
    logits = model(torch.randn(2, 2, 67))
    assert logits.shape == (2, 6)
    logits.sum().backward()
    assert all(p.grad is not None for p in model.parameters())
    with pytest.raises(ValueError, match='class count'):
        build_model({'entrypoint': config['model']['entrypoint'], 'kwargs': {'num_classes': 5}}, num_classes=6)


def test_joint_conversion_training_and_paired_frozen_evaluation(joint_raw):
    root, config, raw = joint_raw
    manifest = prepare_joint(raw, root / 'joint', train_gaps=(0, 12))
    assert len(manifest['evaluation_views']) == 4
    assert verify(root / 'joint')['verified']
    config = deepcopy(config)
    config['data']['manifest'] = str(root / 'joint/dataset.json')
    config['model'] = {'entrypoint': 'longtail.models.transformer:build_model',
                       'kwargs': {'dimension': 16, 'layers': 1, 'heads': 2, 'patch_size': 8}}
    config['train']['selection_metric'] = 'weak_macro_aupr'
    run(config)
    checkpoint = root / 'run/best.pt'
    payload, _ = load_checkpoint(checkpoint)
    assert payload['history'][0]['validation']['weak_component']['available']
    for gap in (0, 24):
        metrics = evaluate(checkpoint, view=f'gap_{gap}', device_name='cpu')
        report = read_json(root / f'run/evaluation/development_test/gap_{gap}/metrics.json')
        assert report['decision'] == payload['decision']
        assert metrics['calibrated']['weak_component']['available']
        assert (metrics['calibrated']['weak_component']['weak_positive_count'] == 0) == (gap == 0)
    a = np.load(root / 'run/evaluation/development_test/gap_0/predictions.npz')
    b = np.load(root / 'run/evaluation/development_test/gap_24/predictions.npz')
    np.testing.assert_array_equal(a['indices'], b['indices'])
    np.testing.assert_array_equal(a['identities'], b['identities'])
    np.testing.assert_array_equal(a['targets'], b['targets'])
    assert 'component_power_db' in b
    with pytest.raises(FileExistsError):
        prepare_joint(raw, root / 'joint', train_gaps=(0, 12))


def test_converter_rejects_misaligned_view(joint_raw):
    root, _, raw = joint_raw
    path = raw / 'label_if3/gap_24/development.mat'
    with h5py.File(path, 'r+') as f:
        f['receiver_noise_seed'][0, 0] += 99999
    with pytest.raises(ValueError, match='align'):
        prepare_joint(raw, root / 'bad', train_gaps=(0, 12))


def test_generation_plan_checks_counts_provenance_and_existing_identities(tmp_path, monkeypatch):
    from longtail.data import generation
    from longtail.data.manage import CLASSES
    monkeypatch.setattr(generation, 'REPO_ROOT', tmp_path)
    write_json(tmp_path / 'counts.json', {'classes': CLASSES, 'local_root': 'existing',
        'files': {'train': {'combination_counts': [4]*63}, 'development': {'combination_counts': [3]*63}}})
    config = {'base_seed': 3800000000, 'label_count_source': 'counts.json', 'gaps_db': [0, 12, 24],
              'nominal_jnr_db': 30, 'weak_gap_db': 10, 'output_dir': 'new-raw'}
    write_json(tmp_path / 'generation.json', config)
    result = generation.plan(tmp_path / 'generation.json')
    assert result['status'] == 'planned_not_generated'
    assert len(result['profiles']) == 1
    assert result['distribution_summary']['label_if3']['label_if'] == 1
    assert result['count_source_sha256'] == sha256(tmp_path / 'counts.json')
    assert not (tmp_path / 'new-raw').exists()
    (tmp_path / 'existing').mkdir()
    with h5py.File(tmp_path / 'existing/old.mat', 'w') as f:
        f['receiver_noise_seed'] = [(3800000000 + 10000 + 1 + 2*2246822519) % 2**32]
    with pytest.raises(ValueError, match='overlap existing'):
        generation.plan(tmp_path / 'generation.json')
