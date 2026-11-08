import copy
import h5py
import numpy as np
import pytest
from longtail.data.dataset import IQDataset
from longtail.data.preprocess import transform
from longtail.data.splits import build_split, validate_split
from longtail.data.loaders import make_loader


def test_preprocessing_is_parameterized_and_does_not_mutate(experiment):
    _, config, _ = experiment
    p = config['dataloader']['preprocess']
    raw = np.array([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=float)
    copy_raw = raw.copy()
    x = transform(raw, p)
    assert np.mean(x[0] ** 2 + x[1] ** 2) == pytest.approx(1.0)
    assert np.array_equal(raw, copy_raw)
    other = dict(p, normalization='none', crop_start=1, crop_length=3, stride=2, scale=2)
    np.testing.assert_array_equal(transform(raw, other), raw[:, 1:4:2] * 2)
    with pytest.raises(ValueError, match='Crop'):
        transform(raw, dict(p, crop_length=100))


def test_lazy_loading_preserves_requested_order_and_workers(experiment):
    root, config, _ = experiment
    dataset = IQDataset(root / 'train.mat', [7, 2, 7, 0], config['dataloader']['preprocess'])
    loader_config = dict(config['dataloader'], workers=2)
    batches = list(make_loader(dataset, loader_config, batch_size=2))
    assert [i for batch in batches for i in batch['index'].tolist()] == [7, 2, 7, 0]
    assert np.allclose(batches[0]['signal'][0], batches[1]['signal'][0])
    assert batches[0]['signal'].shape == (2, 2, 64)


def test_splits_are_disjoint_complete_reproducible_and_order_independent(experiment):
    root, config, manifest = experiment
    split = build_split(manifest, root, config['data'])
    reordered = copy.deepcopy(config['data'])
    reordered['per_combination'] = dict(reversed(list(reordered['per_combination'].items())))
    assert split == build_split(manifest, root, reordered)
    roles = [split['roles'][r] for r in ('calibration', 'check', 'development_test')]
    assert set(sum(roles, [])) == set(range(189))
    assert sum(len(v) for v in roles) == 189
    validate_split(split, manifest)
    broken = copy.deepcopy(split)
    broken['roles']['check'] = broken['roles']['calibration']
    with pytest.raises(ValueError):
        validate_split(broken, manifest)


def test_leaking_identities_and_changed_data_are_rejected(experiment):
    root, config, manifest = experiment
    with h5py.File(root / 'development.mat', 'r+') as f:
        f['receiver_noise_seed'][0, 0] = 10000
    with pytest.raises(ValueError, match='overlap'):
        build_split(manifest, root, config['data'])


def test_memory_cache_and_lazy_loading_produce_the_same_preprocessed_signal(experiment):
    root, config, _ = experiment
    p = dict(config['dataloader']['preprocess'], remove_dc=True, crop_start=5, crop_length=48, stride=2)
    memory = IQDataset(root / 'train.mat', [11, 0, 4], p, cache='memory')
    lazy = IQDataset(root / 'train.mat', [11, 0, 4], p, cache='none')
    for i in range(3):
        np.testing.assert_array_equal(memory[i]['signal'], lazy[i]['signal'])
    assert memory[0]['signal'].shape == (2, 24)
    assert not memory._signals.flags.writeable
