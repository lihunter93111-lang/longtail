import numpy as np
import pytest
from longtail.data.sampling import EpochWeightedSampler, EpochMixedComboSampler, sampler_epoch_seed
from longtail.decision import constrained_exact_thresholds
from longtail.metrics import probabilities, fit_thresholds


def labels():
    codes = np.repeat(np.arange(1, 64), np.arange(1, 64))
    return ((codes[:, None] >> np.arange(6)) & 1).astype(np.uint8)


def test_seed_derivation_does_not_reuse_adjacent_seed_shifted_epochs():
    seeds = [sampler_epoch_seed(s, e, stream) for s in range(2050, 2055)
             for e in range(1, 25) for stream in range(3)]
    assert len(set(seeds)) == len(seeds)
    for mode in ['label_aware', 'combo_aware']:
        a, b = EpochWeightedSampler(labels(), mode, 2050), EpochWeightedSampler(labels(), mode, 2051)
        a.set_epoch(2); b.set_epoch(1)
        assert list(a) != list(b)
        first = list(a); a.set_epoch(2)
        assert first == list(a)


def test_mixed_sampler_preserves_budget_natural_coverage_and_expected_marginals():
    y = labels()
    sampler = EpochMixedComboSampler(y, 2050)
    observed = []
    for e in range(40):
        sampler.set_epoch(e);draw = list(sampler);record = sampler.audit()
        assert len(draw) == len(y)
        assert len(np.unique(sampler.last_natural_indices)) == len(y) - len(y) // 2
        assert record['unique_fit_identities'] >= len(y) - len(y) // 2
        assert record['natural_draws_without_replacement'] + record['combo_draws_with_replacement'] == len(y)
        observed.append(y[draw].mean(0))
    np.testing.assert_allclose(np.mean(observed, axis=0), .5 * y.mean(0) + .5 * 32 / 63, atol=.008)
    sampler.set_epoch(8); first = list(sampler); sampler.set_epoch(8)
    assert first == list(sampler)


def test_exact_calibration_keeps_constraints_and_improves_its_calibration_objective():
    rng = np.random.default_rng(23)
    logits = rng.normal(size=(400, 6))
    y = (logits + rng.normal(size=(400, 6)) + np.array([.6, -.3, 0, 0, 0, 0]) > 0)
    thresholds, record = constrained_exact_thresholds(y, logits)
    assert record['calibration_exact_fitted'] >= record['calibration_exact_f1']
    assert np.all(np.array(record['fitted_recall']) >= np.array(record['reference_recall']) - .03 - 1e-12)
    assert np.all(np.array(record['fitted_fpr']) <= np.array(record['reference_fpr']) + .02 + 1e-12)
    for trace in record['search_exact_count_traces']:
        assert all(b >= a for a, b in zip(trace, trace[1:]))
    assert thresholds == constrained_exact_thresholds(y, logits)[0]
    pred = probabilities(logits) >= thresholds
    assert np.all(pred == y, axis=1).mean() == record['calibration_exact_fitted']


def test_exact_zero_tolerances_retain_f1_marginal_bounds():
    rng = np.random.default_rng(4)
    logits = rng.normal(size=(100, 2)); y = rng.integers(0, 2, size=(100, 2))
    thresholds, d = constrained_exact_thresholds(y, logits, recall_drop=0, fpr_rise=0)
    assert np.all(np.array(d['fitted_recall']) >= np.array(d['reference_recall']) - 1e-12)
    assert np.all(np.array(d['fitted_fpr']) <= np.array(d['reference_fpr']) + 1e-12)


@pytest.mark.parametrize('grid', [[0, .5], [.5, 1], [np.nan]])
def test_exact_rejects_invalid_grids(grid):
    with pytest.raises(ValueError):
        constrained_exact_thresholds(np.array([[0, 1], [1, 0]]), np.zeros((2, 2)), grid=grid)


def test_mixed_sampler_runs_with_training_contract(experiment):
    from longtail.train import run
    import json
    root, config, _ = experiment
    config['train']['sampler'] = 'mixed_combo'
    run(config)
    history = json.loads((root / 'run/history.json').read_text())
    assert history[0]['samples_seen'] == 252
    assert history[0]['sampling']['natural_draws_without_replacement'] == 126
