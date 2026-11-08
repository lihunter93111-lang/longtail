"""Controls that distinguish new evidence from training and calibration gains."""
import numpy as np
import pytest
import torch

from longtail.io import sha256, write_json
from longtail.models.components.temporal import GlobalStatisticsPooling1d
from longtail.segment_statistics import (
    StatisticsResidual, cache, compose, fit_decisions, fit_transform,
    guard, segment_statistics, transform,
)
from longtail.metrics import probabilities


def test_segment_statistics_retains_global_and_uses_all_unequal_bins():
    x = torch.arange(2*3*15, dtype=torch.float32).reshape(2, 3, 15)
    result = segment_statistics(x)
    assert result.shape == (2, 42)
    torch.testing.assert_close(result[:, :6], GlobalStatisticsPooling1d(3)(x), rtol=0, atol=0)
    # 15 cannot divide evenly by2 or4: no tail positions dropped or duplicated.
    left, right = torch.tensor_split(x, 2, dim=-1)
    torch.testing.assert_close(result[:, 6:9], left.mean(-1))
    torch.testing.assert_close(result[:, 12:15], right.mean(-1))
    torch.testing.assert_close((result[:, 6:9]*8+result[:, 12:15]*7)/15, x.mean(-1))
    assert torch.isfinite(segment_statistics(torch.ones(2, 3, 15))).all()


def test_full_dimension_transform_preserves_standardized_information_and_binding():
    rng = np.random.default_rng(32)
    fit = rng.normal(size=(200, 8)).astype(np.float32)
    identities = np.arange(200)+10000
    state = fit_transform(fit, identities, 3, dimensions=8)
    cv = rng.normal(size=(20, 8)).astype(np.float32)
    z = transform(cv, state)
    standardized = (cv-state['mean'])/state['scale']-state['center']
    np.testing.assert_allclose(z @ state['components'], standardized, rtol=2e-5, atol=3e-6)
    assert state['fit_role'] == 'natural fit only' and state['fit_samples'] == 200
    saved = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in state.items()}
    transform(cv+1000, state)
    for key in ['mean', 'scale', 'components']:
        np.testing.assert_array_equal(state[key], saved[key])
    assert fit_transform(fit, identities+1, 3, dimensions=8)['fit_id_sha256'] != state['fit_id_sha256']


def test_residual_initial_prediction_is_identical_and_four_columns_cannot_drift():
    torch.manual_seed(15)
    model = StatisticsResidual()
    assert sum(p.numel() for p in model.parameters()) == 4194
    x = torch.randn(32, 128)
    anchor = np.random.default_rng(16).normal(size=(32, 6)).astype(np.float32)
    np.testing.assert_array_equal(compose(anchor, model(x).detach().numpy()), anchor)
    opt = torch.optim.AdamW(model.parameters(), lr=.01)
    opt.zero_grad()
    (model(x)-1).square().mean().backward()
    opt.step()
    result = compose(anchor, model(x).detach().numpy())
    np.testing.assert_array_equal(result[:, [0, 1, 3, 5]], anchor[:, [0, 1, 3, 5]])
    assert np.any(result[:, [2, 4]] != anchor[:, [2, 4]])


def test_auxiliary_decisions_share_calibration_fp_counts_but_lock_other_thresholds():
    rng = np.random.default_rng(51)
    y = rng.integers(0, 2, (256, 6)).astype(np.uint8)
    z = rng.normal(size=y.shape)
    shifted = z.copy()
    shifted[:, [2, 4]] += .7
    locked = [.5, .4, .6, .35, .55, .5]
    decisions = fit_decisions({'targets': y}, {'anchor_pair1': z, 'probe': shifted}, locked)
    for rule, th in decisions['models']['probe']['thresholds'].items():
        np.testing.assert_array_equal(np.asarray(th)[[0, 1, 3, 5]], np.asarray(locked)[[0, 1, 3, 5]])
    cap = decisions['models']['probe']['thresholds']['matched_calibration_fpr']
    ref = probabilities(z) >= locked
    pred = probabilities(shifted) >= cap
    for c in [2, 4]:
        assert np.sum(pred[~y[:, c].astype(bool), c]) <= np.sum(ref[~y[:, c].astype(bool), c])
    assert decisions['primary'] == 'locked'


@pytest.mark.parametrize('role', ['development_test', 'ood_wide', 'pair_report', 'shift7'])
def test_report_gate_precedes_even_existing_cache_reads(tmp_path, role):
    folder = tmp_path/'cache/seed2050'
    folder.mkdir(parents=True)
    np.savez(folder/(role+'.npz'), features=np.zeros((1, 128)))
    write_json(tmp_path/'plan.json', {})
    with pytest.raises(FileNotFoundError):
        cache(tmp_path, 2050, role)
    bound = tmp_path/'selection.json'
    write_json(bound, {'selected': 'global128'})
    write_json(tmp_path/'frozen.json', {'plan_sha256': sha256(tmp_path/'plan.json'), 'files': {str(bound): sha256(bound)}})
    write_json(bound, {'selected': 'segments124'})
    with pytest.raises(AssertionError):
        cache(tmp_path, 2050, role)


def test_guard_rejects_recall_gained_only_by_excess_false_positives():
    labels = ['FMB', 'AMA', 'SawFM', 'SinFM', 'MFT', 'ISRJ']
    ref = {'exact': .75, 'saw_weak_recall': .5, 'mft_weak_recall': .7,
           'focus_weak_aupr': .8, 'tail_f1': .98,
           'per_label': {c: {'negative_fpr': .04} for c in labels}}
    row = {**ref, 'saw_weak_recall': .55,
           'per_label': {c: {'negative_fpr': .04} for c in labels}}
    row['per_label']['SawFM']['negative_fpr'] = .046
    assert not guard(row, ref)['class_fpr']
