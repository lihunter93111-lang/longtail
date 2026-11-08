"""ISRJ-only update invariants and complete frozen-evaluation smoke test."""
import numpy as np
import pytest
import torch

from longtail import isrj_segment as experiment
from longtail.io import read_json, sha256, write_json
from longtail.metrics import probabilities


def records():
    rng = np.random.default_rng(85)
    y = ((np.repeat(np.arange(1, 64), 8)[:, None] >> np.arange(6)) & 1).astype(np.uint8)
    z = np.where(y, 2., -2.)+rng.normal(size=y.shape)
    weak = y.astype(bool) & (rng.random(y.shape) < .3)
    relative = np.where(y, np.where(weak, -12., 0.), np.nan)
    return {'targets': y, 'reference_logits': z, 'weak_mask': weak,
            'relative_power_db': relative, 'indices': np.arange(len(y)), 'identities': 10000+np.arange(len(y))}


def test_equal_capacity_zero_start_and_update_preserves_other_five():
    torch.manual_seed(7)
    model = experiment.ISRJResidual()
    assert sum(p.numel() for p in model.parameters()) == 4161
    x = torch.randn(32, 128)
    z = np.random.default_rng(8).normal(size=(32, 6)).astype(np.float32)
    np.testing.assert_array_equal(experiment.compose(z, model(x).detach().numpy()), z)
    opt = torch.optim.AdamW(model.parameters(), lr=.01)
    opt.zero_grad()
    (model(x)-1).square().mean().backward()
    opt.step()
    updated = experiment.compose(z, model(x).detach().numpy())
    np.testing.assert_array_equal(updated[:, :5], z[:, :5])
    assert np.any(updated[:, 5] != z[:, 5])
    with pytest.raises(ValueError):
        experiment.compose(z, np.zeros((32, 2)))


def test_auxiliary_calibration_changes_only_isrj_and_matches_reference_fp_budget():
    rec = records()
    anchor = rec['reference_logits']
    z = experiment.compose(anchor, np.full((len(anchor), 1), .9))
    locked = [.5, .4, .6, .45, .55, .5]
    bundle = experiment.fit_decisions(rec, {'reference': anchor, 'probe': z}, locked)
    for th in bundle['models']['probe']['thresholds'].values():
        np.testing.assert_array_equal(np.array(th)[:5], np.array(locked)[:5])
    cap = bundle['models']['probe']['thresholds']['matched_calibration_fpr']
    negative = ~rec['targets'][:, 5].astype(bool)
    assert (probabilities(z)[negative, 5] >= cap[5]).sum() <= (probabilities(anchor)[negative, 5] >= locked[5]).sum()
    assert bundle['primary'] == 'locked'


def test_development_and_frozen_report_measurement_end_to_end(tmp_path, monkeypatch):
    rec = records()
    roles = []
    write_json(tmp_path/'plan.json', {'references': {'2050': {'new': {'thresholds': [.5]*6}}}})

    def cached(suite, seed, role):
        roles.append(role)
        return rec

    def predict(suite, seed, methods, records):
        z = records['reference_logits']
        return {'reference': z, 'global128': experiment.compose(z, np.full((len(z), 1), .1))}

    monkeypatch.setattr(experiment, 'cache', cached)
    monkeypatch.setattr(experiment, 'predictions', predict)
    experiment.measure(tmp_path, 2050, ['global128'])
    assert roles == ['calibration', 'check']
    output = tmp_path/'evaluation/seed2050'
    assert (output/'metrics_check.json').exists()
    before = roles.copy()
    with pytest.raises(FileNotFoundError):
        experiment.measure(tmp_path, 2050, ['global128'], report=True)
    assert roles == before
    decision = output/'decisions.json'
    write_json(tmp_path/'frozen.json', {'plan_sha256': sha256(tmp_path/'plan.json'),
                                      'files': {str(decision): sha256(decision)}})
    experiment.measure(tmp_path, 2050, ['global128'], report=True)
    for role in ['development_test', 'ood_wide', 'shift0', 'shift1', 'shift3', 'shift7']:
        saved = read_json(output/('metrics_'+role+'.json'))['metrics']['global128']['locked']
        with np.load(output/('predictions_'+role+'.npz')) as pred:
            np.testing.assert_array_equal(pred['logits_global128'][:, :5], pred['logits_reference'][:, :5])
            measured = experiment.metric(rec, pred['logits_global128'], [.5]*6)
            assert measured['exact'] == saved['exact'] and measured['isrj_weak_ap'] == saved['isrj_weak_ap']
    write_json(decision, {'tampered': True})
    before = roles.copy()
    with pytest.raises(AssertionError):
        experiment.measure(tmp_path, 2050, ['global128'], report=True)
    assert roles == before


@pytest.mark.parametrize('role', ['development_test', 'ood_wide', 'shift7'])
def test_real_extractor_report_gate_precedes_any_cached_report_read(tmp_path, role):
    path = tmp_path/'cache/seed2050'
    path.mkdir(parents=True)
    np.savez(path/(role+'.npz'), features=np.zeros((1, 128)))
    write_json(tmp_path/'plan.json', {})
    with pytest.raises(FileNotFoundError):
        experiment.cache(tmp_path, 2050, role)


def test_guard_rejects_missed_isrj_and_excess_false_positives():
    ref = experiment.metric(records(), records()['reference_logits'], [.5]*6)
    worse = {**ref, 'isrj_fpr': ref['isrj_fpr']+.006,
             'isrj_weak_recall': ref['isrj_weak_recall']-.02}
    checks = experiment.guard(worse, ref)
    assert not checks['isrj_fpr'] and not checks['isrj_weak_recall']
