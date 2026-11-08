"""Projection equivalence, capacity matching and isolated evaluation barriers."""
from copy import deepcopy
import numpy as np
import pytest
import torch

from longtail import saw_projection as exp
from longtail.io import read_json, write_json, sha256
from longtail.models.temporal_heads import GlobalResidual
from longtail.metrics import probabilities


def old_state():
    torch.manual_seed(16)
    # StatisticsResidual uses the same hidden layout with two output rows.
    return {'hidden.weight': torch.randn(32, 128)*.03, 'hidden.bias': torch.randn(32)*.01,
            'output.weight': torch.randn(2, 32)*.1, 'output.bias': torch.randn(2)*.01}


def test_fold_and_wide_initialization_preserve_original_function_and_capacity():
    old = old_state()
    state = {'components': np.eye(128, 1792, dtype=np.float32),
             'center': np.random.default_rng(1).normal(size=1792).astype(np.float32)*.1}
    raw = torch.randn(48, 1792)
    pca = (raw-torch.from_numpy(state['center'])) @ torch.from_numpy(state['components']).T
    expected = torch.nn.functional.linear(torch.nn.functional.gelu(torch.nn.functional.linear(pca,
        old['hidden.weight'], old['hidden.bias'])), old['output.weight'][0:1], old['output.bias'][0:1])
    for method, count in [('pca_small', 4161), ('pca_wide', 57461), ('learned_projection', 57409)]:
        model = exp.initialize_head(method, old, state)
        assert sum(p.numel() for p in model.parameters()) == count
        x = raw if method == 'learned_projection' else pca
        torch.testing.assert_close(model(x), expected, rtol=1e-5, atol=2e-6)


def test_learned_projection_can_update_directions_discarded_by_PCA():
    state = {'components': np.eye(128, 1792, dtype=np.float32), 'center': np.zeros(1792, np.float32)}
    model = exp.initialize_head('learned_projection', old_state(), state)
    assert torch.count_nonzero(model.hidden.weight[:, 128:]) == 0
    x = torch.randn(32, 1792)
    model(x).square().mean().backward()
    assert model.hidden.weight.grad[:, 128:].abs().sum() > 0


@pytest.mark.parametrize('method', exp.METHODS)
def test_anchored_predictions_start_identically_and_preserve_other_five(method):
    state = {'components': np.eye(128, 1792, dtype=np.float32), 'center': np.zeros(1792, np.float32)}
    model = exp.initialize_head(method, old_state(), state)
    initial = deepcopy(model).eval().requires_grad_(False)
    x = torch.randn(32, exp.SHAPES[method][0])
    z = np.random.default_rng(7).normal(size=(32, 6)).astype(np.float32)
    with torch.no_grad():
        np.testing.assert_array_equal(exp.compose(z, (model(x)-initial(x)).numpy()), z)
    opt = torch.optim.AdamW(model.parameters(), lr=.00003)
    opt.zero_grad()
    (model(x)-initial(x)-1).square().mean().backward()
    opt.step()
    after = exp.compose(z, (model(x)-initial(x)).detach().numpy())
    np.testing.assert_array_equal(after[:, exp.NON_TARGETS], z[:, exp.NON_TARGETS])
    assert np.any(after[:, 2] != z[:, 2])


@pytest.mark.parametrize('role', ['arrival_ood', 'saw_period_ood', 'development_test', 'ood_wide'])
def test_all_reports_are_gated_before_cache_or_reference_access(tmp_path, role):
    write_json(tmp_path/'plan.json', {})
    folder = tmp_path/'cache/seed2050'
    folder.mkdir(parents=True)
    np.savez(folder/(role+'.npz'), features=np.zeros((1, 128)))
    with pytest.raises(FileNotFoundError):
        exp.cache(tmp_path, 2050, role)


def test_full_measurement_and_immutable_decision_barrier(tmp_path, monkeypatch):
    rng = np.random.default_rng(81)
    y = ((np.repeat(np.arange(1, 64), 8)[:, None] >> np.arange(6)) & 1).astype(np.uint8)
    weak = y.astype(bool) & (rng.random(y.shape) < .3)
    records = {'targets': y, 'weak_mask': weak, 'relative_power_db': np.where(y, np.where(weak, -12., 0.), np.nan),
        'reference_logits': np.where(y, 2., -2.)+rng.normal(size=y.shape),
        'identities': np.arange(len(y))+10000, 'indices': np.arange(len(y))}
    roles = []
    write_json(tmp_path/'plan.json', {'references': {'2050': {'new': {'thresholds': [.5]*6}}}})

    def cached(suite, seed, role):
        roles.append(role)
        return records

    def predict(suite, seed, methods, rec):
        return {'reference': rec['reference_logits'], **{method: exp.compose(rec['reference_logits'], np.full((len(y), 1), .1)) for method in methods}}

    monkeypatch.setattr(exp, 'cache', cached)
    monkeypatch.setattr(exp, 'predictions', predict)
    exp.measure(tmp_path, 2050, exp.METHODS)
    assert roles == ['calibration', 'check']
    output = tmp_path/'evaluation/seed2050'
    decisions = output/'decisions.json'
    bundle = read_json(decisions)
    for name in bundle['models']:
        for th in bundle['models'][name]['thresholds'].values():
            np.testing.assert_array_equal(np.asarray(th)[exp.NON_TARGETS], np.array([.5]*5))
    with pytest.raises(FileNotFoundError):
        exp.measure(tmp_path, 2050, exp.METHODS, report=True)
    assert roles == ['calibration', 'check']
    write_json(tmp_path/'frozen.json', {'plan_sha256': sha256(tmp_path/'plan.json'), 'files': {str(decisions): sha256(decisions)}})
    exp.measure(tmp_path, 2050, exp.METHODS, report=True)
    assert roles[2:] == exp.REPORT_ROLES
    for role in exp.REPORT_ROLES:
        assert (output/('metrics_'+role+'.json')).exists()
        with np.load(output/('predictions_'+role+'.npz')) as rec:
            for method in exp.METHODS:
                np.testing.assert_array_equal(rec['logits_'+method][:, exp.NON_TARGETS], rec['logits_reference'][:, exp.NON_TARGETS])
    write_json(decisions, {'tampered': True})
    with pytest.raises(AssertionError):
        exp.measure(tmp_path, 2050, exp.METHODS, report=True)
