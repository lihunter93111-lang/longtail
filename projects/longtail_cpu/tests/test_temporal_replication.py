import numpy as np
import pytest
import torch

from longtail.models.factory import build_model
from longtail.models.temporal_heads import build_residual, unpooled_features
from longtail.models.decision_heads import residual_logits
from longtail.replication_temporal import contrast


def test_exposing_unpooled_map_preserves_original_features_and_checkpoint_layout():
    torch.manual_seed(13)
    model = build_model({'entrypoint': 'longtail.models.dfms:build_model'}, num_classes=6).eval()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    x = torch.randn(3, 2, 3000)
    with torch.inference_mode():
        global_feature, temporal = unpooled_features(model, x)
        assert temporal.ndim == 3 and temporal.shape[:2] == (3, 128)
        torch.testing.assert_close(global_feature, model.forward_features(x), rtol=0, atol=0)
    assert before.keys() == model.state_dict().keys()
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())


@pytest.mark.parametrize('kind', ['global32', 'global33', 'temporal'])
def test_local_training_preserves_other_labels_and_frozen_evidence(kind):
    torch.manual_seed(22)
    model = build_residual(kind)
    feature = torch.randn(16, 128, 37) if kind == 'temporal' else torch.randn(16, 128)
    before = feature.clone()
    base = torch.randn(16, 6)
    assert torch.equal(residual_logits(base, model(feature)), base)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    target = torch.randint(0, 2, (16, 6)).float()
    for _ in range(3):
        optimizer.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(residual_logits(base, model(feature)), target)
        loss.backward()
        optimizer.step()
    z = residual_logits(base, model(feature))
    assert torch.equal(z[:, [0, 1, 3]], base[:, [0, 1, 3]])
    assert not torch.equal(z[:, [2, 4, 5]], base[:, [2, 4, 5]])
    assert torch.equal(feature, before)


def test_attention_weights_positions_but_adds_no_implicit_position_encoding():
    torch.manual_seed(9)
    model = build_residual('temporal').eval()
    with torch.no_grad():
        model.output_weight.normal_()
    x = torch.randn(4, 128, 25)
    # This readout sees contextual backbone positions, not their absolute index.
    torch.testing.assert_close(model(x), model(x[:, :, torch.randperm(25)]), atol=1e-6, rtol=1e-5)
    changed = x.clone()
    changed[:, :, 0] += 10
    assert not torch.allclose(model(x), model(changed))


def test_parameter_matched_global_control_and_conditional_interval():
    counts = {kind: sum(p.numel() for p in build_residual(kind).parameters()) for kind in ['global32', 'global33', 'temporal']}
    assert abs(counts['global33']/counts['temporal']-1) < .01
    d = contrast(np.array([.73, .72, .74, .75, .73]), np.array([.72, .71, .73, .74, .72]))
    assert d['positive_seeds'] == 5 and d['paired95'][0] > 0


def test_calibration_is_frozen_across_independent_report_measurement(tmp_path, monkeypatch):
    from longtail import replication_temporal as tr
    from longtail.io import read_json, sha256
    rng = np.random.default_rng(18)
    roles = {}
    for role, repeats in [('calibration', 4), ('check', 6), ('development_test', 8)]:
        y = np.tile(((np.arange(1, 64)[:, None] >> np.arange(6)) & 1), (repeats, 1)).astype(np.uint8)
        weak = y.astype(bool) & (rng.random(y.shape) < .3)
        roles[role] = {'targets': y, 'weak_mask': weak, 'relative_power_db': np.where(y, np.where(weak, -12., 0.), np.nan),
                       'z': np.where(y, 2., -2.)+rng.normal(0, 1.4, y.shape), 'identities': np.arange(len(y))}
    monkeypatch.setattr(tr, 'cache_records', lambda suite, stage, seed, tag, role, **kwargs: roles[role])
    monkeypatch.setattr(tr, 'predictions', lambda suite, stage, seed, tag, r: {'base': r['z'], 'global32': r['z']+.1, 'temporal': r['z']-.2})
    folder = tmp_path/'temporal/seed2050'
    folder.mkdir(parents=True)
    tr.measure(tmp_path, 'temporal', 2050, 'temporal')
    decisions = folder/'decisions_temporal.json'
    frozen = sha256(decisions)
    tr.measure(tmp_path, 'temporal', 2050, 'temporal', report=True)
    assert sha256(decisions) == frozen
    saved = read_json(decisions)
    assert all(th[0] == .5 for th in saved['thresholds'].values())
    report = read_json(folder/'metrics_temporal_development_test.json')
    assert report['metrics']['temporal']['samples'] == 504
    assert report['matched_calibration_fpr_metrics']['temporal']['samples'] == 504
