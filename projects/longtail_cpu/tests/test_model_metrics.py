import json
import numpy as np
import pytest
import torch
from longtail.models.factory import build_model
from longtail.metrics import evaluate_metrics, fit_thresholds


def test_baseline_parameter_count_and_gradient():
    model = build_model({'entrypoint': 'longtail.models.dfms:build_model', 'kwargs': {}})
    assert sum(p.numel() for p in model.parameters()) == 257511
    logits = model(torch.randn(2, 2, 64))
    assert logits.shape == (2, 6)
    logits.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_new_model_requires_only_a_module_file(tmp_path, monkeypatch):
    (tmp_path / 'example_model.py').write_text('from torch import nn\ndef build_model(num_classes):\n return nn.Sequential(nn.AdaptiveAvgPool1d(1),nn.Flatten(),nn.Linear(2,num_classes))\n')
    monkeypatch.syspath_prepend(str(tmp_path))
    model = build_model({'entrypoint': 'example_model:build_model', 'kwargs': {'num_classes': 6}})
    assert model(torch.randn(3, 2, 64)).shape == (3, 6)


def test_exact_and_macro_combination_accuracy_differ_and_missing_labels_are_finite():
    y = np.array([[1, 0], [1, 0], [1, 1]])
    z = np.array([[4, -4], [4, -4], [4, -4]])
    m = evaluate_metrics(y, z, ['A', 'B'])
    assert m['subset_accuracy'] == pytest.approx(2 / 3)
    assert m['macro_combination_accuracy'] == pytest.approx(1 / 2)
    empty = evaluate_metrics(np.zeros((2, 2)), np.zeros((2, 2)), ['A', 'B'])
    assert empty['macro_aupr'] is None
    json.dumps(empty, allow_nan=False)


def test_calibration_improves_a_known_low_probability_positive():
    y = np.array([[1], [1], [0], [0]])
    z = np.array([[-0.2], [-0.1], [-2.0], [-3.0]])
    t = fit_thresholds(y, z, [0.05, 0.95, 181])
    m = evaluate_metrics(y, z, ['A'], t)
    assert m['macro_f1'] == 1
    assert t[0] < 0.5


def test_historical_feature_initialization_keeps_new_head(tmp_path):
    from longtail.engine.checkpoint import initialize
    from longtail.io import sha256
    recipe = {'entrypoint': 'longtail.models.dfms:build_model', 'kwargs': {'width': 4}}
    torch.manual_seed(1);source = build_model(recipe)
    torch.manual_seed(2);target = build_model(recipe)
    head = target.classifier[1].weight.detach().clone()
    path = tmp_path / 'historical.pt'
    torch.save({'model_state_dict': {'backbone.' + k: v for k, v in source.state_dict().items()}}, path)
    init = {'checkpoint': str(path), 'sha256': sha256(path), 'prefix': 'backbone.', 'exclude': ['classifier.']}
    initialize(target, init)
    assert torch.equal(target.classifier[1].weight, head)
    assert torch.equal(target.complex_encoder.block1.conv1.real_conv.weight,
                       source.complex_encoder.block1.conv1.real_conv.weight)
    init['sha256'] = '0' * 64
    with pytest.raises(ValueError, match='SHA-256'):
        initialize(target, init)
