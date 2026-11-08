"""Feature selection, gradient flow and saved-run contracts for route 1."""
from copy import deepcopy
import json
import numpy as np
import pytest
import torch
from longtail.config import resolve_config
from longtail.models.dfms import DFMSIQ
from longtail.models.dfms_query import DFMSQueryIQ
from longtail.train import run
from longtail.evaluate import run as evaluate
from longtail.engine.checkpoint import load_checkpoint


@pytest.mark.parametrize('levels', [(3,), (1, 2, 3)])
def test_query_uses_expected_complex_levels_and_backpropagates(levels):
    model = DFMSQueryIQ(width=4, query_dimension=16, query_heads=2,
                        query_feedforward=32, memory_levels=levels)
    values = torch.randn(2, 2, 67)
    output = model(values)
    assert output.shape == (2, 6)
    torch.nn.functional.binary_cross_entropy_with_logits(
        output, torch.randint(0, 2, output.shape).float()).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert model.label_queries.weight.grad.abs().sum() > 0
    assert all(p.weight.grad.abs().sum() > 0 for p in model.memory_projections)
    model.eval()
    with torch.no_grad():
        logits, diagnostic = model.forward_diagnostics(values)
        lengths = [67, 33, 16, 8]
        assert diagnostic['memory_lengths'] == [lengths[i] for i in levels]
        attention = diagnostic['attention']
        assert attention.shape == (2, 6, sum(lengths[i] for i in levels))
        torch.testing.assert_close(attention.sum(-1), torch.ones(2, 6))
        torch.testing.assert_close(logits, model(values), atol=2e-6, rtol=2e-5)
        expected = diagnostic['global_logits'] + diagnostic['query_scale'] * diagnostic['query_logits']
        torch.testing.assert_close(logits, expected)


def test_query_retains_identical_initialized_global_path_and_capacity_control():
    torch.manual_seed(2050); baseline = DFMSIQ()
    torch.manual_seed(2050); query = DFMSQueryIQ()
    for key, value in baseline.state_dict().items():
        torch.testing.assert_close(value, query.state_dict()[key], rtol=0, atol=0)
    baseline.eval();query.eval()
    with torch.no_grad():
        query.query_scale.zero_()
        values = torch.randn(2, 2, 65)
        torch.testing.assert_close(query(values), baseline(values), atol=1e-6, rtol=1e-5)
    multi = sum(p.numel() for p in query.parameters())
    capacity = sum(p.numel() for p in DFMSQueryIQ(memory_levels=(3,), query_feedforward=208).parameters())
    assert abs(multi - capacity) / multi < 0.001
    for config in ['query_route1_cpu', 'query_route1_single', 'query_route1_capacity', 'query_route1_multiscale']:
        resolved = resolve_config(f'configs/{config}.json')
        assert resolved['train']['device'] == 'cpu'
        assert resolved['train']['selection_metric'] == 'weak_macro_aupr'


def test_query_checkpoint_reload_and_frozen_threshold_evaluation(experiment):
    root, config, _ = experiment
    config = deepcopy(config)
    config['model'] = {'entrypoint': 'longtail.models.dfms_query:build_model',
        'kwargs': {'width': 4, 'query_dimension': 16, 'query_heads': 2,
                   'query_feedforward': 32, 'memory_levels': [1, 2, 3]}}
    run(config)
    payload, _ = load_checkpoint(root / 'run/best.pt')
    decision = deepcopy(payload['decision'])
    metrics = evaluate(root / 'run/best.pt', output=root / 'query_evaluation', device_name='cpu')
    assert metrics['calibrated']['samples'] == 63
    assert metrics['calibrated']['thresholds'] == decision['thresholds']
    payload, _ = load_checkpoint(root / 'run/best.pt')
    assert payload['decision'] == decision


@pytest.mark.parametrize('kwargs', [
    {'memory_levels': []}, {'memory_levels': [1, 1]}, {'memory_levels': [4]},
    {'query_dimension': 15}, {'query_heads': 3}, {'query_layers': 0}])
def test_query_rejects_invalid_shapes(kwargs):
    with pytest.raises(ValueError):
        DFMSQueryIQ(**kwargs)
