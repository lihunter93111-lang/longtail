"""Evidence-source and active-capacity controls for the 24-epoch comparison."""
from copy import deepcopy
import pytest
import torch
from longtail.models.dfms import DFMSIQ
from longtail.models.dfms_query import DFMSQueryIQ
from longtail.models.dfms_query_variants import DFMSFusedQueryIQ, DFMSUniformIQ
from longtail.train import run
from longtail.evaluate import run as evaluate
from longtail.engine.checkpoint import load_checkpoint
from longtail.config import resolve_config


def test_evidence_variants_have_equal_capacity_and_initialized_backbone():
    models = []
    for build in [DFMSIQ, lambda: DFMSQueryIQ(memory_levels=(3,)), DFMSFusedQueryIQ, DFMSUniformIQ]:
        torch.manual_seed(2050);models.append(build())
    assert [sum(p.numel() for p in m.parameters()) for m in models] == [257511, 300211, 300211, 300211]
    for model in models[1:]:
        for key, value in models[0].state_dict().items():
            torch.testing.assert_close(value, model.state_dict()[key], rtol=0, atol=0)
    for key, value in models[1].state_dict().items():
        torch.testing.assert_close(value, models[2].state_dict()[key], rtol=0, atol=0)
    single = resolve_config('configs/query_evidence24_single.json')
    fused = resolve_config('configs/query_evidence24_fused.json')
    assert single['model']['kwargs'] == fused['model']['kwargs']
    for suffix in ['cpu','single','fused','uniform']:
        config = resolve_config(f'configs/query_evidence24_{suffix}.json')
        assert config['train'] == single['train']
        assert config['dataloader'] == single['dataloader']


@pytest.mark.parametrize('build', [DFMSFusedQueryIQ, DFMSUniformIQ])
def test_evidence_variants_gradients_and_disabled_residual(build):
    model = build(width=4, query_dimension=16, query_feedforward=32)
    values = torch.randn(3, 2, 67)
    output = model(values)
    torch.nn.functional.binary_cross_entropy_with_logits(output, torch.randint(0,2,output.shape).float()).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert all(p.grad.abs().sum()>0 for p in model.parameters())
    model.eval()
    with torch.no_grad():
        logits, d = model.forward_diagnostics(values)
        assert d['memory_lengths']==[8]
        torch.testing.assert_close(logits,model(values))
        model.query_scale.zero_()
        reference = DFMSIQ(width=4).eval()
        reference.load_state_dict({k:v for k,v in model.state_dict().items() if k in reference.state_dict()})
        torch.testing.assert_close(model(values),reference(values))


def test_fused_query_reuses_the_actual_temporal_features():
    model = DFMSFusedQueryIQ(width=4,query_dimension=16,query_feedforward=32).eval()
    captured = {}
    first = model.temporal_backbone.register_forward_hook(lambda module,args,result:captured.update(temporal=result))
    second = model.memory_projections[0].register_forward_pre_hook(lambda module,args:captured.update(memory_input=args[0]))
    with torch.no_grad():model(torch.randn(2,2,67))
    first.remove();second.remove()
    assert captured['memory_input'] is captured['temporal']


def test_uniform_readout_is_invariant_to_token_order():
    model = DFMSUniformIQ(width=4,query_dimension=16,query_feedforward=32).eval()
    tokens = torch.randn(2,31,16)
    with torch.no_grad():
        torch.testing.assert_close(model._read_tokens(tokens),model._read_tokens(tokens[:,torch.randperm(31)]))
        _,d=model.forward_diagnostics(torch.randn(2,2,67))
        torch.testing.assert_close(d['attention'],torch.full((2,6,8),1/8))


@pytest.mark.parametrize('entrypoint', ['DFMSFusedQueryIQ','DFMSUniformIQ'])
def test_evidence_variant_checkpoint_reload(experiment,entrypoint):
    root,config,_=experiment;config=deepcopy(config)
    config['model']={'entrypoint':f'longtail.models.dfms_query_variants:{entrypoint}',
                     'kwargs':{'width':4,'query_dimension':16,'query_feedforward':32}}
    run(config)
    payload,_=load_checkpoint(root/'run/best.pt');decision=deepcopy(payload['decision'])
    metrics=evaluate(root/'run/best.pt',output=root/'variant_eval',device_name='cpu')
    assert metrics['calibrated']['thresholds']==decision['thresholds']
    assert load_checkpoint(root/'run/best.pt')[0]['decision']==decision
