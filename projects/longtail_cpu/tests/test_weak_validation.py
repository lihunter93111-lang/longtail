import json
import numpy as np
import pytest
import torch
from longtail.data.sampling import EpochBudgetSampler
from longtail.data.signals import Simulator, DEFAULT_PHYSICS
from longtail.data.v3_fresh import compose_v3
from longtail.data.weak_validation import paired_sample, conditional_jnr
from longtail.engine.trainer import WeakPositiveBCE, build_loss, train_epoch


def test_draw_budget_uses_distinct_natural_identities_and_independent_epochs():
    code = np.repeat(np.arange(1,64), 4)
    y = ((code[:,None] >> np.arange(6)) & 1).astype(np.uint8)
    sampler = EpochBudgetSampler(y, 2050, 128)
    sampler.set_epoch(1); first = list(sampler)
    assert len(first) == len(set(first)) == 128
    assert sampler.audit()['draws'] == 128 and sampler.audit()['fit_pool_size'] == 252
    assert sampler.audit()['replacement'] is False
    sampler.set_epoch(2); assert list(sampler) != first
    sampler.set_epoch(1); assert list(sampler) == first
    with pytest.raises(ValueError): EpochBudgetSampler(y, 2050, 253)


def test_fixed_budget_training_does_not_scale_steps_with_pool(experiment):
    from longtail.train import run
    root, cfg, _ = experiment
    cfg['train']['draws_per_epoch'] = 128
    run(cfg)
    h = json.loads((root/'run/history.json').read_text())[0]
    assert h['samples_seen'] == 128 and h['sampler_padding'] == 0 and h['fit_pool_size'] == 252
    state = torch.load(root/'run/last.pt', map_location='cpu', weights_only=True)
    assert {int(v['step']) for v in state['optimizer']['state'].values()} == {8}


def test_hard_view_changes_only_power_and_preserves_noise_and_natural_equations():
    simulator = Simulator(DEFAULT_PHYSICS)
    classes = ['FMB','AMA','SawFM','SinFM','MFT','ISRJ']
    natural, hard = paired_sample(simulator, np.array([0,2,5]), classes, 3700000001, 2)
    original = compose_v3(simulator, ['FMB','SawFM','ISRJ'], 3700000001)
    np.testing.assert_array_equal(natural['iq'], original['iq'])
    assert hard['noise_seed'] == natural['noise_seed'] and hard['component_seeds'] == natural['component_seeds']
    np.testing.assert_array_equal(hard['receiver_noise'], natural['receiver_noise'])
    assert hard['relative'][1] <= -10
    assert np.all((hard['jnr'] >= 10) & (hard['jnr'] <= 30))
    measured = 10*np.log10(np.mean(abs(hard['clean_components'])**2, axis=1)/simulator.radar['noise_power'])
    np.testing.assert_allclose(measured, hard['jnr'], atol=1e-10)
    natural2, hard2 = paired_sample(simulator, np.array([0,2,5]), classes, 3700000001, 2)
    np.testing.assert_array_equal(hard['iq'], hard2['iq'])


def test_conditional_jnr_is_reproducible_and_rejects_single_component():
    for s in range(10):
        jnr = conditional_jnr(s, 3, 1)
        assert jnr[1] <= max(jnr)-10
        np.testing.assert_array_equal(jnr, conditional_jnr(s,3,1))
    with pytest.raises(ValueError): conditional_jnr(4,1,0)


def test_weak_positive_loss_has_unit_fit_weight_mass_and_selective_gradients():
    y = torch.tensor([[1,1,1,1,1,1],[0,0,1,0,1,1],[1,0,0,1,0,0],[0,1,0,0,0,0]],dtype=torch.float32)
    weak = y.bool(); counts = weak.sum(0)
    loss_fn = WeakPositiveBCE(counts, len(y), torch.device('cpu'))
    weights = (1 + .5*(weak & y.bool() & loss_fn.selected))/loss_fn.normalizer
    torch.testing.assert_close(weights.mean(0), torch.ones(6))
    z = torch.zeros_like(y,requires_grad=True)
    actual = loss_fn(z,y,weak)
    expected = (torch.nn.functional.binary_cross_entropy_with_logits(z,y,reduction='none')*weights).mean()
    torch.testing.assert_close(actual,expected)
    actual.backward(); assert torch.isfinite(z.grad).all()
    assert torch.all(weights[:,[0,1,3]]==1)
    with pytest.raises(ValueError): build_loss({'loss':'weak_positive_bce'},y.sum(0),len(y),torch.device('cpu'))


def test_weak_loss_training_reads_batch_truth_and_updates_parameters():
    model = torch.nn.Linear(2,6)
    x = torch.randn(4,2); y = torch.randint(0,2,(4,6)).float(); weak = y.bool()
    loss = WeakPositiveBCE(weak.sum(0),4,torch.device('cpu'))
    before = model.weight.detach().clone(); opt = torch.optim.AdamW(model.parameters(), lr=.01)
    config={'amp':'none','gradient_clip':5}
    scaler=torch.amp.GradScaler('cuda',enabled=False)
    value, samples=train_epoch(model,[{'signal':x,'target':y,'weak_mask':weak}],opt,loss,torch.device('cpu'),config,scaler)
    assert np.isfinite(value) and samples==4 and not torch.equal(before,model.weight)


def test_fpr_thresholds_respect_negative_budget_and_score_ties():
    from longtail.decision import fit_fpr_caps
    from longtail.metrics import probabilities
    y=np.array([[0,0],[0,0],[0,1],[1,1],[1,0]])
    z=np.array([[1.,0.],[1.,0.],[0.,2.],[2.,3.],[-1.,-1.]])
    t,meta=fit_fpr_caps(y,z,[1,0]);pred=probabilities(z)>=t
    assert all(meta['empirical_caps_satisfied'])
    assert (pred&~y.astype(bool)).sum(0).tolist()==meta['achieved_false_positives']
    assert meta['achieved_false_positives']==[0,0]
    json.dumps(meta)


def test_fpr_threshold_records_infeasible_saturated_ties():
    from longtail.decision import fit_fpr_caps
    t,meta=fit_fpr_caps(np.array([[0],[1]]),np.array([[1000.],[1000.]]),[0])
    assert t==[1.] and meta['empirical_caps_satisfied']==[False]
