import numpy as np
import pytest
import torch

from longtail.calibration_tradeoff import fit_policies, small_calibration_blocks, two_target_exact
from longtail.data import targeted_pairs
from longtail.data.signals import DEFAULT_PHYSICS
from longtail.metrics import probabilities
from longtail.models.temporal_heads import GlobalResidual
from longtail.targeted_validation import SawMFTHead, compose_targets, pair_schedule


def test_large_calibration_is_eight_disjoint_balanced_small_controls():
    y = ((np.repeat(np.arange(1,64),128)[:,None] >> np.arange(6)) & 1).astype(np.uint8)
    blocks = small_calibration_blocks(y)
    assert len(blocks) == 8 and all(len(b) == 1008 for b in blocks)
    np.testing.assert_array_equal(np.sort(np.concatenate(blocks)), np.arange(8064))
    for block in blocks:
        code = y[block].astype(int) @ (1 << np.arange(6))
        assert np.all(np.bincount(code, minlength=64)[1:] == 16)


def test_calibration_changes_only_two_targets_and_shares_fp_budgets():
    rng = np.random.default_rng(21)
    y = ((np.repeat(np.arange(1,64),128)[:,None] >> np.arange(6)) & 1).astype(np.uint8)
    z = np.where(y, 2., -2.) + rng.normal(0, 1., y.shape)
    weak = y.astype(bool) & (rng.random(y.shape) < .3)
    anchors = {'new': [.5]*6, 'old': [.4,.45,.5,.55,.5,.6]}
    bundle = fit_policies(y, {'new': z, 'old': z+.1}, weak, anchors)
    for model, values in bundle['models'].items():
        for rule, threshold in values['thresholds'].items():
            if threshold is None:
                continue
            np.testing.assert_array_equal(np.array(threshold)[[0,1,3,5]], np.array(anchors[model])[[0,1,3,5]])
        for rule, rate in [('fpr03',.03),('fpr05',.05),('fpr07',.07)]:
            pp = probabilities(z+(0.1 if model == 'old' else 0.)) >= values['thresholds'][rule]
            for c in [2,4]:
                assert pp[~y[:,c].astype(bool),c].mean() <= rate


def test_infeasible_weak_recall_is_not_silently_relaxed():
    y = np.tile([[1,0,1,1,0,1],[0,1,0,0,1,0]], (24,1)).astype(bool)
    z = np.where(y, -3., 3.)
    th, meta = two_target_exact(y,z,y,[.5]*6,[.5]*6,np.zeros(6),np.ones(6)*.99)
    assert th is None and not meta['feasible'] and set(meta['infeasible_labels']) == {2,4}


@pytest.mark.parametrize('gap_bin', [0,2])
def test_conditional_pairs_preserve_original_power_and_single_component_change(gap_bin):
    targeted_pairs.initialize(DEFAULT_PHYSICS)
    block = targeted_pairs.conditional_worker((1,2,gap_bin,4200000001+gap_bin*16384,2,129,['FMB','AMA','SawFM','SinFM','MFT','ISRJ']))
    y = block['label_selected'].T
    assert np.all((y[::2] ^ y[1::2]).sum(1) == 1)
    lo,hi = targeted_pairs.GAP_BINS[gap_bin]
    gap = -block['relative_power_dB'][2,1::2]
    assert np.all((gap >= lo) & (gap < hi))
    power = block['component_effective_JNR_dB']
    assert np.nanmin(power) >= 10 and np.nanmax(power) <= 30
    assert block['raw_difference_relative_error'].max() < 1e-12
    np.testing.assert_array_equal(block['receiver_noise_seed'][:,::2],block['receiver_noise_seed'][:,1::2])


def test_head_copy_initializes_from_reference_and_keeps_other_logits_exactly_fixed():
    torch.manual_seed(8)
    reference = GlobalResidual()
    with torch.no_grad():
        reference.output.weight.normal_(0,.1)
    state = {k:v.clone() for k,v in reference.state_dict().items()}
    copy = SawMFTHead(state)
    x = torch.randn(32,128)
    torch.testing.assert_close(copy(x),reference(x)[:,[0,1]],atol=1e-6,rtol=1e-5)
    records = {'candidate_logits':np.random.default_rng(8).normal(size=(32,6)).astype(np.float32),
               'linear_logits':np.zeros((32,6),np.float32)}
    opt = torch.optim.AdamW(copy.parameters(),lr=.01)
    opt.zero_grad();copy(x).square().mean().backward();opt.step()
    result = compose_targets(records,copy(x).detach().numpy())
    np.testing.assert_array_equal(result[:,[0,1,3,5]],records['candidate_logits'][:,[0,1,3,5]])
    assert all(torch.equal(v,state[k]) for k,v in reference.state_dict().items())


def test_small_pair_budgets_balance_target_gap_strata_across_training():
    pairs = {'pair_target':np.repeat([2,2,2,4,4,4],2), 'target_gap_bin':np.repeat([0,1,2,0,1,2],2)}
    for number in [1,2]:
        counts = np.zeros(6,int)
        for epoch in range(1,25):
            draws, strata = pair_schedule(pairs,2050,epoch,number)
            counts += np.bincount(strata,minlength=6)
            assert len(draws) == 250*number
            assert 250*(32-2*number)+2*len(draws) == 8000
        np.testing.assert_array_equal(counts,np.full(6,1000*number))
