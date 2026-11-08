import numpy as np
import pytest
import torch
from longtail.data.sampling import sampling_probabilities,EpochWeightedSampler
from longtail.engine.trainer import build_loss
from longtail.config import load_config
from longtail.train import run
from longtail.metrics import fit_thresholds,probabilities


def test_combo_sampler_balances_combinations_and_multilabel_marginals():
    codes=np.repeat(np.arange(1,64),np.arange(1,64))
    y=((codes[:,None]>>np.arange(6))&1).astype(np.uint8)
    p=sampling_probabilities(y,'combo_aware')
    np.testing.assert_allclose(np.bincount(codes,weights=p)[1:],np.full(63,1/63))
    np.testing.assert_allclose(p@y,np.full(6,32/63))


def test_label_sampler_matches_uniform_label_mixture_not_uniform_marginals():
    y=np.array([[1,0],[1,0],[1,1],[0,1]])
    p=sampling_probabilities(y,'label_aware')
    np.testing.assert_allclose(p,[1/6,1/6,5/12,1/4])
    assert not np.allclose(p@y,[.5,.5])
    sampler=EpochWeightedSampler(y,'label_aware',7)
    sampler.set_epoch(2);first=list(sampler);sampler.set_epoch(2)
    assert first==list(sampler) and len(first)==len(y)
    assert sampler.audit()['draws']==len(y)


@pytest.mark.parametrize('name',['weighted_bce','logit_adjustment'])
def test_prior_losses_are_finite_and_reduce_to_bce_for_balanced_counts(name):
    z=torch.tensor([[-1000.,1000.],[.1,-.4]],requires_grad=True);y=torch.tensor([[1.,0.],[0.,1.]])
    loss=build_loss({'loss':name},[5,5],10,torch.device('cpu'))(z,y)
    torch.testing.assert_close(loss,torch.nn.functional.binary_cross_entropy_with_logits(z,y))
    loss.backward();assert torch.isfinite(z.grad).all()
    z=torch.tensor([[.2,-.3]],requires_grad=True);y=torch.tensor([[1.,0.]])
    actual=build_loss({'loss':name},[8,2],10,torch.device('cpu'))(z,y)
    prior=torch.tensor([.8,.2])
    expected=(torch.nn.functional.binary_cross_entropy_with_logits(z,y,pos_weight=(1-prior)/prior)
              if name=='weighted_bce' else torch.nn.functional.binary_cross_entropy_with_logits(z+torch.logit(prior),y))
    torch.testing.assert_close(actual,expected)


@pytest.mark.parametrize('sampling',['label_aware','combo_aware'])
def test_sampled_training_keeps_epoch_draw_budget_and_records_exposures(experiment,sampling):
    root,config,_=experiment;config['train']['sampler']=sampling
    config['train']['loss']='weighted_bce' if sampling=='label_aware' else 'logit_adjustment'
    run(config)
    import json
    history=json.loads((root/'run/history.json').read_text())
    assert history[0]['samples_seen']==252 and history[0]['sampler_padding']==0
    assert history[0]['sampling']['draws']==252
    assert history[0]['sampling']['unique_fit_identities']<=252


def test_exact_f1_calibration_matches_observed_score_search_and_monotone_shifts():
    z=np.array([[-7.,.1],[-6.,.1],[-5.,.3],[-4.,.5],[-3.,.9]])
    y=np.array([[0,0],[0,1],[1,1],[1,0],[1,1]],dtype=np.uint8)
    t=fit_thresholds(y,z,None);p=probabilities(z)
    for c in range(2):
        def f1(th):
            pred=p[:,c]>=th;return 2*(pred&y[:,c].astype(bool)).sum()/(pred.sum()+y[:,c].sum())
        assert f1(t[c])==max(f1(th) for th in np.unique(p[:,c]))
    shifted=z+np.array([4.2,-2.3]);ts=fit_thresholds(y,shifted,None)
    np.testing.assert_array_equal(p>=t,probabilities(shifted)>=ts)
