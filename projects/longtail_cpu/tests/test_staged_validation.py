import h5py
import numpy as np
import pytest
from longtail.data import one_bit
from longtail.data.signals import DEFAULT_PHYSICS,Simulator
from longtail.data.dataset import read_metadata,IQDataset
from longtail.data.power_truth import read_power_truth


def test_one_bit_removes_only_target_and_preserves_single_view_descriptors():
    one_bit.initialize(DEFAULT_PHYSICS)
    block=one_bit.chunk_worker(([1,3],[2,5],[3900000001,3900000002],[129,323],['FMB','AMA','SawFM','SinFM','MFT','ISRJ']))
    y=block['label_selected'].T
    assert np.all((y[::2]^y[1::2]).sum(1)==1)
    np.testing.assert_array_equal(block['receiver_noise_seed'][:,::2],block['receiver_noise_seed'][:,1::2])
    assert len(np.unique(block['sample_id']))==4 and len(np.unique(block['pair_id']))==2
    assert block['raw_difference_relative_error'].max()<1e-12
    np.testing.assert_array_equal(block['physical_descriptor'][:,0],one_bit.physical_descriptor(block['iq_complex'][:,0]))
    active,plus,minus,position,error=one_bit.pair(Simulator(),1,2,3900000001,['FMB','AMA','SawFM','SinFM','MFT','ISRJ'])
    np.testing.assert_allclose(plus['iq']-minus,plus['clean_components'][position],rtol=1e-12,atol=1e-12)


def test_pair_ids_allow_shared_noise_but_reject_duplicate_samples(tmp_path):
    one_bit.initialize(DEFAULT_PHYSICS)
    block=one_bit.chunk_worker(([1],[2],[3900000001],[129],['FMB','AMA','SawFM','SinFM','MFT','ISRJ']))
    path=tmp_path/'pairs.h5'
    with h5py.File(path,'w') as f:
        f.attrs.update(identity_schema='one-bit-pairs-v1',power_truth_schema='per-label-v1',weak_gap_db=10.,noise_power_db=0.)
        for key,value in block.items():f[key]=value
    y,ids,length=read_metadata(path);assert length==3000 and len(np.unique(ids))==2
    assert read_power_truth(path,y) is not None
    with h5py.File(path,'r+') as f:f['sample_id'][0,1]=f['sample_id'][0,0]
    with pytest.raises(ValueError,match='identities'):read_metadata(path)


def test_pairing_rejects_empty_or_already_present_background():
    with pytest.raises(ValueError):one_bit.pair(Simulator(),0,2,1,[])
    with pytest.raises(ValueError):one_bit.pair(Simulator(),5,2,1,[])


def test_joint_decision_preserves_fmb_and_calibration_constraints():
    from longtail.staged_decision import constrained
    from longtail.metrics import probabilities
    rng=np.random.default_rng(20);y=rng.integers(0,2,(120,6)).astype(bool)
    z=np.where(y,2.,-2.)+rng.normal(0,.8,(120,6));weak=y&(rng.random(y.shape)<.4)
    th,meta=constrained(y,z,weak,[.5]*6,[.5]*6,np.ones(6)*.05,.03,.03)
    assert meta['feasible'] and th[0]==.5
    pred=probabilities(z)>=th
    assert np.all(np.array(meta['actual_fpr'])[1:]<=.05+1e-12)
    for c in [2,4,5]:assert meta['actual_weak_recall'][c]>=meta['minimum_weak_recall'][c]-1e-12
    assert all(all(b>=a for a,b in zip(trace,trace[1:])) for trace in meta['traces'])


def test_joint_decision_discloses_infeasible_weak_recall_instead_of_relaxing():
    from longtail.staged_decision import constrained
    y=np.tile([[0,0,1,0,0,0],[1,1,0,1,1,1]],(8,1)).astype(bool)
    z=np.where(y,2.,-2.);z[:,2]=np.where(y[:,2],-2.,2.)
    weak=y.copy()
    th,meta=constrained(y,z,weak,[.01]*6,[.01]*6,np.zeros(6),0.,0.)
    assert th is None and 2 in meta['infeasible_labels']


def test_classifier_retraining_does_not_share_or_change_source_weights():
    import torch
    from longtail.models.decision_heads import FiveLabelHead
    weight=torch.randn(6,128);bias=torch.randn(6);before=weight.clone()
    head=FiveLabelHead(weight,bias,dropout=0.)
    optimizer=torch.optim.SGD(head.parameters(),lr=.1)
    loss=head(torch.randn(32,128)).square().mean();loss.backward();optimizer.step()
    assert torch.equal(weight,before) and not torch.equal(head.linear.weight,weight[1:])
    assert sum(p.numel() for p in head.parameters())==645


def test_local_residual_initially_matches_base_and_disabled_target_stays_fixed():
    import torch
    from longtail.models.decision_heads import LocalResidual,residual_logits
    local=LocalResidual(active=[True,False,True]);x=torch.randn(32,128);base=torch.randn(32,6)
    assert torch.equal(residual_logits(base,local(x)),base)
    optimizer=torch.optim.SGD(local.parameters(),lr=.1)
    for _ in range(2):
        optimizer.zero_grad();loss=residual_logits(base,local(x))[:,2].square().mean();loss.backward();optimizer.step()
    out=residual_logits(base,local(x))
    assert torch.equal(out[:,[0,1,3,4]],base[:,[0,1,3,4]]) and not torch.equal(out[:,2],base[:,2])


def test_fixed_head_sampling_is_independent_and_has_equal_draw_budgets():
    from longtail.staged_training import draw_indices
    y=((np.arange(1,64)[:,None]>>np.arange(6))&1).astype('uint8');y=np.tile(y,(160,1))
    a=draw_indices(y,2050,1,'natural');b=draw_indices(y,2051,1,'natural');m=draw_indices(y,2050,1,'mixed')
    assert len(a)==len(b)==len(m)==8000 and len(np.unique(a))==8000
    assert not np.array_equal(a,b);np.testing.assert_array_equal(a,draw_indices(y,2050,1,'natural'))


def test_one_bit_margin_moves_only_target_logits_in_correct_directions():
    import torch
    from longtail.models.decision_heads import pair_margin_loss
    logits=torch.zeros(8,6,requires_grad=True);target=torch.tensor([2,4,5,2])
    loss=pair_margin_loss(logits,target);assert loss.item()==.5;loss.backward()
    for i,c in enumerate(target):
        assert logits.grad[2*i,c]==.25 and logits.grad[2*i+1,c]==-.25
    assert torch.count_nonzero(logits.grad)==8
