"""Calibration-only joint decisions with explicit feasibility and fixed FMB."""
import numpy as np
from .metrics import probabilities,fit_thresholds
from .decision import fit_fpr_caps

RULES=['f1','matched_fpr','exact_fpr','exact_weak']


def constrained(targets,logits,weak,anchor,reference,max_fpr,weak_drop,recall_drop,max_passes=4):
    y=np.asarray(targets,bool);p=probabilities(logits);weak=np.asarray(weak,bool)
    if y.shape!=p.shape or weak.shape!=y.shape or np.any(weak&~y):raise ValueError('Invalid decision truth')
    anchor=np.asarray(anchor,float).copy();anchor[0]=.5;reference=np.asarray(reference,float).copy();reference[0]=.5
    pos=y.sum(0);neg=(~y).sum(0);weak_count=weak.sum(0)
    if (pos==0).any() or (neg==0).any():raise ValueError('Calibration needs positive and negative support')
    ref=p>=reference;recall_floor=(ref&y).sum(0)/pos-recall_drop
    weak_floor=np.divide((ref&weak).sum(0),weak_count,out=np.zeros(6),where=weak_count>0)-weak_drop
    candidates=[];predictions=[];infeasible=[]
    for c in range(6):
        values=np.array([.5]) if c==0 else np.unique(np.r_[p[:,c],anchor[c],reference[c],.5,1.])
        pred=p[:,c,None]>=values
        valid=(pred&~y[:,c,None]).sum(0)/neg[c]<=max_fpr[c]+1e-12
        valid&=(pred&y[:,c,None]).sum(0)/pos[c]>=recall_floor[c]-1e-12
        if c in [2,4,5] and weak_count[c]>0:valid&=(pred&weak[:,c,None]).sum(0)/weak_count[c]>=weak_floor[c]-1e-12
        if c==0:valid[:]=True
        candidates.append(values[valid]);predictions.append(pred[:,valid])
        if not valid.any():infeasible.append(c)
    meta={'fit_role':'calibration','feasible':not infeasible,'infeasible_labels':infeasible,'maximum_fpr':np.asarray(max_fpr).tolist(),
          'minimum_recall':recall_floor.tolist(),'minimum_weak_recall':weak_floor.tolist(),'weak_recall_drop':weak_drop,'recall_drop':recall_drop,
          'objective':'six-label Exact','starts':['nearest_anchor','nearest_0.5'],'max_passes':max_passes,'FMB_fixed':.5}
    if infeasible:return None,meta
    def dist(v):return np.abs(np.log(np.clip(v,1e-12,1-1e-12)/(1-np.clip(v,1e-12,1-1e-12))))
    solutions=[]
    for origin in [anchor,np.full(6,.5)]:
        th=np.array([v[np.argmin(np.abs(v-origin[c]))] for c,v in enumerate(candidates)])
        pred=p>=th;trace=[int((pred==y).all(1).sum())]
        for _ in range(max_passes):
            before=th.copy()
            for c in range(1,6):
                others=np.delete(pred==y,c,axis=1).all(1)
                score=((predictions[c]==y[:,c,None])&others[:,None]).sum(0)
                best=np.lexsort((candidates[c],dist(candidates[c]),-score))[0]
                th[c]=candidates[c][best];pred[:,c]=predictions[c][:,best]
            trace.append(int((pred==y).all(1).sum()))
            if np.array_equal(before,th):break
        solutions.append((trace[-1],float(dist(th).sum()),th,trace))
    winner=min(solutions,key=lambda v:(-v[0],v[1],tuple(v[2])));th=winner[2];pred=p>=th
    meta.update(calibration_exact=winner[0]/len(y),traces=[v[3] for v in solutions],
                actual_fpr=((pred&~y).sum(0)/neg).tolist(),actual_recall=((pred&y).sum(0)/pos).tolist(),
                actual_weak_recall=np.divide((pred&weak).sum(0),weak_count,out=np.zeros(6),where=weak_count>0).tolist())
    return th.tolist(),meta


def fit_rules(targets,logits,weak,reference_A_logits):
    y=np.asarray(targets,bool);f1=fit_thresholds(y,logits,None);f1[0]=.5
    a=fit_thresholds(y,reference_A_logits,None);a[0]=.5
    ap=probabilities(reference_A_logits)>=a;caps=(ap&~y).sum(0)
    matched,meta=fit_fpr_caps(y,logits,caps);matched[0]=.5
    max_fpr=np.minimum(1.,caps/(~y).sum(0)+.01)
    exact_fpr,m1=constrained(y,logits,weak,matched,matched,max_fpr,weak_drop=.01,recall_drop=.01)
    exact_weak,m2=constrained(y,logits,weak,matched,f1,max_fpr,weak_drop=.03,recall_drop=.03)
    return {'fit_role':'calibration','thresholds':{'f1':f1,'matched_fpr':matched,'exact_fpr':exact_fpr,'exact_weak':exact_weak},
            'metadata':{'matched_fpr':meta,'exact_fpr':m1,'exact_weak':m2},'A_f1_thresholds':a,
            'rule_on_future_calibration_infeasibility':'fall back to matched_fpr, record the fallback explicitly'}


def get_rule(decisions,name):
    threshold=decisions['thresholds'][name]
    return (threshold,False) if threshold is not None else (decisions['thresholds']['matched_fpr'],True)
