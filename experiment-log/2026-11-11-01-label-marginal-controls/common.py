"""Current campaign paths and shared historical numerical interfaces."""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from datetime import datetime, timezone
import torch
import numpy as np
import h5py

HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE/'protocol.json').read_text(encoding='utf-8'))
ROOT = Path(CONFIG['storage'])
SOURCE = Path(CONFIG['remote_source'])
BASELINES = Path(CONFIG['baseline_root'])
PREVIOUS = Path(CONFIG['previous_root'])
spec = importlib.util.spec_from_file_location('matched_weak_helpers', SOURCE/'experiment-log/2026-11-06-05-weak-component-goal/run_weak_goal.py')
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
base = helper.base
LABELS = helper.LABELS
QUOTAS = json.loads((SOURCE/'experiment-log/2026-11-06-01-lt-abc/2026-11-06-lt-abc-quotas.json').read_text(encoding='utf-8'))
read = lambda path: json.loads(Path(path).read_text(encoding='utf-8'))
write = base.write
sha = base.sha

def stamp():
    return datetime.now(timezone.utc).isoformat()

def setup():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert torch.cuda.is_available()

def objective_weights(y, mixture=.25):
    codes=y.astype(int)@(1<<np.arange(6))
    cardinality=y.sum(1).astype(int)
    combo_counts=np.bincount(codes,minlength=64)
    cardinality_counts=np.bincount(cardinality,minlength=7)
    possible=np.bincount([int(c).bit_count() for c in range(1,64)],minlength=7)
    n=len(y)
    q=(1-mixture)/n+mixture*cardinality_counts[cardinality]/(n*possible[cardinality]*combo_counts[codes])
    q/=q.sum()
    p=y.mean(0,dtype=np.float64)
    target=q@y
    positive=target/p
    negative=(1-target)/(1-p)
    marginal=y*positive+(1-y)*negative
    combo=n*q[:,None]*np.ones((1,6))
    np.testing.assert_allclose((marginal*y).mean(0),(combo*y).mean(0),atol=1e-12)
    np.testing.assert_allclose((marginal*(1-y)).mean(0),(combo*(1-y)).mean(0),atol=1e-12)
    np.testing.assert_allclose(marginal.mean(0),np.ones(6),atol=1e-12)
    return marginal.astype(np.float32),combo.astype(np.float32),{'p':p.tolist(),'target_p':target.tolist(),'positive_weight':positive.tolist(),'negative_weight':negative.tolist(),'q':q.tolist()}

def metadata(path,indices):
    indices=np.asarray(indices,dtype=np.int64)
    with h5py.File(path,'r') as f:
        y=np.asarray(f['known_label'],dtype=np.float32).T[indices]
        noise=np.asarray(f['receiver_noise_seed']).reshape(-1).astype(np.int64)[indices]
        power=np.full(y.shape,np.nan)
        for row,index in enumerate(indices):
            power[row,y[row].astype(bool)]=np.asarray(f[f['component_nominal_JNR_dB'][0,index]]).reshape(-1)
    relative=power-np.where(y.astype(bool),power,-np.inf).max(1,keepdims=True)
    return {'labels':y,'identity':noise,'relative':relative,'weak':y.astype(bool)&(y.sum(1,keepdims=True)>1)&(relative<=-10+1e-6)}

def evaluate(z,meta,thresholds):
    metrics,labels,combos,_=helper.metrics(z,meta['labels'],thresholds,QUOTAS['tail_codes'])
    from scipy.special import expit
    pred=expit(z)>=np.asarray(thresholds)
    for j,row in enumerate(labels):
        mask=meta['weak'][:,j]
        row['weak_n']=int(mask.sum())
        row['weak_recall']=float(pred[mask,j].mean()) if mask.any() else None
    return {'metrics':metrics,'labels':labels,'combos':combos}
