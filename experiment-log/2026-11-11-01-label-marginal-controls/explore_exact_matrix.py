"""Bounded exact-margin feasibility retry; future data, never current model tuning."""
from common import *
from scipy.optimize import Bounds, LinearConstraint, milp

def main():
    path=HERE.parent/'2026-11-10-01-readout-integration/results/analysis/future_dual_tail_matrix.json'
    cells=read(path)['cells']
    quotas=np.asarray(cells['CF16_IF3']['counts'],int)
    target=np.asarray(cells['CF7p85_IF1']['label_margins'],int)
    labels=((np.arange(1,64)[:,None]>>np.arange(6))&1).astype(int)
    cardinality=labels.sum(1)
    choices=[(i,int(q)) for i in range(63) for q in np.unique(quotas[cardinality==cardinality[i]])]
    rows=[];rhs=[]
    for i in range(63):
        rows.append([int(dst==i) for dst,q in choices]);rhs.append(1)
    for k in range(1,7):
        values,counts=np.unique(quotas[cardinality==k],return_counts=True)
        for value,count in zip(values,counts):
            rows.append([int(cardinality[dst]==k and q==value) for dst,q in choices]);rhs.append(int(count))
    for j in range(6):
        rows.append([q*labels[dst,j] for dst,q in choices]);rhs.append(int(target[j]))
    result=milp(c=np.zeros(len(choices)),integrality=np.ones(len(choices)),bounds=Bounds(0,1),
        constraints=LinearConstraint(np.asarray(rows),rhs,rhs),options={'time_limit':300,'mip_rel_gap':0})
    record={'role':'future_data_only','new_training':False,'status':int(result.status),'message':str(result.message),
        'feasible_found':result.x is not None,'source_sha256':sha(Path(__file__)),'input_sha256':sha(path),'time':stamp()}
    if result.x is not None:
        selected=np.rint(result.x).astype(int)
        np.testing.assert_array_equal(np.asarray(rows)@selected,rhs)
        q=np.zeros(63,dtype=int)
        for use,(dst,value) in zip(selected,choices):q[dst]+=use*value
        np.testing.assert_array_equal(labels.T@q,target)
        assert q.sum()==8000 and q.max()/q.min()==16
        for k in range(1,7):np.testing.assert_array_equal(np.sort(q[cardinality==k]),np.sort(quotas[cardinality==k]))
        pool=base.candidates(ROOT/'split1/data','fit_pool',LABELS)
        indices=np.concatenate([np.flatnonzero(pool['codes']==i)[:count] for i,count in enumerate(q,1)]).tolist()
        assert len(indices)==8000
        record.update(counts=q.tolist(),label_margins=target.tolist(),combination_IF=16.,label_IF=float(target.max()/target.min()),integer_constraints_verified=True)
        write(ROOT/'future_matrix/CF16_IF1_exact/manifest.json',{'role':'future_fit_only_not_trained','path':str(pool['path']),'sha256':pool['sha256'],'indices':indices,**record})
    write(ROOT/'future_matrix/exact_retry.json',record)
    print(json.dumps(record),flush=True)

if __name__=='__main__':main()
