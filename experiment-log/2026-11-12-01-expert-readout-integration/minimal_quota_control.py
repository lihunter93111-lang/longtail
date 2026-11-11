"""Future design only: minimum L1 CF intervention with exact margins."""
from common import *
from scipy.optimize import milp,Bounds,LinearConstraint
from scipy.sparse import lil_matrix

def solve(name,counts):
    counts=np.asarray(counts,int);codes=np.arange(1,64);labels=((codes[:,None]>>np.arange(6))&1).T
    cardinality=np.array([codes.astype(int).tolist()[i].bit_count() for i in range(63)])
    group=np.array([cardinality==k for k in range(1,7)],int)
    # x[63], absolute-change[63], head-choice[63], tail-choice[63]
    rows=[];lower=[];upper=[]
    def constraint(entries,lo,hi):
        row=np.zeros(252)
        for indexes,values in entries:row[indexes]=values
        rows.append(row);lower.append(lo);upper.append(hi)
    for vector in np.r_[labels,group]:constraint([(np.arange(63),vector)],float(vector@counts),float(vector@counts))
    for i in range(63):
        constraint([([i,63+i],[1,-1])],-np.inf,float(counts[i]))
        constraint([([i,63+i],[-1,-1])],-np.inf,float(-counts[i]))
        constraint([([i,126+i],[1,-450])],30,np.inf)
        constraint([([i,189+i],[1,450])],-np.inf,480)
    constraint([(np.arange(126,189),np.ones(63))],1,1);constraint([(np.arange(189,252),np.ones(63))],1,1)
    objective=np.r_[np.zeros(63),np.ones(63),np.zeros(126)]
    bounds=Bounds(np.r_[np.full(63,30),np.zeros(189)],np.r_[np.full(63,480),np.full(63,np.inf),np.ones(126)])
    started=time.time();solution=milp(objective,integrality=np.r_[np.ones(63),np.zeros(63),np.ones(126)],bounds=bounds,
        constraints=LinearConstraint(np.asarray(rows),lower,upper),options={'time_limit':120,'mip_rel_gap':0.0})
    report={'name':name,'status':int(solution.status),'message':solution.message,'elapsed_seconds':time.time()-started,
        'feasible':solution.x is not None,'time_limit_seconds':120,'role':'future_quota_only_not_trained'}
    if solution.x is not None:
        x=np.rint(solution.x[:63]).astype(int);assert x.sum()==8000 and x.min()==30 and x.max()==480
        np.testing.assert_array_equal(labels@x,labels@counts);np.testing.assert_array_equal(group@x,group@counts)
        report.update(counts=x.tolist(),label_margins=(labels@x).tolist(),cardinality_counts=(group@x).tolist(),
            label_IF=float(max(labels@x)/min(labels@x)),combination_IF=float(x.max()/x.min()),
            combination_gini=float((2*np.arange(1,64)-64)@np.sort(x)/(63*x.sum())),effective_combination_count=float(x.sum()**2/(x@x)),
            L1_distance=int(abs(x-counts).sum()),total_variation=float(abs(x-counts).sum()/16000),
            changed_codes=(np.flatnonzero(x!=counts)+1).tolist(),mip_gap=float(solution.mip_gap),
            objective_lower_bound=float(solution.mip_dual_bound),globally_optimal=bool(solution.status==0 and solution.mip_gap<1e-9))
    write(ROOT/'future_minimal_quota'/f'{name}.json',report);print(json.dumps(report),flush=True)

if __name__=='__main__':
    m=read(HERE/'matrix_protocol.json')
    for name in ['CF7p85_IF3','CF7p85_IF1']:solve(name,m['cells'][name]['counts'])
