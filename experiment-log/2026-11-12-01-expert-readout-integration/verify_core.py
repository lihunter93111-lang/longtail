"""Independent arithmetic checks for the complete factorial prediction set."""
import argparse
from common import *
from scipy.special import expit
from factorial_fusion import threshold_guard,ThreeScoreAffine

def average_precision(y,p):
    order=np.argsort(-p,kind='stable');truth=y[order];score=p[order]
    ends=np.r_[np.flatnonzero(np.diff(score)!=0),len(score)-1]
    tp=np.cumsum(truth)[ends];recall=tp/max(float(truth.sum()),1);precision=tp/(ends+1)
    return float(np.sum(np.diff(np.r_[0.,recall])*precision))

def verify(role):
    errors=[];count=0;maximum=0.;model_count=0
    def equal(actual,expected,label):
        nonlocal count,maximum
        error=abs(float(actual)-float(expected));count+=1;maximum=max(maximum,error)
        if error>1e-9:errors.append({'label':label,'error':error})
    for split in CONFIG['splits']:
        folder=ROOT/split/(role+'_results');summary=read(folder/'summary.json')
        assert len(summary['results'])==95 and summary['seeds']==CONFIG['seeds']
        m=role_manifest(split,role);meta=np.load(folder/'metadata.npz');y=meta['labels'];truth=y.astype(bool)
        with h5py.File(m['path'],'r') as f:np.testing.assert_array_equal(np.asarray(f['known_label'],dtype=np.float32).T[m['indices']],y)
        for row in summary['results']:
            path=folder/f"{row['method']}_{row['seed']}.npz";assert sha(path)==row['prediction_sha256']
            decision=ROOT/split/'decision_models'/f"{row['method']}_{row['seed']}.json";assert sha(decision)==row['decision_sha256']
            prediction=np.load(path);z=prediction['logits'];p=expit(z);chosen=p>=prediction['thresholds'];np.testing.assert_array_equal(prediction['thresholds'],read(decision)['thresholds'])
            equal(np.mean(np.logaddexp(0,z)-y*z),row['metrics']['BCE'],'BCE')
            equal(np.mean(np.all(chosen==truth,axis=1)),row['metrics']['f1_exact'],'Exact')
            equal(np.mean(np.all((z>=0)==truth,axis=1)),row['metrics']['fixed05_exact'],'fixed_exact')
            equal(np.mean((p-y)**2),row['metrics']['Brier'],'Brier')
            aps=[];eces=[];f1s=[]
            for j,label in enumerate(row['labels']):
                tp=np.sum(chosen[:,j]&truth[:,j]);fp=np.sum(chosen[:,j]&~truth[:,j]);fn=np.sum(~chosen[:,j]&truth[:,j])
                equal(fp,label['FP'],'FP');equal(fn,label['FN'],'FN');equal(fp/max(np.sum(~truth[:,j]),1),label['FPR'],'FPR')
                equal(tp/max(np.sum(truth[:,j]),1),label['recall'],'recall');f1s.append(2*tp/max(2*tp+fp+fn,1))
                weak=meta['weak'][:,j];equal(weak.sum(),label['weak_n'],'weak_n')
                if weak.any():equal(chosen[weak,j].mean(),label['weak_recall'],'weak_recall')
                ap=average_precision(y[:,j],p[:,j]);aps.append(ap);equal(ap,label['AP'],'AP')
                bins=np.minimum(np.floor(p[:,j]*15).astype(int),14)
                ece=sum(np.mean(bins==k)*abs(float(p[bins==k,j].mean())-float(y[bins==k,j].mean())) for k in range(15) if np.any(bins==k))
                eces.append(ece);equal(ece,label['ECE'],'ECE')
            equal(np.mean(aps),row['metrics']['macro_AP'],'macro_AP');equal(np.mean(eces),row['metrics']['ECE'],'ECE');equal(np.mean(f1s),row['metrics']['macro_F1'],'macro_F1');model_count+=1
    synthetic=np.array([[1]*6,[0]*6],dtype=np.float32);ref=(synthetic*2-1)*10
    thresholds,fallback,_=threshold_guard(np.zeros_like(ref),ref,{'labels':synthetic,'weak':synthetic.astype(bool)},[.5]*6)
    assert all(fallback) and thresholds==[.5]*6
    affine=ThreeScoreAffine();inputs=torch.tensor([[1.,2.,0.],[-2.,3.,0.]])
    torch.testing.assert_close(affine(inputs)[0],inputs[:,:1],rtol=0,atol=0)
    result={'passed':not errors,'role':role,'models':model_count,'scalars_recomputed':count,'maximum_absolute_error':maximum,'errors':errors,
        'unachievable_guard_falls_back_all_columns':True,'zero_local_initial_affine_preserves_reference':True,'source_sha256':sha(Path(__file__)),'time':stamp()}
    write(ROOT/(role+'_core_verification.json'),result);print(json.dumps(result),flush=True);assert not errors

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--role',default='development');a=p.parse_args();verify(a.role)
