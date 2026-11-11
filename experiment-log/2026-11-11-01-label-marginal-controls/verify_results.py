"""Independent numerical verification of delivered predictions and training receipts."""
import argparse
from common import *
from scipy.special import expit

def average_precision(y,z):
    order=np.argsort(-z,kind='stable');truth=y[order];score=z[order]
    ends=np.r_[np.flatnonzero(np.diff(score)!=0),len(score)-1]
    tp=np.cumsum(truth)[ends]
    recall=tp/max(float(truth.sum()),1)
    precision=tp/(ends+1)
    return float(np.sum(np.diff(np.r_[0,recall])*precision))

def verify(role):
    errors=[];count=0;maximum=0.;training=0
    def equal(actual,expected,label):
        nonlocal count,maximum
        error=abs(float(actual)-float(expected));count+=1;maximum=max(maximum,error)
        if error>1e-9:errors.append({'label':label,'error':error})
    for split in CONFIG['splits']:
        folder=ROOT/split/(role+'_results')
        summary=read(folder/'summary.json')
        meta=np.load(folder/'metadata.npz');y=meta['labels'];truth=y.astype(bool)
        for row in summary['results']:
            path=folder/f"{row['method']}_{row['seed']}.npz"
            assert sha(path)==row['prediction_sha256']
            prediction=np.load(path);z=prediction['logits'];p=expit(z);decision=p>=prediction['thresholds']
            equal(np.mean(np.logaddexp(0,z)-y*z),row['metrics']['BCE'],'BCE')
            equal(np.mean(np.all(decision==truth,axis=1)),row['metrics']['f1_exact'],'Exact')
            equal(np.mean(np.all((z>=0)==truth,axis=1)),row['metrics']['fixed05_exact'],'fixed_exact')
            aps=[];eces=[]
            for j,label in enumerate(row['labels']):
                tp=np.sum(decision[:,j]&truth[:,j]);fp=np.sum(decision[:,j]&~truth[:,j]);fn=np.sum(~decision[:,j]&truth[:,j])
                equal(fp,label['FP'],'FP');equal(fn,label['FN'],'FN')
                equal(fp/max(np.sum(~truth[:,j]),1),label['FPR'],'FPR')
                equal(tp/max(np.sum(truth[:,j]),1),label['recall'],'recall')
                weak=meta['weak'][:,j]
                equal(np.sum(weak),label['weak_n'],'weak_n')
                if weak.any():equal(np.mean(decision[weak,j]),label['weak_recall'],'weak_recall')
                ap=average_precision(y[:,j],z[:,j]);aps.append(ap);equal(ap,label['AP'],'AP')
                bins=np.minimum(np.floor(p[:,j]*15).astype(int),14)
                ece=sum(np.mean(bins==k)*abs(float(p[bins==k,j].mean())-float(y[bins==k,j].mean())) for k in range(15) if np.any(bins==k))
                eces.append(ece);equal(ece,label['ECE'],'ECE')
            equal(np.mean(aps),row['metrics']['macro_AP'],'macro_AP')
            equal(np.mean(eces),row['metrics']['ECE'],'macro_ECE')
        for seed in CONFIG['seeds']:
            rec=read(ROOT/split/'training/marginal'/f'M_{seed}/receipt.json')
            assert rec['epochs']==40 and rec['updates']==10000
            assert sha(ROOT/split/'training/marginal'/f'M_{seed}/best_BCE.pth')==rec['checkpoint_sha256']
            original=read(PREVIOUS/split/'expert_training/ordinary_expert'/f'M_{seed}/receipt.json')
            assert rec['initial_state_sha256']==original['initial_state_sha256'] and rec['order_sha256']==original['order_sha256']
            protocol=read(ROOT/split/'training/marginal/protocol.json')
            for path,digest in protocol['source_sha256'].items(): assert sha(path)==digest
            training+=1
    result={'passed':not errors,'role':role,'scalars_recomputed':count,'maximum_absolute_error':maximum,'errors':errors,'new_complete_trainings':training,'time':stamp(),'verification_source_sha256':sha(Path(__file__))}
    write(ROOT/(role+'_verification.json'),result)
    print(json.dumps(result),flush=True)
    assert not errors

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--role',default='development');a=p.parse_args();verify(a.role)
