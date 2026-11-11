"""Independent fixed-FPR diagnostic arithmetic, including slice AP."""
from common import *
from scipy.special import expit
from sklearn.metrics import average_precision_score

def verify():
    count=0;maximum=0.;models=0
    def equal(a,b):
        nonlocal count,maximum
        error=abs(float(a)-float(b));maximum=max(maximum,error);count+=1;assert error<1e-9,(a,b)
    for role in ['development','confirmation']:
        report=read(ROOT/'analysis'/f'matrix_operating_points_{role}.json')
        for r in report['records']:
            folder=ROOT/'matrix'/r['split']/r['cell'];name=f"{r['expert']}_{r['seed']}";state=folder/'models'/name/'fixed_fpr_operating_point.json'
            assert sha(state)==r['operating_point_sha256'];thresholds=read(state)['logit_thresholds'];source=folder/(role+'_results')/(name+'.npz')
            assert sha(source)==r['prediction_source_sha256'];z=np.load(source)['logits'];meta=np.load(source.parent/'metadata.npz');y=meta['labels'].astype(bool)
            pred=z>=thresholds;equal(np.all(pred==y,1).mean(),r['exact']);equal(np.mean(np.logaddexp(0,z)-y*z),r['BCE'])
            for j,row in enumerate(r['labels']):
                pos=y[:,j];neg=~pos;weak=meta['weak'][:,j]
                equal(pred[neg,j].mean(),row['FPR']);equal(pred[pos,j].mean(),row['recall']);equal(pred[weak,j].mean(),row['weak_recall'])
                equal(weak.sum(),row['weak_positive_n']);equal(neg.sum(),row['negative_n']);mask=neg|weak
                equal(average_precision_score(y[mask,j],expit(z[mask,j])),row['weak_vs_negative_AP'])
            models+=1
    write(ROOT/'operating_points_verification.json',{'passed':True,'models':models,'scalars':count,'maximum_error':maximum,'time':stamp()})
    print('OPERATING_POINTS_VERIFIED',models,count,maximum,flush=True)

if __name__=='__main__':verify()
