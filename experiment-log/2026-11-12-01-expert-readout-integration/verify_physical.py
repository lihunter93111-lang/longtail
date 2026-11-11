"""Independent saved-waveform pairing and score arithmetic checks."""
import argparse
from common import *
from scipy.special import expit
from scipy.stats import rankdata
from verify_core import average_precision

def verify(role):
    rows=[];checks=0;largest=0.;wave_errors=[]
    def equal(a,b):
        nonlocal checks,largest
        err=abs(float(a)-float(b));checks+=1;largest=max(largest,err);assert err<1e-9,(a,b)
    for split in CONFIG['splits']:
        folder=ROOT/split/('physical_'+role);out=ROOT/split/('physical_'+role+'_results');summary=read(out/'summary.json');raw={}
        for view in ['gap0','gap10','gap15','absent']:
            with h5py.File(folder/(view+'.mat'),'r') as f:
                raw[view]=np.stack([np.asarray(f['x_real']).T,np.asarray(f['x_imag']).T],1).astype(float)
        base_delta=raw['gap0']-raw['absent']
        for view,gap in [('gap10',10),('gap15',15)]:
            predicted=base_delta*10**(-gap/20);actual=raw[view]-raw['absent'];err=np.linalg.norm((actual-predicted).reshape(len(actual),-1),axis=1)/np.linalg.norm(predicted.reshape(len(actual),-1),axis=1)
            assert float(err.max())<1e-5
            wave_errors.append({'split':split,'view':view,'maximum_relative_L2_error_after_float32_storage':float(err.max())})
        del raw,base_delta
        metadata_view=np.load(out/'gap0_metadata.npz');targets=metadata_view['targets'].astype(int);indices=np.arange(len(targets))
        for record in summary['records']:
            method=record['method'];seed=record['seed'];prob={};pred={}
            for view in ['gap0','gap10','gap15','absent']:
                data=np.load(out/f'{method}_{seed}_{view}.npz');prob[view]=expit(data['logits'][indices,targets]);pred[view]=prob[view]>=data['thresholds'][targets]
            for view in ['gap0','gap10','gap15']:equal(pred[view].mean(),record[view+'_TPR'])
            equal(pred['absent'].mean(),record['absent_FPR'])
            for stat in record['labels']:
                mask=targets==LABELS.index(stat['label']);n=int(mask.sum());view=stat['view'];p=prob[view][mask];q=prob['absent'][mask]
                equal(n,stat['n']);equal(pred[view][mask].mean(),stat['TPR']);equal(pred['absent'][mask].mean(),stat['absent_FPR'])
                equal(np.mean(np.where(p>q,1.,np.where(p==q,.5,0.))),stat['paired_order_accuracy'])
                scores=np.r_[p,q];ranks=rankdata(scores,method='average');auc=(ranks[:n].sum()-n*(n+1)/2)/(n*n)
                equal(auc,stat['controlled_pair_AUC']);equal(average_precision(np.r_[np.ones(n),np.zeros(n)],scores),stat['controlled_pair_AP'])
            rows.append({'split':split,'method':method,'seed':seed})
    result={'passed':True,'role':role,'models':len(rows),'scalars_recomputed':checks,'maximum_scalar_error':largest,'waveform_scaling_checks':wave_errors,
            'boundary':'Checks target-delta scaling of saved raw waveforms; absolute target correctness additionally relies on generator reconstruction receipts.',
            'source_sha256':sha(Path(__file__)),'time':stamp()}
    write(ROOT/(role+'_physical_verification.json'),result);print('PHYSICAL_VERIFIED',len(rows),checks,largest,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--role',default='development');a=p.parse_args();verify(a.role)
