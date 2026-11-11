"""Development-only, fit-defined descriptive groups; no model selection."""
from common import *
from scipy.special import expit
from analyze_results import mean_ci

def main():
    reports=[]
    for split in CONFIG['splits']:
        manifest=read(BASELINES/split/'data/manifest.json')
        with h5py.File(manifest['fit_path'],'r') as f:
            fit=np.asarray(f['known_label'],dtype=np.float32).T[manifest['fit_A']]
        counts=np.bincount(fit.astype(int)@(1<<np.arange(6)),minlength=64)
        folder=ROOT/split/'development_results'
        meta=np.load(folder/'metadata.npz');y=meta['labels'].astype(bool)
        code=y.astype(int)@(1<<np.arange(6));card=y.sum(1)
        ranked=sorted(range(1,64),key=lambda c:(counts[c],c))
        groups={name:np.isin(code,ranked[i*21:(i+1)*21]) for i,name in enumerate(['low_frequency','middle_frequency','high_frequency'])}
        groups.update({f'cardinality_{k}':card==k for k in range(1,7)})
        groups['any_weak_target']=meta['weak'].any(1)
        groups['no_weak_target']=~meta['weak'].any(1)
        for j,label in enumerate(LABELS):
            groups[label+'_present']=y[:,j]
            groups[label+'_absent']=~y[:,j]
            groups[label+'_weak']=meta['weak'][:,j]
        pred={}
        for method in ['natural','marginal','combo']:
            pred[method]=[]
            for seed in CONFIG['seeds']:
                z=np.load(folder/f'{method}_{seed}.npz')
                pred[method].append((expit(z['logits'])>=z['thresholds'])==y)
        rows=[]
        for candidate,reference in [('combo','marginal'),('marginal','natural')]:
            for name,mask in groups.items():
                a=np.array(pred[candidate])[:,mask].all(2);b=np.array(pred[reference])[:,mask].all(2)
                rows.append({'candidate':candidate,'reference':reference,'group':name,'n':int(mask.sum()),'exact_difference':mean_ci(a.mean(1)-b.mean(1)),
                    'corrected_mean':float(np.mean((a&~b).sum(1))),'new_errors_mean':float(np.mean((~a&b).sum(1)))})
        weights,combo,design=objective_weights(fit)
        reports.append({'split':split,'fit_frequency_groups':{name:ranked[i*21:(i+1)*21] for i,name in enumerate(['low_frequency','middle_frequency','high_frequency'])},
            'fit_counts':counts.tolist(),'combo_effective_sample_size':float(combo[:,0].sum()**2/(combo[:,0]**2).sum()),
            'marginal_effective_sample_size_per_label':(weights.sum(0,dtype=np.float64)**2/np.square(weights,dtype=np.float64).sum(0)).tolist(),'rows':rows})
    write(ROOT/'analysis/background_development.json',{'role':'exploratory_development_only','no_selection':True,'group_rule':'fit frequency ascending, code ascending ties, 21 codes each; overlap across other groups',
        'source_sha256':sha(Path(__file__)),'reports':reports,'time':stamp()})
    print('BACKGROUND_DIAGNOSTIC_COMPLETE',flush=True)

if __name__=='__main__':main()
