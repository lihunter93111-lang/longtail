"""Paired seed summaries, errors and label/combination trade-offs."""
import argparse
from common import *
from scipy.special import expit
from scipy.stats import t

COMPARISONS=[('combo','marginal'),('marginal','natural'),('combo','natural'),
 ('combo_affine','marginal_affine'),('marginal_affine','natural_affine'),('combo_affine','natural_affine'),
 ('combo_mean','marginal_mean'),('marginal_mean','natural_mean'),('combo_mean','natural_mean')]

def mean_ci(values):
    a=np.asarray(values,float); m=float(a.mean())
    if len(a)<2:return [m,None,None]
    radius=float(t.ppf(.975,len(a)-1)*a.std(ddof=1)/np.sqrt(len(a)))
    return [m,m-radius,m+radius]

def analyze(split,role):
    folder=ROOT/split/(role+'_results')
    data=read(folder/'summary.json')
    by={(r['method'],r['seed']):r for r in data['results']}
    methods=sorted(set(r['method'] for r in data['results']))
    means={name:{metric:float(np.mean([by[name,s]['metrics'][metric] for s in data['seeds']])) for metric in by[name,data['seeds'][0]]['metrics']} for name in methods}
    meta=np.load(folder/'metadata.npz');y=meta['labels'].astype(bool);codes=y.astype(int)@(1<<np.arange(6))
    pairs=[]
    for candidate,reference in COMPARISONS:
        diffs=[]; bce=[];corrections=[];new_errors=[];label_rows=[];guards=[];identity_differences=[]
        for seed in data['seeds']:
            a=by[candidate,seed];b=by[reference,seed]
            diffs.append(a['metrics']['f1_exact']-b['metrics']['f1_exact'])
            bce.append(a['metrics']['BCE']-b['metrics']['BCE'])
            za=np.load(folder/f'{candidate}_{seed}.npz');zb=np.load(folder/f'{reference}_{seed}.npz')
            pa=expit(za['logits'])>=za['thresholds'];pb=expit(zb['logits'])>=zb['thresholds']
            ea=(pa==y).all(1);eb=(pb==y).all(1)
            corrections.append(int((ea&~eb).sum()));new_errors.append(int((~ea&eb).sum()))
            identity_differences.append(ea.astype(float)-eb.astype(float))
            fpr=np.array([aa['FPR']-bb['FPR'] for aa,bb in zip(a['labels'],b['labels'])])
            weak=np.array([aa['weak_recall']-bb['weak_recall'] if aa['weak_recall'] is not None and bb['weak_recall'] is not None else np.nan for aa,bb in zip(a['labels'],b['labels'])])
            guards.append(bool((fpr<=.005+1e-12).all() and (weak[[2,4,5]]>=-.01-1e-12).all()))
            label_rows.append({'seed':seed,'fpr_difference':fpr.tolist(),'weak_recall_difference':weak.tolist()})
        fpr_mean=np.mean([r['fpr_difference'] for r in label_rows],axis=0)
        weak_mean=np.mean([r['weak_recall_difference'] for r in label_rows],axis=0)
        rng=np.random.default_rng(20261111)
        per_identity=np.mean(identity_differences,axis=0)
        groups=[np.flatnonzero(codes==code) for code in range(1,64)]
        boot=np.zeros(2000)
        for group in groups:boot+=per_identity[rng.choice(group,(2000,len(group)),replace=True)].sum(1)/len(y)
        row={'candidate':candidate,'reference':reference,'exact_difference':mean_ci(diffs),'bce_difference':mean_ci(bce),'positive_seeds':sum(d>0 for d in diffs),'seed_differences':diffs,'identity_bootstrap95':np.quantile(boot,[.025,.975]).tolist(),'corrected_mean':float(np.mean(corrections)),'new_errors_mean':float(np.mean(new_errors)),'mean_fpr_difference':fpr_mean.tolist(),'mean_weak_recall_difference':weak_mean.tolist(),'mean_guards':bool((fpr_mean<=.005+1e-12).all() and (weak_mean[[2,4,5]]>=-.01-1e-12).all()),'per_seed_guard_passes':sum(guards),'per_seed_label_differences':label_rows}
        pairs.append(row)
    result={'split':split,'role':role,'n':len(y),'seeds':data['seeds'],'means':means,'comparisons':pairs,'time':stamp()}
    write(ROOT/'analysis'/f'{split}_{role}.json',result)
    for pair in pairs:
        print(split,role,pair['candidate'],pair['reference'],'Exact pp',np.round(np.array(pair['exact_difference'])*100,4).tolist(),'positive',pair['positive_seeds'],'guard',pair['mean_guards'],flush=True)
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--split',required=True);parser.add_argument('--role',default='development');args=parser.parse_args()
    analyze(args.split,args.role)
