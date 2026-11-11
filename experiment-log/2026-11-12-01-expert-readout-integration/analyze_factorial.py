"""Paired factorial differences, interaction, correction costs and six-label guards."""
import argparse
from common import *
from scipy.special import expit
from scipy.stats import t

def ci(values):
    v=np.asarray(values,float);m=float(v.mean());radius=float(t.ppf(.975,len(v)-1)*v.std(ddof=1)/np.sqrt(len(v))) if len(v)>1 else 0.
    return [m,m-radius,m+radius]

def comparisons():
    pairs=[]
    for e in CONFIG['experts']:
        pairs += [(e+'_segment',e+'_global'),(e+'_segment',e+'_none'),(e+'_global',e+'_none')]
        pairs += [(e+'_segment_guarded',e+'_none'),(e+'_global_guarded',e+'_none'),(e+'_segment_guarded',e+'_global_guarded'),
                  (e+'_segment_guarded',e+'_segment'),(e+'_global_guarded',e+'_global')]
    for r in CONFIG['readouts']:
        pairs += [('combo_'+r,'marginal_'+r),('marginal_'+r,'natural_'+r),('combo_'+r,'natural_'+r)]
    return pairs

def analyze(split,role):
    folder=ROOT/split/(role+'_results');summary=read(folder/'summary.json');seeds=summary['seeds']
    by={(r['method'],r['seed']):r for r in summary['results']};methods=sorted(set(m for m,s in by))
    meta=np.load(folder/'metadata.npz');y=meta['labels'].astype(bool);codes=y.astype(int)@(1<<np.arange(6))
    exact={};records=[]
    for method in methods:
        exact[method]=[]
        for seed in seeds:
            z=np.load(folder/f'{method}_{seed}.npz');exact[method].append((expit(z['logits'])>=z['thresholds'])==y)
        exact[method]=np.asarray(exact[method]).all(2)
    for a,b in comparisons():
        da=exact[a];db=exact[b];delta=da.mean(1)-db.mean(1)
        labels=[];passes=[]
        for seed in seeds:
            ra=by[a,seed];rb=by[b,seed]
            fp=np.array([aa['FPR']-bb['FPR'] for aa,bb in zip(ra['labels'],rb['labels'])])
            weak=np.array([aa['weak_recall']-bb['weak_recall'] for aa,bb in zip(ra['labels'],rb['labels'])])
            passes.append(bool((fp<=.005+1e-12).all() and (weak>=-.01-1e-12).all()))
            labels.append({'seed':seed,'fpr_difference':fp.tolist(),'weak_recall_difference':weak.tolist()})
        fp=np.mean([r['fpr_difference'] for r in labels],axis=0);weak=np.mean([r['weak_recall_difference'] for r in labels],axis=0)
        row={'candidate':a,'reference':b,'exact_difference':ci(delta),'seed_differences':delta.tolist(),'positive_seeds':int(np.sum(delta>0)),
            'bce_difference':ci([by[a,s]['metrics']['BCE']-by[b,s]['metrics']['BCE'] for s in seeds]),
            'corrected_mean':float((da&~db).sum(1).mean()),'new_errors_mean':float((~da&db).sum(1).mean()),
            'mean_fpr_difference':fp.tolist(),'mean_weak_recall_difference':weak.tolist(),'mean_guards_all_six':bool((fp<=.005+1e-12).all() and (weak>=-.01-1e-12).all()),
            'per_seed_guard_passes':sum(passes),'per_seed_labels':labels}
        if [a,b] in CONFIG['primary_comparisons']:
            rng=np.random.default_rng(20261112);identity=(da.astype(float)-db).mean(0);boot=np.zeros(2000)
            for code in range(1,64):
                group=np.flatnonzero(codes==code);boot+=identity[rng.choice(group,(2000,len(group)))].sum(1)/len(y)
            row['identity_bootstrap95']=np.quantile(boot,[.025,.975]).tolist()
        records.append(row)
    interaction=[]
    for base_readout in ['none','global']:
        values=(exact['combo_segment'].mean(1)-exact['combo_'+base_readout].mean(1))-(exact['marginal_segment'].mean(1)-exact['marginal_'+base_readout].mean(1))
        interaction.append({'contrast':f'(C_segment-C_{base_readout})-(B_segment-B_{base_readout})','exact_difference':ci(values),'seed_differences':values.tolist()})
    means={m:{k:float(np.mean([by[m,s]['metrics'][k] for s in seeds])) for k in by[m,seeds[0]]['metrics']} for m in methods}
    result={'split':split,'role':role,'identities':len(y),'seeds':seeds,'means':means,'comparisons':records,'interactions':interaction,
        'guards':'all six labels, FPR +.005 / weak recall -.01; diagnostic, not guaranteed on evaluation','multiplicity':'intervals unadjusted; primary contrasts specified in protocol','time':stamp()}
    write(ROOT/'analysis'/f'{split}_{role}.json',result)
    for r in records:
        if [r['candidate'],r['reference']] in CONFIG['primary_comparisons'] or r['candidate'].endswith('segment_guarded'):
            print(split,role,r['candidate'],r['reference'],'Exact pp',np.round(np.asarray(r['exact_difference'])*100,4).tolist(),'positive',r['positive_seeds'],'all6guard',r['mean_guards_all_six'],r['per_seed_guard_passes'],flush=True)
    print('INTERACTIONS',split,[(r['contrast'],np.round(np.array(r['exact_difference'])*100,4).tolist()) for r in interaction],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--split',required=True);p.add_argument('--role',default='development');a=p.parse_args();analyze(a.split,a.role)
