"""Paired target presence/power/removal diagnostics with frozen decisions."""
import argparse
from common import *
from scipy.special import expit
from sklearn.metrics import average_precision_score,roc_auc_score
from factorial_fusion import cache_scores,apply_record
from analyze_factorial import ci

VIEWS=['gap0','gap10','gap15','absent']
METHODS=['natural_none','natural_segment','marginal_none','marginal_global','marginal_segment','marginal_segment_guarded',
         'combo_none','combo_global','combo_segment','combo_segment_guarded']

def load_view(path):
    x,positions,targets=helper.load_views(path)
    with h5py.File(path,'r') as f:
        labels=np.asarray(f['labels'],dtype=np.float32).T
        relative=np.asarray(f['relative_power_dB'],dtype=float).T
        identity=np.asarray(f['receiver_noise_seed']).reshape(-1).astype(np.int64)
    assert len(labels)==len(x)==len(targets)==len(identity)
    return x,{'positions':positions,'targets':targets,'labels':labels,'relative':relative,'identity':identity}

def evaluate_physical(split,role):
    if role=='confirmation':
        for path,digest in read(ROOT/'final_freeze.json')['files'].items():assert sha(path)==digest,path
    folder=ROOT/split/('physical_'+role);receipt=read(folder/'generation_receipt.json')
    source=role_manifest(split,role);assert receipt['source_sha256']==source['sha256']
    out=ROOT/split/('physical_'+role+'_results');out.mkdir(parents=True,exist_ok=True)
    # Evaluate all 19 frozen decisions; METHODS is only the predeclared comparison subset.
    for view in VIEWS:
        path=folder/(view+'.mat');x,meta=load_view(path)
        manifest={'path':str(path),'sha256':sha(path),'indices':list(range(len(x)))}
        np.savez(out/(view+'_metadata.npz'),**meta)
        for seed in CONFIG['seeds']:
            scores={k:cache_scores(split,k,seed,'physical_'+role+'_'+view,manifest,x) for k in ['reference','natural','marginal','combo','global','segment']}
            for decision in sorted((ROOT/split/'decision_models').glob(f'*_{seed}.json')):
                state=read(decision);z=apply_record(scores,state)
                np.savez(out/(decision.stem+'_'+view+'.npz'),logits=z,thresholds=state['thresholds'])
        print('PHYSICAL_SCORED',split,role,view,len(x),flush=True)
    analyze(split,role)

def analyze(split,role):
    out=ROOT/split/('physical_'+role+'_results');meta={v:np.load(out/(v+'_metadata.npz')) for v in VIEWS}
    pos=meta['gap0']['positions'];target=meta['gap0']['targets'].astype(int);n=len(pos);rows=np.arange(n)
    source=role_manifest(split,role);original=metadata(source['path'],source['indices'])
    for view in VIEWS:
        np.testing.assert_array_equal(meta[view]['positions'],pos);np.testing.assert_array_equal(meta[view]['targets'],target)
        np.testing.assert_array_equal(meta[view]['identity'],original['identity'][pos])
        labels=original['labels'][pos].copy()
        if view=='absent':labels[rows,target]=0
        np.testing.assert_array_equal(labels,meta[view]['labels'])
    records=[];scores={};binary={}
    decisions=sorted((ROOT/split/'decision_models').glob('*_2200.json'))
    methods=[p.stem.rsplit('_',1)[0] for p in decisions]
    for method in methods:
        for seed in CONFIG['seeds']:
            p={};pred={}
            for view in VIEWS:
                data=np.load(out/f'{method}_{seed}_{view}.npz');p[view]=expit(data['logits'][rows,target]);pred[view]=p[view]>=data['thresholds'][target]
                scores[method,seed,view]=p[view];binary[method,seed,view]=pred[view]
            stats=[]
            for j,label in enumerate(LABELS):
                mask=target==j
                assert mask.any()
                for view in ['gap0','gap10','gap15']:
                    positive=p[view][mask];negative=p['absent'][mask];truth=np.r_[np.ones(mask.sum()),np.zeros(mask.sum())]
                    stats.append({'label':label,'view':view,'n':int(mask.sum()),'TPR':float(pred[view][mask].mean()),
                        'absent_FPR':float(pred['absent'][mask].mean()),'paired_order_accuracy':float(np.mean((positive>negative)+.5*(positive==negative))),
                        'controlled_pair_AUC':float(roc_auc_score(truth,np.r_[positive,negative])),
                        'controlled_pair_AP':float(average_precision_score(truth,np.r_[positive,negative]))})
            records.append({'method':method,'seed':seed,'n':n,'gap0_TPR':float(pred['gap0'].mean()),'gap10_TPR':float(pred['gap10'].mean()),
                'gap15_TPR':float(pred['gap15'].mean()),'absent_FPR':float(pred['absent'].mean()),'labels':stats})
    pairs=[('marginal_segment','marginal_none'),('marginal_segment','marginal_global'),('combo_segment','combo_none'),('combo_segment','combo_global'),
           ('marginal_segment_guarded','marginal_none'),('combo_segment_guarded','combo_none')]
    comparisons=[]
    for a,b in pairs:
        by_label=[]
        for j,label in enumerate(LABELS):
            mask=target==j
            by_label.append({'label':label,'n':int(mask.sum()),'gap15_TPR_difference':ci([binary[a,s,'gap15'][mask].mean()-binary[b,s,'gap15'][mask].mean() for s in CONFIG['seeds']]),
                'absent_FPR_difference':ci([binary[a,s,'absent'][mask].mean()-binary[b,s,'absent'][mask].mean() for s in CONFIG['seeds']])})
        comparison={'candidate':a,'reference':b,'gap15_TPR_difference':ci([binary[a,s,'gap15'].mean()-binary[b,s,'gap15'].mean() for s in CONFIG['seeds']]),
            'absent_FPR_difference':ci([binary[a,s,'absent'].mean()-binary[b,s,'absent'].mean() for s in CONFIG['seeds']]),'labels':by_label}
        comparisons.append(comparison)
        print('PHYSICAL_COMPARE',split,role,a,b,'TPRpp',np.round(np.array(comparison['gap15_TPR_difference'])*100,4).tolist(),
            'absentFPRpp',np.round(np.array(comparison['absent_FPR_difference'])*100,4).tolist(),flush=True)
    write(out/'summary.json',{'split':split,'role':role,'shared_identities':n,'target_counts':{label:int(np.sum(target==j)) for j,label in enumerate(LABELS)},
        'records':records,'comparisons':comparisons,'boundary':'Controlled paired observations with fixed background/noise. Pair AUC/AP is not natural single-observation recognition performance.',
        'source_sha256':sha(Path(__file__)),'time':stamp()})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--split',required=True);p.add_argument('--role',default='development');p.add_argument('--analyze-only',action='store_true');a=p.parse_args();setup()
    if a.analyze_only:analyze(a.split,a.role)
    else:evaluate_physical(a.split,a.role)
