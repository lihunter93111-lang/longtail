"""Describe frozen residuals; no fitting, threshold search or selection."""
import argparse
from common import *
from scipy.stats import rankdata

def auc(pos,neg):
    n=len(pos);m=len(neg);r=rankdata(np.r_[pos,neg],method='average');return float((r[:n].sum()-n*(n+1)/2)/(n*m))

def diagnose(role):
    rows=[];audit=read(ROOT/'input_audit.json')
    heads=[m for m in audit['models'] if m['kind'] in ['global','segment']]
    for split in CONFIG['splits']:
        meta=np.load(ROOT/split/('physical_'+role+'_results')/'gap0_metadata.npz');targets=meta['targets'].astype(int);n=len(targets)
        for seed in CONFIG['seeds']:
            for readout in ['global','segment']:
                delta={}
                for view in ['gap0','gap10','gap15','absent']:
                    scores=ROOT/split/'scores';name='physical_'+role+'_'+view+'.npz'
                    delta[view]=np.load(scores/f'{readout}_{seed}'/name)['logits']-np.load(scores/f'reference_{seed}'/name)['logits']
                for j,label in enumerate(LABELS):
                    mask=targets==j;d0=delta['absent'][mask,j];d15=delta['gap15'][mask,j]
                    rows.append({'split':split,'seed':seed,'readout':readout,'label':label,'n':int(mask.sum()),
                        'local_residual_all_views_zero':bool(all(np.all(d[:,j]==0) for d in delta.values())),
                        'gap15_delta_mean':float(d15.mean()),'absent_delta_mean':float(d0.mean()),
                        'gap15_minus_absent_mean':float((d15-d0).mean()),'gap15_delta_std':float(d15.std()),
                        'absent_delta_std':float(d0.std()),'local_residual_pair_AUC':auc(d15,d0),
                        'pair_order_accuracy':float(np.where(d15>d0,1.,np.where(d15==d0,.5,0.)).mean()),
                        'fused_local_coefficients':{e:read(ROOT/split/'decision_models'/f'{e}_{readout}_{seed}.json')['weights'][j][2] for e in CONFIG['experts']}})
    result={'role':role,'head_models':heads,'selected_epoch_zero_stages':sum(e==0 for h in heads for e in h['head_selected_epochs']),
        'total_stages':sum(len(h['head_selected_epochs']) for h in heads),'physical_residuals':rows,
        'interpretation':'Residual ordering is a diagnostic, not a standalone classifier or natural test metric. Zero-selected heads are retained; no candidates changed.',
        'time':stamp()}
    write(ROOT/'analysis'/f'readout_diagnostics_{role}.json',result);print('READOUT_DIAGNOSED',len(rows),'zero_stages',result['selected_epoch_zero_stages'],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--role',default='development');a=p.parse_args();diagnose(a.role)
