"""Completed matched triads only; quota-specific tails and factorial contrasts."""
import argparse
from common import *
from scipy.special import expit
from scipy.stats import t
from verify_core import average_precision

MATRIX=read(HERE/'matrix_protocol.json')

def interval(values):
    v=np.asarray(values,float);n=len(v);m=float(v.mean())
    radius=float(t.ppf(.975,n-1)*v.std(ddof=1)/np.sqrt(n)) if n>1 else None
    return {'mean':m,'n_seeds':n,'ci95':None if radius is None else [m-radius,m+radius],'values':v.tolist(),
            'positive_seeds':int((v>0).sum())}

def analyze(role):
    cells=[];contrasts=[];cross=[];all_values={};checks=0;errors=[]
    def check(a,b,name):
        nonlocal checks
        checks+=1
        if not np.isclose(a,b,atol=1e-9,rtol=0):errors.append({'name':name,'actual':float(a),'expected':float(b)})
    for split in CONFIG['splits']:
        for cell in MATRIX['cells']:
            folder=ROOT/'matrix'/split/cell;result_folder=folder/(role+'_results')
            seeds=[s for s in MATRIX['seeds'] if (folder/f'matched_triad_{s}.json').exists() and all((result_folder/f'{e}_{s}.json').exists() for e in MATRIX['experts'])]
            if not seeds:continue
            counts=np.asarray(MATRIX['cells'][cell]['counts']);tail=np.flatnonzero(counts<=50)+1;head=np.flatnonzero(counts>=250)+1
            meta=np.load(result_folder/'metadata.npz');y=meta['labels'].astype(bool);codes=y.astype(int)@(1<<np.arange(6));values={};labels={}
            for seed in seeds:
                assert read(folder/f'matched_triad_{seed}.json')['passed']
                for expert in MATRIX['experts']:
                    key=(expert,seed);path=result_folder/f'{expert}_{seed}.npz';record=read(result_folder/f'{expert}_{seed}.json')
                    assert sha(path)==record['prediction_sha256']
                    assert sha(folder/'models'/f'{expert}_{seed}/model.pth')==record['model_sha256']
                    prediction=np.load(path);z=prediction['logits'];p=expit(z);pred=p>=prediction['thresholds'];exact=np.all(pred==y,1)
                    check(exact.mean(),record['metrics']['f1_exact'],'exact');check(np.mean(np.logaddexp(0,z)-y*z),record['metrics']['BCE'],'BCE')
                    check(np.mean((p-y)**2),record['metrics']['Brier'],'Brier')
                    check(np.all((z>=0)==y,axis=1).mean(),record['metrics']['fixed05_exact'],'fixed05_exact')
                    aps=[];eces=[];f1s=[]
                    for j,label in enumerate(record['labels']):
                        check((pred[:,j]&~y[:,j]).sum(),label['FP'],'FP');check((~pred[:,j]&y[:,j]).sum(),label['FN'],'FN')
                        tp=(pred[:,j]&y[:,j]).sum();fp=(pred[:,j]&~y[:,j]).sum();fn=(~pred[:,j]&y[:,j]).sum()
                        check(fp/max((~y[:,j]).sum(),1),label['FPR'],'FPR');check(tp/max(y[:,j].sum(),1),label['recall'],'recall')
                        f1s.append(2*tp/max(2*tp+fp+fn,1))
                        ap=average_precision(y[:,j],p[:,j]);aps.append(ap);check(ap,label['AP'],'AP')
                        bins=np.minimum(np.floor(p[:,j]*15).astype(int),14)
                        ece=sum(np.mean(bins==k)*abs(float(p[bins==k,j].mean())-float(meta['labels'][bins==k,j].mean())) for k in range(15) if np.any(bins==k))
                        eces.append(ece);check(ece,label['ECE'],'ECE')
                        weak=meta['weak'][:,j]
                        check(weak.sum(),label['weak_n'],'weak_n')
                        if weak.any():check(pred[weak,j].mean(),label['weak_recall'],'weak_recall')
                    check(np.mean(aps),record['metrics']['macro_AP'],'macro_AP');check(np.mean(eces),record['metrics']['ECE'],'ECE');check(np.mean(f1s),record['metrics']['macro_F1'],'macro_F1')
                    values[key]={**record['metrics'],'quota_tail_exact':float(exact[np.isin(codes,tail)].mean()),
                        'quota_head_exact':float(exact[np.isin(codes,head)].mean()),
                        'fit_prior_weighted_exact':float((counts/counts.sum())@np.array([exact[codes==c].mean() for c in range(1,64)]))}
                    labels[key]=record['labels'];all_values[split,cell,expert,seed]=values[key]
            design=read(folder/'objective_design.json');q=np.asarray(design['q']);p=np.asarray(design['p']);target=np.asarray(design['target_p'])
            groups={'quota_tail_codes_le50':tail.tolist(),'quota_head_codes_ge250':head.tolist(),
                    'quota_tail_fit_samples':int(counts[tail-1].sum()),'quota_tail_eval_samples':int(np.isin(codes,tail).sum()),
                    'historical_tail_codes':prior.QUOTAS['tail_codes'],'historical_tail_note':'legacy metric uses fixed historical codes; quota_tail_exact uses current cell counts'}
            view=read(folder/'data/fit_pairing_receipt.json')
            mass=counts/counts.sum();sorted_counts=np.sort(counts);gini=float((2*np.arange(1,64)-64)@sorted_counts/(63*sorted_counts.sum()))
            cells.append({'split':split,'cell':cell,'seeds':seeds,'groups':groups,'fit_n':8000,'epochs':40,'sample_visits_per_model':320000,
                'objective_combo_ESS':float(1/(q@q)),'label_p':p.tolist(),'weighted_label_p':target.tolist(),
                'objective_marginal_ESS_per_label':(8000/(target**2/p+(1-target)**2/(1-p))).tolist(),
                'combination_IF':float(counts.max()/counts.min()),'label_IF':MATRIX['cells'][cell]['label_IF'],
                'combination_gini':gini,'combination_entropy':float(-(mass*np.log(mass)).sum()),'effective_combination_count':float(1/(mass@mass)),
                'paired_view_receipt':view,'means':{e:{k:interval([values[e,s][k] for s in seeds]) for k in values[e,seeds[0]]} for e in MATRIX['experts']}})
            for a,b in [('combo','marginal'),('marginal','natural'),('combo','natural')]:
                label_rows=[]
                for j,label in enumerate(LABELS):
                    label_rows.append({'label':label,**{k:interval([labels[a,s][j][k]-labels[b,s][j][k] for s in seeds]) for k in ['FPR','recall','weak_recall','AP']}})
                row={'split':split,'cell':cell,'candidate':a,'reference':b,'seeds':seeds,
                    'differences':{k:interval([values[a,s][k]-values[b,s][k] for s in seeds]) for k in values[a,seeds[0]]},'labels':label_rows}
                contrasts.append(row)
                print('MATRIX_COMPARE',role,split,cell,a,b,'Exactpp',row['differences']['f1_exact']['mean']*100,'n',len(seeds),flush=True)
    pairs=[('CF16_IF3','CF7p85_IF3','higher_CF_at_IF3'),('CF16_near_IF1','CF7p85_IF1','higher_CF_at_near_IF1'),
           ('CF7p85_IF3','CF7p85_IF1','higher_IF_at_CF7p85'),('CF16_IF3','CF16_near_IF1','higher_IF_at_CF16')]
    for split in CONFIG['splits']:
        for c1,c0,name in pairs:
            seeds=[s for s in MATRIX['seeds'] if all((split,c,e,s) in all_values for c in [c1,c0] for e in MATRIX['experts'])]
            if not seeds:continue
            for a,b in [('combo','marginal'),('marginal','natural'),('combo','natural')]:
                v=[(all_values[split,c1,a,s]['f1_exact']-all_values[split,c1,b,s]['f1_exact'])-(all_values[split,c0,a,s]['f1_exact']-all_values[split,c0,b,s]['f1_exact']) for s in seeds]
                mass1=np.asarray(MATRIX['cells'][c1]['counts'])/8000;mass0=np.asarray(MATRIX['cells'][c0]['counts'])/8000
                cross.append({'split':split,'condition':name,'candidate':a,'reference':b,'seeds':seeds,'difference_of_differences':interval(v),
                              'quota_distribution_total_variation':float(abs(mass1-mass0).sum()/2)})
    result={'role':role,'cells':cells,'comparisons':contrasts,'cross_condition_interactions':cross,'complete_models':sum(len(c['seeds'])*3 for c in cells),
        'verification':{'passed':not errors,'scalars':checks,'errors':errors},
        'boundaries':['single-seed intervals omitted','seed intervals are unadjusted and conditional on each fixed cohort',
                      'quota changes preserve cardinality counts; they also alter combination distribution, so ratio alone is not identified causally',
                      'B matches both positive and negative label marginals of C; B is not necessarily label-balanced',
                      'primary evaluation is uniform over 63 combinations; fit_prior_weighted_exact is a reweighting of the same observations, not a new test',
                      'weak-view schedule is identical within each A/B/C triad; coverage differs across quota cells'],
        'source_sha256':sha(Path(__file__)),'time':stamp()}
    write(ROOT/'analysis'/f'matrix_{role}.json',result);assert not errors
    print('MATRIX_VERIFIED',result['complete_models'],checks,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--role',default='development');a=p.parse_args();analyze(a.role)
