"""Fixed-prediction background slices; no model fitting or selection."""
import argparse,gzip
from common import *
from scipy.special import expit
from scipy.stats import t

PAIRS=[('combo_segment','combo_none'),('combo_segment','combo_global'),
       ('marginal_segment','marginal_none'),('marginal_segment','marginal_global'),
       ('combo_segment','marginal_segment'),('combo_none','marginal_none'),
       ('combo_segment_guarded','combo_none'),('marginal_segment_guarded','marginal_none'),
       ('natural_segment','natural_none')]


def summary(values):
    v=np.asarray(values,float);radius=t.ppf(.975,len(v)-1)*v.std(ddof=1)/np.sqrt(len(v))
    return {'mean':float(v.mean()),'values':v.tolist(),'ci95':[float(v.mean()-radius),float(v.mean()+radius)]}


def paired(a,b,mask):
    n=int(mask.sum())
    if not n:return {'n':0,'difference':None,'corrected':None,'new_errors':None}
    a=a[:,mask];b=b[:,mask]
    corrected=(a&~b).sum(1);new=(~a&b).sum(1);net=a.sum(1)-b.sum(1)
    np.testing.assert_array_equal(corrected-new,net)
    return {'n':n,'difference':summary(net/n),'corrected':summary(corrected),'new_errors':summary(new)}


def diagnose(role):
    splits=[];checks=0
    for split in CONFIG['splits']:
        folder=ROOT/split/(role+'_results');report=read(folder/'summary.json');seeds=report['seeds']
        meta=np.load(folder/'metadata.npz');y=meta['labels'].astype(bool);weak=meta['weak'].astype(bool)
        codes=y.astype(int)@(1<<np.arange(6));assert set(codes)==set(range(1,64)) and np.all(~weak|y)
        fit=read(BASELINES/split/'data/manifest.json');assert sha(fit['fit_path'])==fit['fit_sha256']
        with h5py.File(fit['fit_path'],'r') as f:train=np.asarray(f['known_label'],dtype=np.int8).T[np.asarray(fit['fit_A'])]
        counts=np.bincount(train@(1<<np.arange(6)),minlength=64);assert counts[0]==0 and counts.sum()==8000
        by={(r['method'],r['seed']):r for r in report['results']};predictions={};sources={}
        for method in sorted({m for pair in PAIRS for m in pair}):
            values=[]
            for seed in seeds:
                path=folder/f'{method}_{seed}.npz';record=by[method,seed]
                assert sha(path)==record['prediction_sha256'];p=np.load(path);pred=expit(p['logits'])>=p['thresholds']
                assert pred.shape==y.shape
                assert abs(np.all(pred==y,1).mean()-record['metrics']['f1_exact'])<1e-12;checks+=1
                for j,lr in enumerate(record['labels']):
                    assert int((pred[:,j]&~y[:,j]).sum())==lr['FP']
                    assert int((~pred[:,j]&y[:,j]).sum())==lr['FN'];checks+=2
                values.append(pred);sources[str(path)]=sha(path)
            predictions[method]=np.asarray(values)
        masks={'tail':counts[codes]<=50,'middle':(counts[codes]>50)&(counts[codes]<250),'head':counts[codes]>=250}
        assert np.all(sum(m.astype(int) for m in masks.values())==1)
        comparisons=[]
        for candidate,reference in PAIRS:
            a=predictions[candidate]==y[None];b=predictions[reference]==y[None]
            exact_a=a.all(2);exact_b=b.all(2)
            groups={name:paired(exact_a,exact_b,mask) for name,mask in masks.items()}
            combinations=[{'code':code,'train_n':int(counts[code]),**paired(exact_a,exact_b,codes==code)} for code in range(1,64)]
            assert sum(row['n'] for row in combinations)==len(y)
            backgrounds=[]
            for j,label in enumerate(LABELS):
                bit=1<<j;background_codes=codes&~bit
                for background in range(64):
                    if background&bit:continue
                    mask=background_codes==background;positive=mask&y[:,j];negative=mask&~y[:,j]
                    recall=paired(a[:,:,j],b[:,:,j],positive)
                    true_negative=paired(a[:,:,j],b[:,:,j],negative)
                    fpr=None if not negative.any() else summary(-np.asarray(true_negative['difference']['values']))
                    backgrounds.append({'label':label,'background_code':background,
                        'background_labels':[LABELS[k] for k in range(6) if background&(1<<k)],
                        'train_positive_n':int(counts[background|bit]),'train_negative_n':int(counts[background]),
                        'recall':recall,'weak_recall':paired(a[:,:,j],b[:,:,j],positive&weak[:,j]),
                        'negative_n':int(negative.sum()),'FPR_difference':fpr,'true_negative_accuracy':true_negative})
            comparisons.append({'candidate':candidate,'reference':reference,'groups':groups,'combinations':combinations,'backgrounds':backgrounds})
        splits.append({'split':split,'seeds':seeds,'identities':len(y),'training_counts_by_code_0_to63':counts.tolist(),
            'fit_manifest_sha256':sha(BASELINES/split/'data/manifest.json'),'prediction_sources':sources,'comparisons':comparisons})
    result={'role':role,'splits':splits,'verification':{'passed':True,'source_metric_checks':checks},
        'boundaries':['Exploratory fixed groups, no candidate or threshold selected.',
            'Seed t intervals are unadjusted and conditional on a fixed evaluation cohort.',
            'Background groups contain different identities; this is not a paired physical intervention.',
            'Zero-count target/background cells remain missing; no observed zero metric is imputed.',
            'Selected readout stages target SawFM/MFT/ISRJ; all six labels are retained to expose other decision costs.'],
        'source_sha256':sha(Path(__file__)),'goal_sha256':sha(HERE/'03C_BACKGROUND_DIAGNOSTIC_GOAL.txt'),'time':stamp()}
    detail_path=ROOT/'analysis'/f'background_diagnostics_{role}_details.json.gz'
    detail_path.write_bytes(gzip.compress(json.dumps(result,ensure_ascii=False,allow_nan=False,separators=(',',':')).encode('utf-8'),mtime=0))
    compact={**result,'splits':[{**split,'comparisons':[{k:v for k,v in row.items() if k not in ['combinations','backgrounds']} for row in split['comparisons']]} for split in splits],
        'details':{'path':str(detail_path),'sha256':sha(detail_path),'encoding':'gzip compressed UTF-8 JSON',
                   'combination_rows':sum(len(row['combinations']) for split in splits for row in split['comparisons']),
                   'background_rows':sum(len(row['backgrounds']) for split in splits for row in split['comparisons'])}}
    assert json.loads(gzip.decompress(detail_path.read_bytes()))==result
    write(ROOT/'analysis'/f'background_diagnostics_{role}.json',compact)
    for split in splits:
        row=split['comparisons'][0]
        print('BACKGROUND_DIAGNOSIS',role,split['split'],row['candidate'],row['reference'],
              {name:100*v['difference']['mean'] if v['n'] else None for name,v in row['groups'].items()},flush=True)
    print('BACKGROUND_VERIFIED',checks,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--role',default='development');args=parser.parse_args();diagnose(args.role)
