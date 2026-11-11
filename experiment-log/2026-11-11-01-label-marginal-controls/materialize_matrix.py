"""Materialize four future fit quota manifests; no claim of trained evidence."""
from common import *
from prepare_evaluation import identity_arrays

def main():
    design_path=HERE.parent/'2026-11-10-01-readout-integration/results/analysis/future_dual_tail_matrix.json'
    design=read(design_path)
    pool=base.candidates(ROOT/'split1/data','fit_pool',LABELS)
    samples,noise,components=identity_arrays(pool['path'])
    audit=[]
    for item in read(ROOT/'split1/development/preflight.json')['audit']:
        path=Path(item['path'])
        if path.resolve()==pool['path'].resolve():continue
        s,n,c=identity_arrays(path)
        overlaps=[len(np.intersect1d(samples,s)),len(np.intersect1d(noise,n)),sum(len(np.intersect1d(v,c.get(k,[]))) for k,v in components.items())]
        assert overlaps==[0,0,0],(path,overlaps)
        audit.append({'path':str(path),'overlap':overlaps})
    labels=((np.arange(1,64)[:,None]>>np.arange(6))&1).astype(int)
    results={};masks={}
    for name,cell in design['cells'].items():
        counts=np.asarray(cell['counts'],int)
        assert len(counts)==63 and counts.min()>0 and counts.sum()==8000
        selected=[]
        for code,count in enumerate(counts,1):
            candidates=np.flatnonzero(pool['codes']==code)
            assert len(candidates)>=count
            selected.extend(candidates[:count].tolist())
        actual=np.bincount(pool['codes'][selected],minlength=64)[1:]
        np.testing.assert_array_equal(actual,counts)
        margins=labels.T@counts
        np.testing.assert_array_equal(margins,cell['label_margins'])
        cardinality=[int(counts[labels.sum(1)==k].sum()) for k in range(1,7)]
        record={'role':'future_fit_only_not_trained','path':str(pool['path']),'sha256':pool['sha256'],'indices':selected,
            'counts':counts.tolist(),'label_margins':margins.tolist(),'label_IF':float(margins.max()/margins.min()),
            'combination_IF':float(counts.max()/counts.min()),'cardinality_counts':cardinality,'design_sha256':sha(design_path)}
        write(ROOT/'future_matrix'/name/'manifest.json',record)
        results[name]={k:v for k,v in record.items() if k!='indices'}
        masks[name]=set(selected)
    assert len({tuple(r['cardinality_counts']) for r in results.values()})==1
    write(ROOT/'future_matrix/acceptance.json',{'passed':True,'new_training':False,'candidate_rows':len(pool['y']),
        'cells':results,'pool_identity_audit':audit,'pairwise_shared_identities':{a+'__'+b:len(masks[a]&masks[b]) for a in masks for b in masks if a<b},
        'source_sha256':sha(Path(__file__)),'time':stamp()})
    print('FOUR_QUOTA_MANIFESTS_ACCEPTED',flush=True)

if __name__=='__main__':main()
