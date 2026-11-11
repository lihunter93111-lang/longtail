"""Direct stored-component verification before fresh confirmation scoring."""
import argparse
from common import *

REGISTRY=ROOT/'identity_registry'


def sources():
    paths=set()
    for root in [PREVIOUS.parent.parent,MARGINAL,ROOT]:
        for receipt in root.rglob('generation_receipt.json'):
            path=Path(read(receipt).get('path',''))
            if path.is_file() and path.suffix=='.mat':paths.add(path.resolve())
    for split in CONFIG['splits']:
        for row in read(MARGINAL/split/'confirmation/preflight.json')['audit']:paths.add(Path(row['path']).resolve())
    return sorted(paths)


def inspect_pool(path):
    path=Path(path).resolve();digest=sha(path);source_sha=sha(Path(__file__))
    folder=REGISTRY/'pools';folder.mkdir(parents=True,exist_ok=True)
    output=folder/(digest+'.npz');receipt=folder/(digest+'.json')
    if output.exists() and receipt.exists():
        saved=read(receipt)
        if saved['source_sha256']==source_sha and saved['index_sha256']==sha(output):return saved
    started=time.perf_counter()
    with h5py.File(path,'r') as f:
        noise=np.asarray(f['receiver_noise_seed']).reshape(-1).astype(np.int64)
        sample=(noise-2*2246822519)%(1<<32);active=np.asarray(f['label_all'],dtype=bool)
        assert active.shape[1]==len(sample) and len(set(sample))==len(sample)==len(set(noise))
        if 'sample_seed' in f:np.testing.assert_array_equal(sample,np.asarray(f['sample_seed']).reshape(-1))
        references=f['component_seeds'][0,:];component_count=0
        assert len(references)==len(sample)
        for i,reference in enumerate(references):
            values=np.asarray(f[reference]).reshape(-1);assert np.isfinite(values).all()
            actual=values.astype(np.int64);np.testing.assert_array_equal(values,actual)
            ids=np.flatnonzero(active[:,i])+1
            expected=(sample[i]+2246822519+ids*2654435761)%(1<<32)
            np.testing.assert_array_equal(np.sort(actual),np.sort(expected));component_count+=len(actual)
    arrays={'sample':np.sort(sample),'noise':np.sort(noise)}
    for j in range(len(active)):
        arrays['component_'+str(j+1)]=np.sort((sample[active[j]]+2246822519+(j+1)*2654435761)%(1<<32))
    np.savez_compressed(output,**arrays)
    result={'path':str(path),'sha256':digest,'identities':len(sample),'stored_components_checked':component_count,
            'stored_component_multisets_verified':True,'sample_and_noise_unique':True,
            'physical_type_counts':{str(j+1):int(active[j].sum()) for j in range(len(active))},
            'index_path':str(output),'index_sha256':sha(output),'source_sha256':source_sha,
            'elapsed_seconds':time.perf_counter()-started,'time':stamp()}
    write(receipt,result);print('STORED_SEED_POOL_VERIFIED',len(sample),component_count,str(path),flush=True)
    return result


def build():
    assert not (ROOT/'final_freeze.json').exists()
    for split in CONFIG['splits']:assert not (ROOT/split/'data/confirmation_pool/generation_receipt.json').exists()
    records=[inspect_pool(path) for path in sources()]
    result={'passed':True,'records':records,'pool_count':len(records),'identity_rows_scanned':sum(r['identities'] for r in records),
            'stored_components_checked':sum(r['stored_components_checked'] for r in records),
            'scope':'Direct metadata audit, not waveform equality; historical pools may intentionally reuse identities.',
            'source_sha256':sha(Path(__file__)),'time':stamp()}
    write(REGISTRY/'registry.json',result);print('IDENTITY_REGISTRY_BUILT',len(records),result['stored_components_checked'],flush=True)


def overlap(left,right):
    result={'sample':int(np.intersect1d(left['sample'],right['sample']).size),'noise':int(np.intersect1d(left['noise'],right['noise']).size),
            'components_by_physical_type':{key:int(np.intersect1d(left[key],right[key]).size) for key in left.files if key.startswith('component_') and key in right.files}}
    assert result['sample']==result['noise']==0 and not any(result['components_by_physical_type'].values()),result
    return result


def confirm():
    frozen=read(ROOT/'final_freeze.json');registry=read(REGISTRY/'registry.json')
    assert registry['passed'] and registry['source_sha256']==sha(Path(__file__))
    assert frozen['files'][str(REGISTRY/'registry.json')]==sha(REGISTRY/'registry.json')
    old=[]
    for record in registry['records']:
        assert sha(record['path'])==record['sha256'] and sha(record['index_path'])==record['index_sha256']
        old.append((record,np.load(record['index_path'])))
    fresh={};rows=[]
    for split in CONFIG['splits']:
        manifest=role_manifest(split,'confirmation');record=inspect_pool(manifest['path']);assert record['sha256']==manifest['sha256']
        assert record['identities']==4032;fresh[split]=(record,np.load(record['index_path']))
        for previous,indices in old:rows.append({'split':split,'reference':previous['path'],**overlap(fresh[split][1],indices)})
    split_names=list(CONFIG['splits']);between=overlap(fresh[split_names[0]][1],fresh[split_names[1]][1])
    result={'passed':True,'registry_sha256':sha(REGISTRY/'registry.json'),'new_pools':{s:v[0] for s,v in fresh.items()},
            'checks':rows,'between_confirmation_splits':between,'old_pool_count':len(old),
            'component_scope':'same physical type, actual stored component metadata verified before indexing',
            'source_sha256':sha(Path(__file__)),'time':stamp()}
    write(ROOT/'confirmation_stored_identity_verification.json',result)
    print('CONFIRMATION_STORED_IDENTITIES_VERIFIED',len(rows),'plus_cross_split',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['build','confirm']);args=parser.parse_args();globals()[args.action]()
