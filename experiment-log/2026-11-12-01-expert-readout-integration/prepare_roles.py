"""Verify new sample, noise and physical component identity domains before use."""
import argparse
from common import *

def identities(path):
    with h5py.File(path,'r') as f:
        noise=np.asarray(f['receiver_noise_seed']).reshape(-1).astype(np.int64)
        sample=(noise-2*2246822519)%(1<<32)
        active=np.asarray(f['label_all'],dtype=bool)
        if 'sample_seed' in f:np.testing.assert_array_equal(sample,np.asarray(f['sample_seed']).reshape(-1))
    components={j+1:(sample[active[j]]+2246822519+(j+1)*2654435761)%(1<<32) for j in range(len(active))}
    return sample,noise,components

def prepare(split,role):
    pool=base.candidates(ROOT/split/'data',role+'_pool',LABELS)
    sample,noise,components=identities(pool['path'])
    assert len(set(sample))==len(sample)==4032 and len(set(noise))==4032
    np.testing.assert_array_equal(sample,pool['seeds']);registry=np.array([1,2,3,4,6,8])
    for i,seeds in enumerate(pool['components']):
        ids=registry[np.flatnonzero(pool['y'][i])]
        np.testing.assert_array_equal(seeds,(sample[i]+2246822519+ids*2654435761)%(1<<32))
    sources=set()
    for root in [PREVIOUS.parent.parent,MARGINAL,ROOT]:
        for receipt in root.rglob('generation_receipt.json'):
            path=Path(read(receipt).get('path',''))
            if path.is_file() and path.suffix=='.mat':sources.add(path.resolve())
    for s in CONFIG['splits']:
        for item in read(MARGINAL/s/'confirmation/preflight.json')['audit']:
            path=Path(item['path']);assert path.is_file();sources.add(path.resolve())
    audit=[]
    for path in sorted(sources):
        if path==pool['path'].resolve():continue
        ss,nn,cc=identities(path)
        overlaps=[len(np.intersect1d(sample,ss)),len(np.intersect1d(noise,nn)),sum(len(np.intersect1d(v,cc.get(k,[]))) for k,v in components.items())]
        assert overlaps==[0,0,0],(str(path),overlaps)
        audit.append({'path':str(path),'sample_overlap':0,'noise_overlap':0,'component_overlap':0})
    indices=np.concatenate([np.flatnonzero(pool['codes']==c) for c in range(1,64)]).tolist()
    assert len(indices)==4032 and all(np.sum(pool['codes']==c)==64 for c in range(1,64))
    manifest={'path':str(pool['path']),'sha256':pool['sha256'],'indices':indices,'used_rows':4032,'role':role,'time':stamp()}
    if role=='confirmation':manifest['freeze_sha256']=sha(ROOT/'final_freeze.json')
    write(ROOT/split/role/'manifest.json',manifest)
    write(ROOT/split/role/'preflight.json',{'passed':True,'audit':audit,'actual_component_seeds_verified':True,'source_sha256':sha(Path(__file__)),'time':stamp()})
    print('ROLE_ACCEPTED',split,role,len(audit),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--split',required=True);p.add_argument('--role',choices=['development','confirmation'],default='development');a=p.parse_args();prepare(a.split,a.role)
