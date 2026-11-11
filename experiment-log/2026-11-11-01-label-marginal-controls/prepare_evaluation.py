"""Audit fresh pools against current and historical sample/component identities."""
import argparse
from common import *

def identity_arrays(path):
    with h5py.File(path,'r') as f:
        noise=np.asarray(f['receiver_noise_seed']).reshape(-1).astype(np.int64)
        samples=(noise-2*2246822519)%(1<<32)
        active=np.asarray(f['label_all'],dtype=bool)
        if 'sample_seed' in f:
            np.testing.assert_array_equal(samples,np.asarray(f['sample_seed']).reshape(-1))
    components={j+1:(samples[active[j]]+2246822519+(j+1)*2654435761)%(1<<32) for j in range(len(active))}
    return samples,noise,components

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--split',required=True,choices=CONFIG['splits'])
    parser.add_argument('--role',choices=['development','confirmation'],default='development')
    args=parser.parse_args()
    data=ROOT/args.split/'data'
    pool=base.candidates(data,'dev_pool' if args.role=='development' else 'confirmation_pool',LABELS)
    samples,noise,components=identity_arrays(pool['path'])
    np.testing.assert_array_equal(samples,pool['seeds'])
    registry=np.array([1,2,3,4,6,8])
    for row,seeds in enumerate(pool['components']):
        ids=registry[np.flatnonzero(pool['y'][row])]
        np.testing.assert_array_equal(seeds,(samples[row]+2246822519+ids*2654435761)%(1<<32))
    sources=set()
    for root in [PREVIOUS.parent.parent,ROOT]:
        for receipt in root.rglob('generation_receipt.json'):
            path=Path(read(receipt).get('path',''))
            if path.is_file() and path.suffix=='.mat': sources.add(path.resolve())
    # Include legacy data whose generator receipts may predate this registry.
    for split in CONFIG['splits']:
        preflight=read(PREVIOUS/split/'development/preflight.json')
        for record in preflight['audit']:
            path=Path(record['path'])
            assert path.is_file(),path
            sources.add(path.resolve())
    audit=[]
    for path in sorted(sources):
        if path==pool['path'].resolve(): continue
        old_s,old_n,old_c=identity_arrays(path)
        overlap=[len(np.intersect1d(samples,old_s)),len(np.intersect1d(noise,old_n))]
        component_overlap=sum(len(np.intersect1d(v,old_c.get(k,[]))) for k,v in components.items())
        assert overlap==[0,0] and component_overlap==0,(str(path),overlap,component_overlap)
        audit.append({'path':str(path),'sample_overlap':0,'noise_overlap':0,'component_overlap':0})
    indices=np.concatenate([np.flatnonzero(pool['codes']==code)[:64] for code in range(1,64)]).tolist()
    assert len(indices)==4032
    manifest={'path':str(pool['path']),'sha256':pool['sha256'],'indices':indices,'role':args.role,'pool_rows':len(samples),'used_rows':4032,'time':stamp()}
    if args.role=='confirmation':
        assert read(ROOT/'final_freeze.json')['all_models_and_calibrations_frozen']
        manifest['freeze_sha256']=sha(ROOT/'final_freeze.json')
    write(ROOT/args.split/args.role/'manifest.json',manifest)
    write(ROOT/args.split/args.role/'preflight.json',{'passed':True,'audit':audit,'actual_new_component_seeds_verified':True,'historical_mapping':'previously audited generator seed mapping','generator_sha256':sha(HERE/'generate_pools.m'),'source_sha256':sha(Path(__file__))})
    print('IDENTITIES_PASSED',args.split,args.role,len(audit),flush=True)

if __name__=='__main__':main()
