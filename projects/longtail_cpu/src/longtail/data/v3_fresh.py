"""Fresh identities under the v3 10--30 dB independent-component distribution.

Reuses the tested Python port of the archived MATLAB physical equations.
PCG64 replaces MATLAB mt19937ar; this is explicitly not bitwise MATLAB parity.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import json
import multiprocessing as mp
from pathlib import Path
import h5py
import numpy as np
from .signals import Simulator,DEFAULT_PHYSICS,REGISTRY_IDS,rng,substream_seed
from ..io import read_json,write_json,fingerprint,sha256

_SIM=None


def initialize(physics):
    global _SIM
    _SIM=Simulator(physics)


def compose_v3(simulator,active_codes,sample_seed,jnr_range=(10.,30.),return_components=False):
    components=[];jnrs=[];seeds=[]
    for code in active_codes:
        seed=substream_seed(sample_seed,1,REGISTRY_IDS[code]);stream=rng(seed)
        # Original MATLAB generators draw JNR before their physical parameters.
        jnr=float(stream.uniform(*jnr_range))
        raw,_=getattr(simulator,'_'+code)(stream,simulator.physics[code])
        power=float(np.mean(np.abs(raw)**2))
        if not np.isfinite(power) or power<=0:raise ValueError('Invalid component power')
        signal=raw/np.sqrt(power)*np.sqrt(simulator.radar['noise_power'])*10**(jnr/20)
        components.append(signal);jnrs.append(jnr);seeds.append(seed)
    noise_seed=substream_seed(sample_seed,2);stream=rng(noise_seed)
    noise=np.sqrt(simulator.radar['noise_power']/2)*(stream.standard_normal(simulator.length)+1j*stream.standard_normal(simulator.length))
    clean=np.stack(components);power=10*np.log10(np.mean(np.abs(clean)**2,axis=1))
    result={'iq':clean.sum(0)+noise,'jnr':np.asarray(jnrs),'power':power,
            'relative':power-power.max(),'component_seeds':seeds,'noise_seed':noise_seed}
    if return_components:
        result.update(clean_components=clean,receiver_noise=noise)
    return result


def chunk_worker(task):
    codes,seeds,classes=task;n=len(codes);length=_SIM.length
    result={'iq_complex':np.empty((length,n),np.complex128),'label_selected':np.zeros((6,n),np.uint8),
            'sample_seed':np.asarray(seeds,dtype=np.uint32)[None,:],
            'receiver_noise_seed':np.empty((1,n),np.uint32)}
    for name in ['component_effective_JNR_dB','component_measured_power_dB','relative_power_dB','component_seeds']:
        result[name]=np.full((6,n),np.nan)
    result['weak_component_mask']=np.zeros((6,n),np.uint8)
    for i,(code,seed) in enumerate(zip(codes,seeds)):
        active=np.flatnonzero((int(code)>>np.arange(6))&1)
        sample=compose_v3(_SIM,[classes[c] for c in active],int(seed))
        result['iq_complex'][:,i]=sample['iq'];result['label_selected'][active,i]=1
        result['receiver_noise_seed'][0,i]=sample['noise_seed']
        for key,field in [('component_effective_JNR_dB','jnr'),('component_measured_power_dB','power'),
                          ('relative_power_dB','relative'),('component_seeds','component_seeds')]:
            result[key][active,i]=sample[field]
        result['weak_component_mask'][active,i]=(len(active)>1)&(sample['relative']<=-10+1e-6)
    return result


def generate(root,reference_manifest,base_seed=3400000000,workers=8):
    from .dataset import read_metadata
    from .manage import verify
    reference=read_json(reference_manifest);root=Path(root).resolve()
    if root.exists():raise FileExistsError('Use a new identity-pool directory')
    classes=reference['classes'];counts={role:reference['files'][role]['combination_counts'] for role in ['train','development']}
    role_seeds={role:base_seed+offset+np.arange(1,sum(counts[role])+1,dtype=np.uint64)
                for role,offset in [('train',0),('development',3000000)]}
    if max(int(v.max()) for v in role_seeds.values())>=2**32:raise ValueError('Seed overflow')
    new_ids=np.concatenate([(v+2*2246822519)%(2**32) for v in role_seeds.values()])
    checked=[];dataset_root=Path(reference_manifest).resolve().parent
    for path in sorted(p for p in dataset_root.rglob('*') if p.suffix in ('.mat','.h5')):
        with h5py.File(path,'r') as f:
            if 'receiver_noise_seed' not in f:continue
            old=np.asarray(f['receiver_noise_seed']).reshape(-1).astype(np.uint64)
        overlap=np.intersect1d(old,new_ids).size
        if overlap:raise ValueError('Fresh identities overlap: '+str(path))
        checked.append(str(path))
    legacy=Path(__file__).resolve().parents[3]/'backup/generation_matlab_20261017'
    legacy_manifest=read_json(legacy/'legacy_sources.json')
    for rec in legacy_manifest['files']:
        assert sha256(legacy/'matlab/legacy'/rec['path'])==rec['sha256']
    physics=deepcopy(DEFAULT_PHYSICS)
    plan={'schema':'v3-independent-power-fresh-identity-python-v1','base_seed':base_seed,
          'classes':classes,'reference_dataset_id':reference['dataset_id'],'reference_manifest_sha256':sha256(reference_manifest),
          'counts':counts,'physics':physics,'independent_component_jnr_db':[10,30],
          'additional_power_offsets_db':0,'rng':'NumPy PCG64','matlab_bitwise_parity':False,
          'provenance':'Archived original v3 physical configuration/equations; existing tested Python physics port; new identity seeds.',
          'legacy_sources':legacy_manifest['files'],'python_physics_sha256':sha256(Path(__file__).with_name('signals.py')),
          'writer_sha256':sha256(Path(__file__)),'existing_identity_files_checked':checked,
          'identity_overlap_with_existing':0,'train_development_identity_overlap':0,
          'train_samples':8000,'development_samples':6048,'workers':workers}
    plan['sha256']=fingerprint(plan);root.mkdir(parents=True);write_json(root/'generation_plan.json',plan)
    records={}
    with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn'),initializer=initialize,initargs=(physics,)) as pool:
        for role in ['train','development']:
            codes=np.repeat(np.arange(1,64),counts[role]);seeds=role_seeds[role];n=len(codes)
            tasks=[(codes[start:start+64],seeds[start:start+64],classes) for start in range(0,n,64)]
            temporary=root/(role+'.partial.h5');final=root/(role+'.h5')
            with h5py.File(temporary,'x') as f:
                f.attrs.update(power_truth_schema='per-label-v1',weak_gap_db=10.,noise_power_db=0.,
                    generator='v3-physics-python-pcg64-v1',generation_plan_sha256=plan['sha256'],classes_json=json.dumps(classes))
                for chunk,block in enumerate(pool.map(chunk_worker,tasks)):
                    start=chunk*64;stop=min(start+64,n)
                    for name,values in block.items():
                        if name not in f:f.create_dataset(name,shape=(values.shape[0],n),dtype=values.dtype,
                            chunks=(values.shape[0],min(64,n)),compression='lzf')
                        f[name][:,start:stop]=values
                    if chunk%16==0 or stop==n:print(json.dumps({'role':role,'generated':stop,'total':n}),flush=True)
            temporary.replace(final);y,identities,length=read_metadata(final)
            assert y.sum(0).tolist()==reference['files'][role]['label_counts'] and length==3000
            records[role]={'path':final.name,'sha256':sha256(final),'bytes':final.stat().st_size,'samples':n,
                           'label_counts':y.sum(0).tolist(),'combination_counts':counts[role]}
    manifest={'schema_version':2,'dataset_id':'ama-sixclass-v3-fresh-python-20261102','classes':classes,
        'signal_length':3000,'files':records,'power_truth_schema':'per-label-v1','generation_plan_sha256':plan['sha256'],
        'reference_dataset':'ama-sixclass-v3','independent_component_jnr_db':[10,30],
        'generator':'v3 physical-equation Python port; PCG64, not MATLAB RNG reproduction',
        'development_role':'Fresh identity pool: calibration/check/report 16/16/64 per combination; reporting deferred until fixed comparison completes'}
    write_json(root/'dataset.json',manifest);verification=verify(root)
    write_json(root/'generation_complete.json',{'status':'completed','verification':verification,
        'generation_plan_sha256':plan['sha256'],'matlab_bitwise_parity':False})
    print(json.dumps({'status':'completed','root':str(root),'verification':verification}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--reference',required=True,type=Path);p.add_argument('--workers',type=int,default=8)
    args=p.parse_args();generate(args.output,args.reference,workers=args.workers)


if __name__=='__main__':main()
