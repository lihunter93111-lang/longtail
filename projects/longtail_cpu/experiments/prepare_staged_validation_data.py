"""Fresh development pools per stage and group-held-out one-bit pairs."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import multiprocessing as mp
from pathlib import Path
import shutil
import os
import h5py
import numpy as np

from longtail.data.signals import DEFAULT_PHYSICS,substream_seed
from longtail.data.v3_fresh import initialize,chunk_worker
from longtail.data import one_bit
from longtail.data.dataset import read_metadata
from longtail.data.power_truth import read_power_truth
from longtail.io import read_json,write_json,sha256,fingerprint

ROOT=Path(__file__).resolve().parents[1]


def save_blocks(path,tasks,initializer,worker,physics,total,paired=False,workers=8):
    temporary=path.with_suffix('.partial.h5')
    with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn'),initializer=initializer,initargs=(physics,)) as pool,h5py.File(temporary,'x') as f:
        f.attrs.update(power_truth_schema='per-label-v1',weak_gap_db=10.,noise_power_db=0.)
        if paired:f.attrs['identity_schema']='one-bit-pairs-v1'
        offset=0
        for block in pool.map(worker,tasks):
            count=block['iq_complex'].shape[1]
            for name,value in block.items():
                if name not in f:f.create_dataset(name,shape=(value.shape[0],total),dtype=value.dtype,chunks=(value.shape[0],min(64,total)),compression='lzf')
                f[name][:,offset:offset+count]=value
            offset+=count
            if offset%1024==0 or offset==total:print({'file':path.name,'generated':offset,'total':total},flush=True)
        assert offset==total
    temporary.replace(path)


def prepare(output,source_manifest,workers=8):
    if output.exists():raise FileExistsError('Use a new dataset directory')
    source=read_json(source_manifest);classes=source['classes'];physics=deepcopy(DEFAULT_PHYSICS)
    definitions=[];all_seeds=[]
    for stage in range(1,4):
        seeds=3800000000+stage*1000000+np.arange(1,6049,dtype=np.uint64)
        definitions.append((stage,np.repeat(np.arange(1,64),96),seeds));all_seeds.extend(seeds.tolist())
    roles={key:[] for key in ['fit','check','development_test']};seed=3900000001;edge_manifest=[]
    for target in [0,2,4,5]:
        backgrounds=np.array([c for c in range(1,64) if not (c>>target)&1])
        order=np.random.default_rng(6100+target).permutation(backgrounds)
        for index,background in enumerate(order):
            role='fit' if index<19 else 'check' if index<25 else 'development_test';count=64 if role=='development_test' else 32
            edge=target*64+int(background)
            edge_manifest.append({'target':int(target),'background_code':int(background),'edge_id':edge,'role':role,'pairs':count})
            for _ in range(count):roles[role].append((int(background),target,seed,edge));all_seeds.append(seed);seed+=1
    noise_ids=np.array([substream_seed(int(s),2) for s in all_seeds]);assert len(np.unique(noise_ids))==len(noise_ids)
    checked=[]
    for directory in [source_manifest.parent.parent.parent,ROOT/'data',ROOT/'backup']:
        for path in sorted(directory.rglob('*')):
            if path.suffix not in ['.h5','.mat'] or not h5py.is_hdf5(path):continue
            with h5py.File(path) as f:
                if 'receiver_noise_seed' not in f:continue
                existing=f['receiver_noise_seed'][:].ravel()
            assert not np.intersect1d(noise_ids,existing).size,str(path);checked.append(str(path.resolve()))
    output.mkdir(parents=True)
    provenance=[]
    for file in [Path(__file__),ROOT/'src/longtail/data/one_bit.py',ROOT/'src/longtail/data/v3_fresh.py',ROOT/'src/longtail/data/signals.py']:
        dest=output/'generator_source'/file.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(file,dest)
        provenance.append({'path':str(file.relative_to(ROOT)),'sha256':sha256(dest)})
    plan={'schema':'three-stage-validation-data-v1','source_manifest':str(source_manifest),'source_manifest_sha256':sha256(source_manifest),
        'physics':physics,'classes':classes,'fresh_development_per_stage':6048,'stage_base_seeds':[3801000001,3802000001,3803000001],
        'pair_seed_start':3900000001,'pair_edges':edge_manifest,'checked_identity_files':sorted(set(checked)),
        'identity_overlap_existing':0,'generator_sources':provenance,'matlab_bitwise_parity':False,
        'pair_group_policy':'Per target 19 train / 6 check / 6 report edges, disjoint. All sides of a pair share background/noise; IDs differ.',
        'limitations':'Pair-edge holdout is relative to paired training; the pretrained natural backbone has seen all63 label combinations.'}
    plan['sha256']=fingerprint(plan);write_json(output/'generation_plan.json',plan)
    manifests={}
    for stage,codes,seeds in definitions:
        folder=output/f'stage{stage}';folder.mkdir();train=source_manifest.parent/source['files']['train']['path']
        try:os.link(train,folder/'train.h5')
        except OSError:shutil.copyfile(train,folder/'train.h5')
        tasks=[(codes[i:i+64],seeds[i:i+64],classes) for i in range(0,len(seeds),64)]
        save_blocks(folder/'development.h5',tasks,initialize,chunk_worker,physics,6048,workers=workers)
        y,ids,_=read_metadata(folder/'development.h5');read_power_truth(folder/'development.h5',y)
        manifest=deepcopy(source);manifest.update(dataset_id=f'ama-v3-staged-{stage}-20261104',generation_plan_sha256=plan['sha256'])
        manifest['development_role']='All calibration/check/report identities regenerated; reporting is stage-gated.'
        for role in ['train','development']:
            path=folder/(role+'.h5'); yy,_,_=read_metadata(path)
            manifest['files'][role]={'path':path.name,'sha256':sha256(path),'bytes':path.stat().st_size,'samples':len(yy),
                'label_counts':yy.sum(0).astype(int).tolist(),'combination_counts':np.bincount(yy.astype(int)@(1<<np.arange(6)),minlength=64)[1:].tolist()}
        write_json(folder/'dataset.json',manifest);manifests[str(stage)]=str(folder/'dataset.json')
    pair_folder=output/'one_bit';pair_folder.mkdir();pair_files={};audits={}
    for role,rows in roles.items():
        array=np.asarray(rows,dtype=np.int64)
        tasks=[(array[i:i+64,0],array[i:i+64,1],array[i:i+64,2],array[i:i+64,3],classes) for i in range(0,len(rows),64)]
        file=pair_folder/(role+'.h5');save_blocks(file,tasks,one_bit.initialize,one_bit.chunk_worker,physics,2*len(rows),paired=True,workers=workers)
        y,ids,_=read_metadata(file);read_power_truth(file,y)
        with h5py.File(file) as f:
            pair_ids=f['pair_id'][:].ravel();sides=f['pair_side'][:].ravel();targets=f['pair_target'][:].ravel();noise=f['receiver_noise_seed'][:].ravel()
            assert np.array_equal(pair_ids[::2],pair_ids[1::2]) and np.array_equal(noise[::2],noise[1::2])
            assert np.array_equal(sides,np.tile([0,1],len(rows)))
            assert np.all((y[::2]^y[1::2]).sum(1)==1)
            assert np.all(y[np.arange(0,len(y),2),np.asarray(targets[::2],int)]==0)
            assert np.all(y[np.arange(1,len(y),2),np.asarray(targets[::2],int)]==1)
            assert np.nanmax(f['raw_difference_relative_error'][:])<1e-12
            audit={'pairs':len(rows),'samples':len(y),'unique_sample_ids':len(np.unique(ids)),'unique_noise_ids':len(np.unique(noise)),
                'edges':len(np.unique(f['edge_id'][:])),'one_bit_exact':True,'raw_difference_verified':True}
        pair_files[role]={'path':str(file),'sha256':sha256(file),'pairs':len(rows),'samples':len(y)};audits[role]=audit
    write_json(pair_folder/'manifest.json',{'files':pair_files,'edges':edge_manifest,'generation_plan_sha256':plan['sha256'],'audits':audits})
    write_json(output/'generation_complete.json',{'status':'completed','stage_manifests':manifests,'pair_manifest':str(pair_folder/'manifest.json'),
        'old_identity_files_checked':len(set(checked)),'all_new_background_noise_ids_unique':True,'new_backgrounds_overlap_existing':0,'pair_audits':audits})
    print({'completed':str(output),'pairs':audits},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--source',type=Path,required=True);p.add_argument('--workers',type=int,default=8)
    a=p.parse_args();prepare(a.output.resolve(),a.source.resolve(),a.workers)
