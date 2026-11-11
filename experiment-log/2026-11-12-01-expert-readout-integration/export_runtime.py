"""Deduplicated, relocatable source/weights/decisions with fresh-process replay."""
import argparse,shutil,subprocess,os
from common import *

def export_bundle():
    bundle=ROOT/'runtime_bundle';bundle.mkdir(exist_ok=True)
    for p in (REPO/'projects/CJ_Imbalance_0716/cjlt').rglob('*.py'):
        relative=p.relative_to(REPO/'projects/CJ_Imbalance_0716');dest=bundle/'source'/relative;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
    shutil.copyfile(HERE/'portable_runtime.py',bundle/'portable_runtime.py')
    shutil.copyfile(HERE/'verify_portable.py',bundle/'verify_portable.py')
    shutil.copyfile(ROOT/'environment.json',bundle/'environment_reference.json')
    index={};source_binding={}
    def add_record(name,record,original):
        for key,ref in record['models'].items():
            source=Path(ref['path']);assert sha(source)==ref['sha256'];dest=bundle/'weights'/(ref['sha256']+'.pth');dest.parent.mkdir(exist_ok=True)
            if not dest.exists():shutil.copyfile(source,dest)
            assert sha(dest)==ref['sha256'];source_binding[str(source)]=ref['sha256'];ref['path']=dest.relative_to(bundle).as_posix()
        dest=bundle/'records'/(name+'.json');write(dest,record);index[name]={'record':dest.relative_to(bundle).as_posix(),'source':str(original),'record_sha256':sha(dest)}
    for split in CONFIG['splits']:
        for p in sorted((ROOT/split/'decision_models').glob('*_22*.json')):add_record(split+'/'+p.stem,read(p),p)
        controls=sorted((ROOT/'supervision_control'/split/'models').glob('reference_affine_22*.json'));assert len(controls)==5
        for p in controls:add_record('supervision/'+split+'/'+p.stem,read(p),p)
    matrix=read(HERE/'matrix_protocol.json')
    for split in CONFIG['splits']:
        for cell in matrix['cells']:
            folder=ROOT/'matrix'/split/cell
            for seed in matrix['seeds']:
                if not (folder/f'matched_triad_{seed}.json').exists():continue
                for expert in matrix['experts']:
                    model=folder/'models'/f'{expert}_{seed}/model.pth';cal=read(model.parent/'calibration.json')
                    record={'kind':'single','group':'reference','seed':seed,'thresholds':cal['thresholds'],'models':{'reference':{'path':str(model),'sha256':sha(model)}}}
                    name='matrix/'+split+'/'+cell+'/'+f'{expert}_{seed}';add_record(name,record,model)
                    op=model.parent/'fixed_fpr_operating_point.json'
                    if op.exists():
                        dest=bundle/'diagnostics'/name/'fixed_fpr_operating_point.json';dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(op,dest)
                        assert sha(dest)==sha(op);index[name]['fixed_fpr_diagnostic']=dest.relative_to(bundle).as_posix()
    write(bundle/'index.json',{'labels':LABELS,'input':'raw [N,2,3000] float32; normalize_iq uses per-identity complex RMS','models':index,'source_models':source_binding,'time':stamp()})
    (bundle/'requirements.txt').write_text('numpy\ntorch\nscikit-learn\n',encoding='utf-8')
    (bundle/'README.txt').write_text('可迁移推理包\n读取顺序：index.json → records/ → portable_runtime.py。\nsource/ 为运行所需源码；weights/ 按SHA256去重；records/ 保存冻结融合参数、阈值和权重路径。\n不依赖原机器 E/D 盘路径。index 的 source 字段仅追溯来源，不参与加载。\n示例：python portable_runtime.py --record records/split1/combo_segment_2200.json --iq-npy raw_iq.npy --output predictions.npz --device cuda\n输入：未经RMS归一化的实数数组[N,2,3000]，第0/1通道为I/Q。输出为6标签logit、概率及冻结阈值判定。\n标签顺序：FMB, AMA, SawFM, SinFM, MFT, ISRJ。\nCheckpoint使用PyTorch可信文件加载；仅加载本包且通过SHA256校验的权重。\n',encoding='utf-8')
    hashes={p.relative_to(bundle).as_posix():sha(p) for p in bundle.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='manifest.json'}
    write(bundle/'manifest.json',{'files':hashes,'records':len(index),'unique_models':len(source_binding),'time':stamp()})
    print('BUNDLE_EXPORTED',len(index),len(source_binding),flush=True)

def verify_bundle(role):
    bundle=ROOT/'runtime_bundle';index=read(bundle/'index.json');fixtures=ROOT/('runtime_replay_'+role);fixtures.mkdir(exist_ok=True);rows=[];selection={}
    for split in CONFIG['splits']:
        m=role_manifest(split,role);positions=np.linspace(0,len(m['indices'])-1,64,dtype=int);indices=np.asarray(m['indices'])[positions]
        x,y=base.load(m['path'],indices);np.save(fixtures/(split+'_normalized.npy'),x.numpy())
        codes=y.astype(int)@(1<<np.arange(6));assert len(np.unique(codes))==63;selection[split]={'positions':positions.tolist(),'combination_codes':codes.tolist()}
        with h5py.File(m['path'],'r') as f:
            order=np.argsort(indices);inverse=np.argsort(order)
            raw=np.asarray(f['iq_complex'][:,indices[order]]);raw=np.stack([raw['real'].T,raw['imag'].T],1)[inverse]
        np.save(fixtures/(split+'_raw.npy'),raw)
    for name,entry in index['models'].items():
        parts=name.split('/');split=parts[1] if parts[0] in ['matrix','supervision'] else parts[0]
        if parts[0]=='matrix':expected=ROOT/'matrix'/split/parts[2]/(role+'_results')/(parts[3]+'.npz')
        elif parts[0]=='supervision':expected=ROOT/'supervision_control'/split/(role+'_results')/(parts[2]+'.npz')
        else:expected=ROOT/split/(role+'_results')/(parts[1]+'.npz')
        filename=name.replace('/','__')+'.npy';np.save(fixtures/filename,np.load(expected)['logits'][selection[split]['positions']])
        rows.append({'name':name,'record':entry['record'],'split':split,'expected':filename})
    write(fixtures/'index.json',rows)
    write(fixtures/'selection.json',selection)
    env=os.environ.copy();env.pop('PYTHONPATH',None)
    subprocess.run([sys.executable,'-I',str(bundle/'verify_portable.py'),'--bundle',str(bundle),'--fixtures',str(fixtures),
                    '--output',str(ROOT/(role+'_portable_verification.json'))],cwd=bundle,env=env,check=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--verify',action='store_true');p.add_argument('--role',default='development');a=p.parse_args()
    if a.verify:setup();verify_bundle(a.role)
    else:export_bundle()
