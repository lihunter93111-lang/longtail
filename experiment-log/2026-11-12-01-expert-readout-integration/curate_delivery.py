"""Copy reviewable evidence to Git; large data/models remain outside Git."""
import argparse,shutil,zipfile
from common import *

def curate():
    assert read(ROOT/'confirmation_core_verification.json')['passed']
    assert read(ROOT/'confirmation_physical_verification.json')['passed']
    assert read(ROOT/'confirmation_portable_verification.json')['passed']
    matrix=read(ROOT/'analysis/matrix_confirmation.json');assert matrix['verification']['passed']
    sources=set()
    for name in ['input_audit.json','final_freeze.json','environment.json','runtime_benchmark.json','development_core_verification.json',
                 'confirmation_core_verification.json','development_physical_verification.json','confirmation_physical_verification.json',
                 'development_portable_verification.json','confirmation_portable_verification.json']:
        sources.add(ROOT/name)
    sources.add(ROOT/'operating_points_verification.json')
    sources.add(ROOT/'budget_audit.json')
    sources.add(ROOT/'matrix_contract_verification.json')
    for role in ['development','confirmation']:
        sources.add(ROOT/(role+'_supervision_verification.json'))
        sources.add(ROOT/(role+'_supervision_runtime_verification.json'))
    sources.update((ROOT/'supervision_control').rglob('*.json'))
    sources.add(ROOT/'supervision_precision_repair_receipt.json')
    sources.add(ROOT/'confirmation_stored_identity_verification.json')
    sources.add(ROOT/'development_stored_identity_verification.json')
    sources.update((ROOT/'identity_registry').rglob('*.json'))
    for folder in ['analysis','future_minimal_quota']:
        sources.update((ROOT/folder).glob('*.json'))
    sources.update((ROOT/'analysis').glob('*.json.gz'))
    for split in CONFIG['splits']:
        for role in ['development','confirmation']:
            sources.update((ROOT/split/role).glob('*.json'))
            sources.update((ROOT/split/(role+'_results')).glob('*.json'))
            sources.add(ROOT/split/(role+'_runtime_verification.json'))
            sources.add(ROOT/split/('physical_'+role)/'generation_receipt.json')
            sources.add(ROOT/split/('physical_'+role+'_results')/'summary.json')
        sources.update((ROOT/split/'decision_models').glob('*.json'))
    sources.add(ROOT/'matrix/data_acceptance.json')
    sources.update((ROOT/'matrix').glob('*/*/objective_design.json'))
    for p in (ROOT/'matrix').glob('*/*/data/*.json'):sources.add(p)
    for p in (ROOT/'matrix').glob('*/*/matched_triad_*.json'):sources.add(p)
    for p in (ROOT/'matrix').glob('*/*/models/*/*.json'):sources.add(p)
    for p in (ROOT/'matrix').glob('*/*/training/*/protocol.json'):sources.add(p)
    for p in (ROOT/'matrix').glob('*/*/training/*/M_*/*.json'):sources.add(p)
    for role in ['development','confirmation']:
        sources.update((ROOT/'matrix').glob(f'*/*/{role}_results/*.json'))
        sources.update((ROOT/'matrix').glob(f'*/*/fixed_fpr_{role}_results/*.json'))
    sources.update((ROOT/'matrix/queue').glob('*.json'))
    sources.update(ROOT.glob('*queue*.json'))
    sources.update([ROOT/'runtime_bundle/index.json',ROOT/'runtime_bundle/manifest.json'])
    records=[]
    for p in sorted(sources):
        assert p.is_file(),p;dest=HERE/'results'/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
        assert sha(dest)==sha(p);records.append({'path':dest.relative_to(HERE).as_posix(),'source':str(p),'sha256':sha(dest),'bytes':dest.stat().st_size})
    external={}
    for p in ROOT.rglob('*'):
        if p.is_file() and p.suffix.lower() in ['.mat','.pth','.npz','.npy']:
            external[str(p)]={'bytes':p.stat().st_size,'sha256':sha(p)}
    write(HERE/'external_artifacts.json',{'storage':str(ROOT),'files':external,'not_in_git':True,'time':stamp()})
    archive=ROOT/'longtail_integration_runtime_20261112.zip'
    replay_files={};bundle_records=read(ROOT/'runtime_bundle/manifest.json')['records']
    for role in ['development','confirmation']:
        fixture=ROOT/('runtime_replay_'+role);assert len(read(fixture/'index.json'))==bundle_records
        for p in sorted(fixture.iterdir()):
            if p.is_file():replay_files['replay/'+role+'/'+p.name]={'path':p,'sha256':sha(p)}
    replay_hashes={name:record['sha256'] for name,record in replay_files.items()}
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=4) as z:
        for p in sorted((ROOT/'runtime_bundle').rglob('*')):
            if p.is_file() and '__pycache__' not in p.parts:z.write(p,p.relative_to(ROOT/'runtime_bundle').as_posix())
        for name,record in replay_files.items():z.write(record['path'],name)
        z.writestr('REPLAY_FILES_SHA256.json',json.dumps(replay_hashes,indent=2)+'\n')
        z.writestr('REPLAY_GUIDE.txt','迁移包复放入口\n解压后在本目录执行。环境版本参考environment_reference.json。\nGPU全记录及CPU代表复放：\npython -I verify_portable.py --bundle . --fixtures replay/development --output development_replay_check.json\npython -I verify_portable.py --bundle . --fixtures replay/confirmation --output confirmation_replay_check.json\n上述验证器需要CUDA，并包含CPU代表检查；每划分64个身份覆盖全部63组合。\n仅CPU普通推理示例：\npython portable_runtime.py --record records/supervision/split1/reference_affine_2200.json --iq-npy replay/development/split1_raw.npy --output cpu_predictions.npz --device cpu\nreplay/包含复放输入及冻结预期logit，数据属于已消费开发/确认子集，只用于检查迁移一致性，不用于训练或选模型。\nREPLAY_FILES_SHA256.json记录复放文件哈希；外部runtime_package.json记录整个ZIP哈希。完整训练池不在本包中。\n')
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert all(hashlib.sha256(z.read(name)).hexdigest()==digest for name,digest in replay_hashes.items())
    write(HERE/'runtime_package.json',{'path':str(archive),'bytes':archive.stat().st_size,'sha256':sha(archive),'records':bundle_records,
        'replay_fixture_hashes':replay_hashes,'not_in_git':True,'portable_verified':True,'time':stamp()})
    (HERE/'.gitignore').write_text('!figures/*.pdf\n',encoding='utf-8')
    paths=[p for p in HERE.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name not in ['delivery_manifest.json','completion_audit.json']]
    write(HERE/'delivery_manifest.json',{'files':{p.relative_to(HERE).as_posix():sha(p) for p in sorted(paths)},'curated_sources':records,
        'data_and_weights_external':True,'matrix_complete_models':matrix['complete_models'],'supervision_control_decisions':10,'time':stamp()})
    print('CURATED',len(records),'evidence files;',len(external),'external artifacts;',archive.stat().st_size,'archive bytes',flush=True)

if __name__=='__main__':curate()
