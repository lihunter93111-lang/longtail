"""Freeze candidates before generating any new confirmation identity."""
from common import *

def freeze():
    target=ROOT/'final_freeze.json'
    if target.exists():
        for path,expected in read(target)['files'].items():assert sha(path)==expected,path
        print('EXISTING_FREEZE_VERIFIED',flush=True);return
    assert read(ROOT/'input_audit.json')['passed'] and read(ROOT/'development_core_verification.json')['passed']
    portable=read(ROOT/'development_portable_verification.json');assert portable['passed'] and portable['cpu_representative_records']
    assert all(len(set(s['combination_codes']))==63 for s in portable['replay_selection'].values())
    matrix=read(ROOT/'analysis/matrix_development.json');assert matrix['verification']['passed']
    assert len(portable['records'])==200+matrix['complete_models']
    assert len({r['name'] for r in portable['records']})==len(portable['records'])
    assert read(ROOT/'development_physical_verification.json')['passed']
    assert read(ROOT/'analysis/background_diagnostics_development.json')['verification']['passed']
    assert read(ROOT/'budget_audit.json')['new_complete_backbones']==matrix['complete_models']
    assert len(read(ROOT/'runtime_benchmark.json')['records'])==16
    assert read(ROOT/'development_supervision_verification.json')['passed']
    assert read(ROOT/'development_supervision_runtime_verification.json')['decisions']==10
    registry=read(ROOT/'identity_registry/registry.json');assert registry['passed']
    assert registry['source_sha256']==sha(HERE/'audit_identity_registry.py')
    stored_dev=read(ROOT/'development_stored_identity_verification.json')
    assert stored_dev['passed'] and stored_dev['registry_sha256']==sha(ROOT/'identity_registry/registry.json')
    audited_data={r['sha256'] for r in registry['records']}
    contract=read(ROOT/'matrix_contract_verification.json');assert contract['passed'] and contract['models']==matrix['complete_models']
    unity=read(ROOT/'analysis/near_unity_development.json');assert unity['passed'] and unity['source_sha256']==sha(HERE/'diagnose_near_unity.py')
    assert len(unity['records'])==sum(len(c['seeds']) for c in matrix['cells'] if c['cell'] in ['CF7p85_IF1','CF16_near_IF1'])
    paths=set();sources=set()
    for split in CONFIG['splits']:
        runtime=read(ROOT/split/'development_runtime_verification.json');assert runtime['passed'] and runtime['distinct_combinations']==63
        assert runtime['source_sha256']==sha(HERE/'runtime.py')
        assert (ROOT/split/'physical_development_results/summary.json').is_file()
        assert not (ROOT/split/'data/confirmation_pool/generation_receipt.json').exists()
        decisions=sorted((ROOT/split/'decision_models').glob('*_22*.json'));assert len(decisions)==95
        for p in decisions:
            paths.add(p)
            for model in read(p)['models'].values():
                q=Path(model['path']);assert sha(q)==model['sha256'];paths.add(q)
        paths.add(ROOT/split/'decision_models/protocol.json')
        for role in ['fusion_fit','calibration','development']:
            m=role_manifest(split,role);assert sha(m['path'])==m['sha256'];paths.add(Path(m['path']))
            assert m['sha256'] in audited_data
        assert read(BASELINES/split/'data/manifest.json')['fit_sha256'] in audited_data
        paths.add(ROOT/split/'development/manifest.json')
    paths.add(ROOT/'input_audit.json');paths.add(ROOT/'matrix/data_acceptance.json')
    paths.add(ROOT/'analysis/near_unity_development.json')
    paths.update(p for p in (ROOT/'identity_registry').rglob('*') if p.is_file())
    controls=list((ROOT/'supervision_control').glob('*/models/reference_affine_22*.json'));assert len(controls)==10
    paths.update(controls);paths.add(ROOT/'supervision_control/protocol.json')
    planned=read(HERE/'matrix_protocol.json');complete=[];not_completed=[]
    for split in CONFIG['splits']:
        for cell in planned['cells']:
            folder=ROOT/'matrix'/split/cell
            for p in (folder/'data').glob('*'):
                if p.is_file():paths.add(p)
            m=read(folder/'data/manifest.json');paths.update([Path(m['fit_path']),Path(m['dev_path'])])
            assert m['fit_sha256'] in audited_data and m['dev_sha256'] in audited_data
            for seed in planned['seeds']:
                marker=folder/f'matched_triad_{seed}.json'
                if not marker.exists():not_completed.append({'split':split,'cell':cell,'seed':seed});continue
                assert read(marker)['passed'];paths.add(marker);complete.append({'split':split,'cell':cell,'seed':seed})
                for expert in planned['experts']:
                    model=folder/'models'/f'{expert}_{seed}'
                    paths.update(p for p in model.iterdir() if p.is_file())
                    for p in (folder/'training'/expert/f'M_{seed}').glob('*.json'):paths.add(p)
                    paths.add(folder/'training'/expert/'protocol.json')
    assert len(complete)*3==matrix['complete_models']
    for folder in [HERE,REPO/'experiment-log/2026-11-11-01-label-marginal-controls',REPO/'experiment-log/2026-11-10-01-readout-integration',
                   SOURCE/'experiment-log/2026-11-06-05-weak-component-goal',SOURCE/'experiment-log/2026-11-06-01-lt-abc',
                   SOURCE/'projects/CJ_Imbalance_0716/cjlt',REPO/'data_generation/0716/可配置组合干扰数据生成']:
        for pattern in ['*.py','*.m','*.ps1','protocol.json']:
            sources.update(folder.rglob(pattern))
    sources.update([HERE/'matrix_protocol.json',HERE/'GOAL_PROMPT.txt',HERE/'03A_PHYSICAL_PAIR_GOAL.txt',HERE/'04A_MATRIX_EXECUTION_GOAL.txt'])
    sources.add(Path(CONFIG['generator_config']))
    sources.update(HERE.glob('*GOAL.txt'));sources.update(HERE.glob('*DECISION.txt'))
    paths.update(sources)
    for p in (ROOT/'runtime_bundle').rglob('*'):
        if p.is_file() and '__pycache__' not in p.parts:paths.add(p)
    files={str(p):sha(p) for p in sorted(paths)}
    write(target,{'all_models_and_calibrations_frozen':True,'files':files,'core_decisions':190,'supervision_control_decisions':10,'matrix_completed_triads':complete,
        'matrix_not_completed_triads':not_completed,'confirmation_pool_seeds':CONFIG['pool_seeds'],
        'primary_comparisons':CONFIG['primary_comparisons'],'physical_comparisons':'fixed evaluate_physical.py contrasts',
        'matrix_comparisons':'C-B, B-A, C-A in all completed matched quota cells; interactions exploratory',
        'development_sources':[str(ROOT/'analysis/matrix_development.json')]+[str(ROOT/'analysis'/f'{s}_development.json') for s in CONFIG['splits']],
        'time':stamp()})
    print('FINAL_FROZEN',len(files),'core',190,'supervision_controls',10,'matrix',len(complete)*3,flush=True)

if __name__=='__main__':freeze()
