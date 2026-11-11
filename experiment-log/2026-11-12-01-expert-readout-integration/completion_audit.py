"""End-to-end completion gate, separate from eventual remote publication."""
from common import *
import zipfile

def audit():
    failures=[];checks={}
    def check(name,condition):
        checks[name]=bool(condition)
        if not condition:failures.append(name)
    freeze=read(ROOT/'final_freeze.json')
    check('all_frozen_files_unchanged',all(sha(path)==expected for path,expected in freeze['files'].items()))
    check('core_190_decisions',freeze['core_decisions']==190)
    check('supervision_10_decisions',freeze['supervision_control_decisions']==10)
    registry=read(ROOT/'identity_registry/registry.json');stored=read(ROOT/'confirmation_stored_identity_verification.json')
    check('stored_component_registry',registry['passed'] and all(r['stored_component_multisets_verified'] for r in registry['records']))
    stored_dev=read(ROOT/'development_stored_identity_verification.json')
    check('development_stored_seed_isolation',stored_dev['passed'] and stored_dev['registry_sha256']==sha(ROOT/'identity_registry/registry.json') and len(stored_dev['checks'])==2*(registry['pool_count']-2))
    check('confirmation_stored_seed_isolation',stored['passed'] and stored['registry_sha256']==sha(ROOT/'identity_registry/registry.json') and len(stored['checks'])==2*registry['pool_count'])
    check('operating_points_arithmetic',read(ROOT/'operating_points_verification.json')['passed'])
    contract=read(ROOT/'matrix_contract_verification.json')
    check('matrix_training_contract',contract['passed'] and contract['models']==3*len(freeze['matrix_completed_triads']))
    unity=read(ROOT/'analysis/near_unity_development.json')
    check('near_unity_development_diagnostic',unity['passed'] and len(unity['records'])==sum(t['cell'] in ['CF7p85_IF1','CF16_near_IF1'] for t in freeze['matrix_completed_triads']) and unity['source_sha256']==sha(HERE/'diagnose_near_unity.py'))
    for role in ['development','confirmation']:
        background=read(ROOT/'analysis'/f'background_diagnostics_{role}.json')
        check(role+'_background_diagnostics',background['verification']['passed'] and len(background['splits'])==2)
        check(role+'_background_details',sha(background['details']['path'])==background['details']['sha256'])
        for kind in ['core','physical','portable']:
            check(role+'_'+kind,read(ROOT/(role+'_'+kind+'_verification.json'))['passed'])
        portable=read(ROOT/(role+'_portable_verification.json'))
        check(role+'_portable_complete',len(portable['records'])==200+3*len(freeze['matrix_completed_triads']) and len({r['name'] for r in portable['records']})==len(portable['records']))
        check(role+'_portable_CPU_and_combinations',len(portable['cpu_representative_records'])==7 and all(len(set(s['combination_codes']))==63 for s in portable['replay_selection'].values()))
        supervision=read(ROOT/(role+'_supervision_verification.json'));runtime_control=read(ROOT/(role+'_supervision_runtime_verification.json'))
        check(role+'_supervision_control',supervision['passed'] and supervision['decisions']==10 and runtime_control['passed'] and runtime_control['decisions']==10 and runtime_control['distinct_combinations']==63)
        matrix=read(ROOT/'analysis'/f'matrix_{role}.json');check(role+'_matrix_arithmetic',matrix['verification']['passed'])
        check(role+'_matrix_matches_frozen_count',matrix['complete_models']==3*len(freeze['matrix_completed_triads']))
        check(role+'_operating_points_complete',len(read(ROOT/'analysis'/f'matrix_operating_points_{role}.json')['records'])==matrix['complete_models'])
        for split in CONFIG['splits']:
            check(split+'_'+role+'_identity_acceptance',read(ROOT/split/role/'preflight.json')['passed'])
            check(split+'_'+role+'_normal_runtime',read(ROOT/split/(role+'_runtime_verification.json'))['passed'])
            check(split+'_'+role+'_runtime_all_combinations',read(ROOT/split/(role+'_runtime_verification.json'))['distinct_combinations']==63)
            check(split+'_'+role+'_190_half',len(read(ROOT/split/(role+'_results')/'summary.json')['results'])==95)
    manifest=read(HERE/'delivery_manifest.json');check('delivery_hashes',all(sha(HERE/path)==expected for path,expected in manifest['files'].items()))
    package=read(HERE/'runtime_package.json');check('package_hash',sha(package['path'])==package['sha256'])
    with zipfile.ZipFile(package['path']) as archive:
        check('package_replay_hashes',all(hashlib.sha256(archive.read(name)).hexdigest()==digest for name,digest in package['replay_fixture_hashes'].items()))
        contents=set(archive.namelist());complete=True
        for role in ['development','confirmation']:
            index=json.loads(archive.read('replay/'+role+'/index.json'))
            complete=complete and len(index)==200+3*len(freeze['matrix_completed_triads'])
            complete=complete and all(row['record'] in contents and 'replay/'+role+'/'+row['expected'] in contents for row in index)
        check('package_replay_complete',complete and 'REPLAY_GUIDE.txt' in contents)
    check('stage_goals_present',all((HERE/name).exists() for name in ['GOAL_PROMPT.txt','LONG_TERM_PLAN.txt','01_AUDIT_GOAL.txt','01A_STORED_SEED_AUDIT_GOAL.txt','02_FACTORIAL_GOAL.txt',
        '03_ERROR_PHYSICS_GOAL.txt','04_MATRIX_GOAL.txt','05_BUDGET_GOAL.txt','06_CONFIRM_DELIVER_GOAL.txt','07_GENERALIZATION_GOAL.txt',
        '03A_PHYSICAL_PAIR_GOAL.txt','03B_DEVELOPMENT_DECISION.txt','03C_BACKGROUND_DIAGNOSTIC_GOAL.txt','03D_COMBINATION_COST_GOAL.txt','04A_MATRIX_EXECUTION_GOAL.txt','04B_QUOTA_DIAGNOSTIC_GOAL.txt',
        '04C_OPERATING_POINT_GOAL.txt','04D_MATRIX_REPLICATION_GOAL.txt','04E_EVALUATION_PRIOR_GOAL.txt','04F_REPLICATION_DECISION.txt','04G_NEAR_UNITY_DIAGNOSTIC_GOAL.txt','05A_SUPERVISION_CONTROL_GOAL.txt','COLLABORATOR_HANDOFF.txt','PATHS_AND_REPRODUCING.txt']))
    check('confirmation_after_freeze',all(datetime.fromisoformat(read(ROOT/split/'confirmation/manifest.json')['time'])>=datetime.fromisoformat(freeze['time']) for split in CONFIG['splits']))
    result={'passed':not failures,'checks':checks,'failures':failures,'matrix_complete_triads':len(freeze['matrix_completed_triads']),
        'matrix_not_completed_triads':freeze['matrix_not_completed_triads'],'remote_publication':'verify separately using publication_receipt.json',
        'time':stamp()}
    write(HERE/'completion_audit.json',result);write(ROOT/'completion_audit.json',result)
    print('COMPLETION_AUDIT',not failures,len(checks),failures,flush=True);assert not failures

if __name__=='__main__':audit()
