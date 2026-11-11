"""Requirement-level scientific and delivery checks, not just a file manifest."""
from common import *
import zipfile

def main():
    checks=[]
    def check(name,condition,evidence):
        assert condition,(name,evidence)
        checks.append({'requirement':name,'passed':True,'evidence':evidence})
    frozen=read(ROOT/'final_freeze.json')
    changed=[p for p,d in frozen['files'].items() if sha(p)!=d]
    check('Frozen models, thresholds and scientific sources unchanged',not changed,{'files':len(frozen['files']),'models':len(frozen['models']),'changed':changed})
    check('Positive/negative marginal loss and gradient controls',read(ROOT/'weight_selfcheck.json')['passed'],read(ROOT/'weight_selfcheck.json'))
    trained=[]
    for split in CONFIG['splits']:
        for seed in CONFIG['seeds']:
            current=ROOT/split/'training/marginal'/f'M_{seed}'
            rec=read(current/'receipt.json');history=read(current/'history.json')
            assert rec['epochs']==40 and rec['updates']==10000 and len(history)==40
            assert sha(current/'best_BCE.pth')==rec['checkpoint_sha256']
            best=torch.load(current/'best_BCE.pth',map_location='cpu',weights_only=False)['model_state_dict']
            exported=torch.load(ROOT/split/'models'/f'marginal_{seed}/model.pth',map_location='cpu',weights_only=False)['model_state_dict']
            assert best.keys()==exported.keys() and all(torch.equal(best[k],exported[k]) for k in best)
            for group in ['ordinary_expert','weighted_expert']:
                folder=PREVIOUS/split/'expert_training'/group/f'M_{seed}'
                old=read(folder/'receipt.json');old_history=read(folder/'history.json')
                assert rec['initial_state_sha256']==old['initial_state_sha256'] and rec['order_sha256']==old['order_sha256']
                assert sha(folder/'best_BCE.pth')==old['checkpoint_sha256']
                assert [(r['augmented_presentations'],r['gap15_presentations']) for r in history]==[(r['augmented_presentations'],r['gap15_presentations']) for r in old_history]
            trained.append({'split':split,'seed':seed,'epochs':40,'updates':10000,'best_epoch':rec['best_BCE_epoch'],'seconds':rec['elapsed_seconds'],'state_equals_export':True})
        for role in ['development','confirmation']:
            summary=read(ROOT/split/(role+'_results/summary.json'))
            assert len(summary['results'])==50 and summary['identities']==4032
            assert read(ROOT/split/role/'preflight.json')['passed']
            replay=read(ROOT/split/(role+'_runtime_replay.json'))
            assert replay['passed'] and len(replay['records'])==50 and max(x['maximum_error'] for x in replay['records'])<=1e-6
            for row in summary['results']:
                assert sha(ROOT/split/'decision_models'/f"{row['method']}_{row['seed']}.json")==row['decision_sha256']
        creation=(ROOT/split/'data/confirmation_pool/generation_config.json').stat().st_ctime
        assert creation>=datetime.fromisoformat(frozen['time']).timestamp()
        check(split+' new confirmation after freeze and roles audited',True,{'samples':4032,'frozen_at':frozen['time'],'config_created_utc':datetime.fromtimestamp(creation,timezone.utc).isoformat(),'identity_audit':str(ROOT/split/'confirmation/preflight.json')})
    check('Ten complete B trainings matched to both A and C',len(trained)==10,trained)
    for role in ['development','confirmation']:
        result=read(ROOT/(role+'_verification.json'))
        check(role+' independent metric recomputation',result['passed'] and result['scalars_recomputed']==5300,result)
    for name in ['portable_inference_verification.json','portable_relocated_verification.json']:
        check(name,read(ROOT/name)['passed'],{'path':str(ROOT/name),'sha256':sha(ROOT/name)})
    matrix=read(ROOT/'future_matrix/acceptance.json')
    check('Four actual future quota manifests without claiming trained performance',matrix['passed'] and len(matrix['cells'])==4 and not matrix['new_training'],{'cells':list(matrix['cells']),'candidate_rows':matrix['candidate_rows']})
    for name in matrix['cells']:
        manifest=read(ROOT/'future_matrix'/name/'manifest.json')
        assert len(manifest['indices'])==8000 and len(set(manifest['indices']))==8000 and min(manifest['counts'])>0
    retry=read(ROOT/'future_matrix/exact_retry.json')
    check('Bounded exact-margin exploration preserved including failure status','feasible_found' in retry,retry)
    check('Development-driven diagnostic completed',read(ROOT/'analysis/background_development.json')['no_selection'],{'path':str(ROOT/'analysis/background_development.json')})
    cost=read(ROOT/'analysis/runtime_cost.json')
    check('Actual inference cost measured',len(cost['records'])==18,{'hardware':cost['device'],'records':len(cost['records'])})
    bundle=read(ROOT/'handoff_bundle_receipt.json')
    assert sha(bundle['archive'])==bundle['sha256']
    with zipfile.ZipFile(bundle['archive']) as z:assert z.testzip() is None
    for row in read(ROOT/'handoff_bundle/bundle_manifest.json')['files']:
        assert sha(ROOT/'handoff_bundle'/row['path'])==row['sha256']
    check('Collaborator replay package verified',bundle['models']==40 and bundle['decisions']==100 and bundle['includes_confirmation_data'],bundle)
    documents=['GOAL_PROMPT.txt','00_E_DRIVE_AMENDMENT_GOAL.txt','01_AUDIT_DATA_GOAL.txt','02_MARGINAL_CONTROLS_GOAL.txt','03_DUAL_TAIL_MATRIX_GOAL.txt',
        '04_SUPERVISION_BUDGET_GOAL.txt','05_GENERALIZATION_GOAL.txt','06_CONFIRM_DELIVER_GOAL.txt','07_BACKGROUND_DIAGNOSTIC_GOAL.txt','08_EXACT_MATRIX_GOAL.txt',
        'RESULTS.txt','LONG_TERM_PLAN.txt','NEXT_ROUND_PLAN.txt','COLLABORATOR_GUIDE.txt','REPRODUCING.txt']
    check('Long-term plan, stage Goals, results, limitations and handoff guide exist',all((HERE/f).is_file() and (HERE/f).stat().st_size>200 for f in documents),documents)
    paths=[]
    for original,digest in frozen['files'].items():
        p=Path(original)
        if p.is_relative_to(SOURCE):
            target=Path(CONFIG['repository'])/p.relative_to(SOURCE)
            assert sha(target)==digest
            paths.append(str(target))
    check('Frozen model library matches portable repository source',len(paths)>0,{'mapped_source_files':len(paths)})
    write(ROOT/'completion_audit.json',{'passed':True,'scope':'Tonight executed stages; future matrix training, physical readout and budget studies remain explicitly planned',
        'checks':checks,'time':stamp(),'source_sha256':sha(Path(__file__))})
    print('COMPLETION_AUDIT_PASSED',len(checks),flush=True)

if __name__=='__main__':main()
