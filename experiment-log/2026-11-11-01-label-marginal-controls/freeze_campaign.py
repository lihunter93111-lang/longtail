"""Freeze both paired cohorts and all comparison models before new confirmation."""
from common import *
from analyze_results import COMPARISONS

if __name__=='__main__':
    target=ROOT/'final_freeze.json'
    assert not target.exists(),'Existing scientific freeze must be preserved'
    assert read(ROOT/'development_verification.json')['passed']
    files={}
    models=[]
    for split in CONFIG['splits']:
        summary=read(ROOT/split/'development_results/summary.json')
        assert summary['seeds']==CONFIG['seeds'] and len(summary['results'])==50
        for path in sorted((ROOT/split/'decision_models').glob('*.json')):
            files[str(path)]=sha(path)
            if path.name=='protocol.json':continue
            state=read(path)
            for key in ['model_path','reference_path']:
                if key in state: files[state[key]]=sha(state[key])
            models.append({'split':split,'name':path.stem,'decision_path':str(path),'decision_sha256':sha(path)})
        for filename in ['manifest.json','preflight.json']:
            path=ROOT/split/'development'/filename;files[str(path)]=sha(path)
    for path in HERE.iterdir():
        if path.suffix in ['.py','.m','.json'] or path.name.endswith('GOAL.txt') or path.name=='GOAL_PROMPT.txt':files[str(path)]=sha(path)
    for directory in [SOURCE/'projects/CJ_Imbalance_0716/cjlt',Path(CONFIG['repository'])/'data_generation/0716/可配置组合干扰数据生成']:
        for path in directory.rglob('*'):
            if path.suffix in ['.py','.m']:files[str(path)]=sha(path)
    write(target,{'all_models_and_calibrations_frozen':True,'models':models,'files':files,'time':stamp(),'primary_comparison':['combo','marginal'],'supporting_comparisons':COMPARISONS,'confirmation_pool_seeds':{s:CONFIG['pool_seeds'][s]['confirmation_pool'] for s in CONFIG['splits']},'selection_role':'development','existing_confirmation_used':False})
    print('FROZEN',len(models),len(files),sha(target),flush=True)
