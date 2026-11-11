"""Matched single-expert and shared-reference fusion through ordinary I/Q inputs."""
import argparse
from common import *
from cjlt.engine.selective_fusion import fit_affine_fusion, apply_fusion

def checkpoint(split,group,seed):
    if group=='reference':
        return BASELINES/split/'w32'/f'M_{seed-100}'/'best_BCE.pth'
    if group=='marginal':
        return ROOT/split/'models'/f'marginal_{seed}'/'model.pth'
    old={'natural':'ordinary_expert','combo':'weighted_expert'}[group]
    return PREVIOUS/split/'expert_training'/old/f'M_{seed}'/'best_BCE.pth'

def load_model(split,group,seed):
    path=checkpoint(split,group,seed)
    model=base.build_backbone('dfms_iq',num_classes=6,width=32,dropout=.2).cuda()
    state=torch.load(path,map_location='cuda',weights_only=False)
    model.load_state_dict(state['model_state_dict'])
    return model.eval().requires_grad_(False)

def cache_scores(split,group,seed,role,path,indices,inputs):
    model_path=checkpoint(split,group,seed)
    binding={'model_sha256':sha(model_path),'data_sha256':sha(path),'indices_sha256':hashlib.sha256(np.asarray(indices,dtype='<i8').tobytes()).hexdigest()}
    output=ROOT/split/'scores'/f'{group}_{seed}'/(role+'.npz')
    receipt=output.with_suffix('.json')
    if output.exists() and receipt.exists():
        rec=read(receipt)
        assert rec['binding']==binding and rec['sha256']==sha(output)
        return np.load(output)['logits']
    model=load_model(split,group,seed)
    z=base.predict(model,inputs,'cuda').astype(np.float64)
    assert np.isfinite(z).all()
    replay=base.predict(model,inputs[:64],'cuda').astype(np.float64)
    np.testing.assert_allclose(z[:64],replay,atol=1e-6,rtol=1e-6)
    output.parent.mkdir(parents=True,exist_ok=True)
    np.savez(output,logits=z)
    write(receipt,{'binding':binding,'sha256':sha(output),'normal_forward_replay':True,'time':stamp()})
    del model
    return z

def fit(split,seeds):
    m=read(BASELINES/split/'data/manifest.json')
    ff=read(PREVIOUS/split/'fusion_fit/manifest.json')
    assert sha(ff['path'])==ff['sha256']
    fit_x,fit_y=base.load(ff['path'],ff['indices'])
    cal_x,cal_y=base.load(m['dev_path'],m['calibration'])
    out=ROOT/split/'decision_models'
    out.mkdir(exist_ok=True)
    protocol={'fusion_fit_manifest_sha256':sha(PREVIOUS/split/'fusion_fit/manifest.json'),'calibration_manifest_sha256':sha(BASELINES/split/'data/manifest.json'),'fit_identities':4032,'threshold_rule':'per-label calibration F1 grid .05:.005:.95, first maximum','fusion_epochs':300,'fusion_lr':.04,'source_sha256':{str(p):sha(p) for p in [HERE/'evaluate_campaign.py',HERE/'common.py',SOURCE/'projects/CJ_Imbalance_0716/cjlt/engine/selective_fusion.py']}}
    pp=out/'protocol.json'
    if pp.exists(): assert read(pp)==protocol
    else: write(pp,protocol)
    for seed in seeds:
        cal={};train={}
        for group in ['reference','natural','marginal','combo']:
            cal[group]=cache_scores(split,group,seed,'calibration',m['dev_path'],m['calibration'],cal_x)
            train[group]=cache_scores(split,group,seed,'fusion_fit',ff['path'],ff['indices'],fit_x)
            thresholds=helper.fit_thresholds(cal[group],cal_y)
            write(out/f'{group}_{seed}.json',{'kind':'single','group':group,'seed':seed,'thresholds':thresholds,'model_path':str(checkpoint(split,group,seed)),'model_sha256':sha(checkpoint(split,group,seed)),'protocol_sha256':sha(pp)})
        for group in ['natural','marginal','combo']:
            for kind in ['mean','affine']:
                if kind=='mean': state={'kind':'affine','targets':LABELS,'weights':[.5,.5]*6,'biases':[0]*6}
                else: state=fit_affine_fusion(train['reference'],train[group],fit_y,seed=seed-100,epochs=300,learning_rate=.04)
                z,_=apply_fusion(cal['reference'],cal[group],state)
                thresholds=helper.fit_thresholds(z,cal_y)
                record={'kind':'fusion','group':group,'seed':seed,'state':state,'thresholds':thresholds,'model_path':str(checkpoint(split,group,seed)),'model_sha256':sha(checkpoint(split,group,seed)),'reference_path':str(checkpoint(split,'reference',seed)),'reference_sha256':sha(checkpoint(split,'reference',seed)),'protocol_sha256':sha(pp)}
                write(out/f'{group}_{kind}_{seed}.json',record)
        print('DECISIONS_FIT',split,seed,flush=True)

def score(split,seeds,role):
    if role=='confirmation':
        frozen=read(ROOT/'final_freeze.json')
        assert frozen['all_models_and_calibrations_frozen']
        for path,digest in frozen['files'].items(): assert sha(path)==digest,(path,'frozen artifact changed')
    m=read(ROOT/split/role/'manifest.json')
    assert sha(m['path'])==m['sha256']
    x,y=base.load(m['path'],m['indices'])
    meta=metadata(m['path'],m['indices'])
    np.testing.assert_array_equal(y,meta['labels'])
    output=ROOT/split/(role+'_results')
    output.mkdir(exist_ok=True)
    np.savez(output/'metadata.npz',**meta)
    rows=[]
    for seed in seeds:
        scores={g:cache_scores(split,g,seed,role,m['path'],m['indices'],x) for g in ['reference','natural','marginal','combo']}
        for path in sorted((ROOT/split/'decision_models').glob(f'*_{seed}.json')):
            state=read(path)
            assert sha(state['model_path'])==state['model_sha256']
            z=scores[state['group']]
            if state['kind']=='fusion':
                assert sha(state['reference_path'])==state['reference_sha256']
                z,_=apply_fusion(scores['reference'],z,state['state'])
            result=evaluate(z,meta,state['thresholds'])
            name=path.stem
            np.savez(output/(name+'.npz'),logits=z,thresholds=state['thresholds'])
            record={'method':name.rsplit('_',1)[0],'seed':seed,**result,'decision_sha256':sha(path),'prediction_sha256':sha(output/(name+'.npz'))}
            write(output/(name+'.json'),record)
            rows.append(record)
        print('SCORED',split,role,seed,flush=True)
    write(output/'summary.json',{'role':role,'split':split,'seeds':seeds,'identities':len(y),'results':rows,'time':stamp()})

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['fit','score'])
    parser.add_argument('--split',required=True,choices=CONFIG['splits'])
    parser.add_argument('--seeds',nargs='+',type=int,default=CONFIG['seeds'])
    parser.add_argument('--role',choices=['development','confirmation'],default='development')
    args=parser.parse_args(); setup()
    if args.action=='fit':fit(args.split,args.seeds)
    else:score(args.split,args.seeds,args.role)

if __name__=='__main__':main()
