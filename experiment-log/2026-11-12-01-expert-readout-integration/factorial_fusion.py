"""Matched expert by readout factorial; only calibration determines thresholds."""
import argparse
from common import *
from scipy.special import expit
from cjlt.engine.selective_fusion import _fit

class ThreeScoreAffine(torch.nn.Module):
    def __init__(self):
        super().__init__();self.linear=torch.nn.Linear(3,1)
        with torch.no_grad():self.linear.weight.copy_(torch.tensor([[1.,0.,0.]]));self.linear.bias.zero_()
    def forward(self,x):
        reference=x[:,:1];return self.linear(x),torch.ones_like(reference),reference

def cache_scores(split,kind,seed,role,manifest,inputs):
    path=checkpoint(split,kind,seed)
    binding={'model_sha256':sha(path),'data_sha256':manifest['sha256'],
        'indices_sha256':hashlib.sha256(np.asarray(manifest['indices'],dtype='<i8').tobytes()).hexdigest()}
    target=ROOT/split/'scores'/f'{kind}_{seed}'/(role+'.npz');receipt=target.with_suffix('.json')
    if target.exists() and receipt.exists():
        rec=read(receipt);assert rec['binding']==binding and rec['sha256']==sha(target)
        return np.load(target)['logits']
    model=load_model(split,kind,seed)
    z=base.predict(model,inputs,'cuda').astype(np.float64)
    replay=base.predict(model,inputs[:64],'cuda').astype(np.float64)
    np.testing.assert_allclose(z[:64],replay,rtol=1e-6,atol=1e-6)
    target.parent.mkdir(parents=True,exist_ok=True);np.savez(target,logits=z)
    write(receipt,{'binding':binding,'sha256':sha(target),'time':stamp(),'replay_max_error':float(abs(z[:64]-replay).max())})
    del model
    return z

def inputs_for(scores,expert,readout):
    delta=np.zeros_like(scores['reference']) if readout=='none' else scores[readout]-scores['reference']
    return np.stack([scores['reference'],scores[expert],delta],axis=-1)

def apply_record(scores,record):
    if record['kind']=='single':return scores[record['group']]
    return (inputs_for(scores,record['expert'],record['readout'])*np.asarray(record['weights'])).sum(-1)+np.asarray(record['biases'])

def threshold_guard(candidate,reference,meta,reference_thresholds):
    y=meta['labels'].astype(bool);weak=meta['weak'];p=expit(candidate);r=expit(reference)>=reference_thresholds
    thresholds=[];fallback=[];checks=[]
    grid=np.linspace(.05,.95,181)
    for j,label in enumerate(LABELS):
        neg=~y[:,j];mask=weak[:,j]
        fpr=float(r[neg,j].mean());recall=float(r[mask,j].mean()) if mask.any() else None
        options=[]
        for threshold in grid:
            pred=p[:,j]>=threshold;fp=float(pred[neg].mean());wr=float(pred[mask].mean()) if mask.any() else None
            if fp>fpr+CONFIG['guard_fpr_slack']+1e-12:continue
            if recall is not None and wr<recall-CONFIG['guard_weak_recall_slack']-1e-12:continue
            tp=np.sum(pred&y[:,j]);f1=2*tp/max(2*tp+np.sum(pred&neg)+np.sum(~pred&y[:,j]),1)
            options.append((float(f1),float(threshold)))
        if options:
            thresholds.append(max(options,key=lambda x:x[0])[1]);fallback.append(False)
        else:thresholds.append(float(reference_thresholds[j]));fallback.append(True)
        checks.append({'label':label,'reference_fpr':fpr,'reference_weak_recall':recall,'weak_n':int(mask.sum()),'feasible_thresholds':len(options)})
    return thresholds,fallback,checks

def fit(split,seeds):
    assert read(ROOT/'input_audit.json')['passed']
    manifests={r:role_manifest(split,r) for r in ['fusion_fit','calibration']}
    data={}
    for role,m in manifests.items():
        assert sha(m['path'])==m['sha256'];data[role]=base.load(m['path'],m['indices'])
    cal_meta=metadata(manifests['calibration']['path'],manifests['calibration']['indices'])
    folder=ROOT/split/'decision_models';folder.mkdir(parents=True,exist_ok=True)
    files=[HERE/'factorial_fusion.py',HERE/'common.py',HERE/'protocol.json',Path(legacy.__file__),Path(prior.__file__),SOURCE/'projects/CJ_Imbalance_0716/cjlt/engine/selective_fusion.py']
    protocol={'source_sha256':{str(p):sha(p) for p in files},'inputs':manifests,'epochs':CONFIG['fusion_epochs'],'learning_rate':CONFIG['fusion_learning_rate'],
        'weight_decay':CONFIG['fusion_weight_decay'],'thresholds':'per-label calibration F1, 181 point grid; first maximum',
        'guard':'all six labels; FPR +.005, weak recall -.01; whole-column fallback to corresponding expert-none'}
    pp=folder/'protocol.json'
    if pp.exists():assert read(pp)==protocol
    else:write(pp,protocol)
    for seed in seeds:
        scores={role:{kind:cache_scores(split,kind,seed,role,manifests[role],data[role][0]) for kind in ['reference','natural','marginal','combo','global','segment']} for role in manifests}
        states={}
        for kind in ['reference','natural','marginal','combo']:
            record={'kind':'single','group':kind,'seed':seed,'thresholds':helper.fit_thresholds(scores['calibration'][kind],data['calibration'][1]),
                'models':{kind:{'path':str(checkpoint(split,kind,seed)),'sha256':sha(checkpoint(split,kind,seed))}},'protocol_sha256':sha(pp)}
            write(folder/f'{kind}_{seed}.json',record)
        for expert in CONFIG['experts']:
            for readout in CONFIG['readouts']:
                train=inputs_for(scores['fusion_fit'],expert,readout);weights=[];biases=[]
                for j in range(6):
                    model=_fit(ThreeScoreAffine(),train[:,j],data['fusion_fit'][1][:,j:j+1],seed+j,CONFIG['fusion_epochs'],CONFIG['fusion_learning_rate'])
                    weights.append(model.linear.weight.detach().numpy()[0].tolist());biases.append(float(model.linear.bias.detach()[0]))
                record={'kind':'factorial','seed':seed,'expert':expert,'readout':readout,'weights':weights,'biases':biases,
                    'models':{k:{'path':str(checkpoint(split,k,seed)),'sha256':sha(checkpoint(split,k,seed))} for k in ['reference',expert]+([] if readout=='none' else [readout])},'protocol_sha256':sha(pp)}
                cal=apply_record(scores['calibration'],record);record['thresholds']=helper.fit_thresholds(cal,data['calibration'][1])
                name=expert+'_'+readout;states[name]=record;write(folder/f'{name}_{seed}.json',record)
            ref=states[expert+'_none'];ref_z=apply_record(scores['calibration'],ref)
            for readout in ['global','segment']:
                name=expert+'_'+readout;record=json.loads(json.dumps(states[name]));z=apply_record(scores['calibration'],record)
                thresholds,fallback,checks=threshold_guard(z,ref_z,cal_meta,ref['thresholds'])
                for j,flag in enumerate(fallback):
                    if flag:record['weights'][j]=ref['weights'][j].copy();record['weights'][j][2]=0.;record['biases'][j]=ref['biases'][j]
                record.update(thresholds=thresholds,guard={'fallback':fallback,'reference':expert+'_none','calibration':checks})
                final=expit(apply_record(scores['calibration'],record))>=thresholds;reference=expit(ref_z)>=ref['thresholds']
                for j in range(6):
                    neg=cal_meta['labels'][:,j]==0;mask=cal_meta['weak'][:,j]
                    assert final[neg,j].mean()<=reference[neg,j].mean()+CONFIG['guard_fpr_slack']+1e-12
                    if mask.any():assert final[mask,j].mean()>=reference[mask,j].mean()-CONFIG['guard_weak_recall_slack']-1e-12
                write(folder/f'{name}_guarded_{seed}.json',record)
        print('FACTORIAL_FIT',split,seed,flush=True)

def score(split,seeds,role):
    if role=='confirmation':
        f=read(ROOT/'final_freeze.json')
        for path,digest in f['files'].items():assert sha(path)==digest,path
    m=role_manifest(split,role);assert sha(m['path'])==m['sha256']
    x,y=base.load(m['path'],m['indices']);meta=metadata(m['path'],m['indices']);np.testing.assert_array_equal(y,meta['labels'])
    out=ROOT/split/(role+'_results');out.mkdir(parents=True,exist_ok=True);np.savez(out/'metadata.npz',**meta)
    rows=[]
    for seed in seeds:
        scores={k:cache_scores(split,k,seed,role,m,x) for k in ['reference','natural','marginal','combo','global','segment']}
        for path in sorted((ROOT/split/'decision_models').glob(f'*_{seed}.json')):
            rec=read(path)
            for model in rec['models'].values():assert sha(model['path'])==model['sha256']
            z=apply_record(scores,rec);result=evaluate(z,meta,rec['thresholds'])
            np.savez(out/(path.stem+'.npz'),logits=z,thresholds=rec['thresholds'])
            row={'method':path.stem.rsplit('_',1)[0],'seed':seed,**result,'decision_sha256':sha(path),'prediction_sha256':sha(out/(path.stem+'.npz'))}
            write(out/(path.stem+'.json'),row);rows.append(row)
        print('FACTORIAL_SCORED',split,role,seed,flush=True)
    assert len(rows)==19*len(seeds)
    write(out/'summary.json',{'split':split,'role':role,'seeds':seeds,'identities':len(y),'results':rows,'time':stamp()})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['fit','score']);p.add_argument('--split',required=True,choices=CONFIG['splits']);p.add_argument('--role',default='development',choices=['development','confirmation']);p.add_argument('--seeds',nargs='+',type=int,default=CONFIG['seeds']);a=p.parse_args();setup()
    if a.action=='fit':fit(a.split,a.seeds)
    else:score(a.split,a.seeds,a.role)
