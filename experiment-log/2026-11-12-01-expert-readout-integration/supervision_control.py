"""Single-backbone access to the same extra supervision; no backbone updates."""
import argparse
from common import *
from scipy.special import expit
from scipy.stats import t
from factorial_fusion import ThreeScoreAffine
from cjlt.engine.selective_fusion import _fit
from verify_core import average_precision

STORE=ROOT/'supervision_control'


def model_path(split,seed):return STORE/split/'models'/f'reference_affine_{seed}.json'


def cache(split,seed,role):
    manifest=role_manifest(split,role);path=ROOT/split/'scores'/f'reference_{seed}'/(role+'.npz')
    receipt=read(path.with_suffix('.json'))
    expected={'model_sha256':sha(checkpoint(split,'reference',seed)),'data_sha256':manifest['sha256'],
              'indices_sha256':hashlib.sha256(np.asarray(manifest['indices'],dtype='<i8').tobytes()).hexdigest()}
    assert receipt['binding']==expected and receipt['sha256']==sha(path)
    return np.load(path)['logits'],{'path':str(path),'sha256':sha(path),'binding':expected}


def apply(z,record):return z*np.asarray(record['scales'])+np.asarray(record['biases'])


def interval(values):
    v=np.asarray(values,float);m=float(v.mean());radius=float(t.ppf(.975,len(v)-1)*v.std(ddof=1)/np.sqrt(len(v)))
    return {'mean':m,'values':v.tolist(),'ci95':[m-radius,m+radius]}


def fit():
    assert not (ROOT/'final_freeze.json').exists()
    torch.set_num_threads(4);STORE.mkdir(exist_ok=True)
    sources=[Path(__file__),HERE/'05A_SUPERVISION_CONTROL_GOAL.txt',HERE/'common.py',HERE/'factorial_fusion.py',
             SOURCE/'projects/CJ_Imbalance_0716/cjlt/engine/selective_fusion.py',Path(helper.__file__)]
    protocol={'sources':{str(p):sha(p) for p in sources},'epochs_per_label':CONFIG['fusion_epochs'],
              'learning_rate':CONFIG['fusion_learning_rate'],'weight_decay':1e-4,'active_parameters':12,
              'allocated_trainable_parameters':24,'backbones':1,'seeds':CONFIG['seeds'],
              'input':'[reference_logit,0,0]','threshold':'181-point calibration F1, first maximum'}
    assert CONFIG['fusion_weight_decay']==1e-4
    pp=STORE/'protocol.json'
    if pp.exists():assert read(pp)==protocol
    else:write(pp,protocol)
    for split in CONFIG['splits']:
        manifests={role:role_manifest(split,role) for role in ['fusion_fit','calibration']}
        for manifest in manifests.values():assert sha(manifest['path'])==manifest['sha256']
        labels={role:metadata(m['path'],m['indices'])['labels'] for role,m in manifests.items()}
        assert len(labels['fusion_fit'])==4032 and len(labels['calibration'])==1008
        for seed in CONFIG['seeds']:
            output=model_path(split,seed)
            if output.exists():assert read(output)['protocol_sha256']==sha(pp);continue
            start=time.perf_counter();z,train_binding=cache(split,seed,'fusion_fit');cal,cal_binding=cache(split,seed,'calibration')
            scales=[];biases=[];full_weights=[]
            for j in range(6):
                inputs=np.stack([z[:,j],np.zeros(len(z)),np.zeros(len(z))],axis=1)
                fitted=_fit(ThreeScoreAffine(),inputs,labels['fusion_fit'][:,j:j+1],seed+j,CONFIG['fusion_epochs'],CONFIG['fusion_learning_rate'])
                weights=fitted.linear.weight.detach().numpy()[0];assert np.all(weights[1:]==0)
                scales.append(float(weights[0]));biases.append(float(fitted.linear.bias.detach()[0]));full_weights.append(weights.tolist())
            record={'kind':'single_affine','group':'reference','seed':seed,'scales':scales,'biases':biases,
                    'models':{'reference':{'path':str(checkpoint(split,'reference',seed)),'sha256':sha(checkpoint(split,'reference',seed))}},
                    'protocol_sha256':sha(pp),'full_allocated_weights':full_weights,
                    'training':{'fit':manifests['fusion_fit'],'calibration':manifests['calibration'],
                                'fit_cache':train_binding,'calibration_cache':cal_binding,'optimizer_steps':6*CONFIG['fusion_epochs'],
                                'elapsed_seconds':time.perf_counter()-start,'backbone_updates':0}}
            record['thresholds']=helper.fit_thresholds(apply(cal,record),labels['calibration'])
            write(output,record);print('SUPERVISION_FIT',split,seed,'seconds',round(record['training']['elapsed_seconds'],2),flush=True)


def score(role):
    if role=='confirmation':
        for path,digest in read(ROOT/'final_freeze.json')['files'].items():assert sha(path)==digest,path
    for split in CONFIG['splits']:
        manifest=role_manifest(split,role);assert sha(manifest['path'])==manifest['sha256']
        meta=metadata(manifest['path'],manifest['indices']);folder=STORE/split/(role+'_results');folder.mkdir(parents=True,exist_ok=True)
        np.savez(folder/'metadata.npz',**meta);rows=[]
        for seed in CONFIG['seeds']:
            path=model_path(split,seed);record=read(path);raw,binding=cache(split,seed,role);z=apply(raw,record)
            output=folder/f'reference_affine_{seed}.npz';np.savez(output,logits=z,thresholds=record['thresholds'])
            row={'method':'reference_affine','seed':seed,**evaluate(z,meta,record['thresholds']),
                 'decision_sha256':sha(path),'prediction_sha256':sha(output),'source_cache':binding}
            write(output.with_suffix('.json'),row);rows.append(row)
        write(folder/'summary.json',{'role':role,'split':split,'results':rows,'identities':len(meta['labels']),'time':stamp()})
    analyze(role)


def analyze(role):
    splits=[]
    for split in CONFIG['splits']:
        control=read(STORE/split/(role+'_results')/'summary.json');by_control={r['seed']:r for r in control['results']}
        core=read(ROOT/split/(role+'_results')/'summary.json');by_core={(r['method'],r['seed']):r for r in core['results']}
        contrasts=[]
        pairs=[('reference_affine','reference')]+[(m,'reference_affine') for m in ['natural_none','marginal_none','combo_none','combo_global','combo_segment']]
        def get(method,seed):return by_control[seed] if method=='reference_affine' else by_core[method,seed]
        for candidate,reference in pairs:
            values={key:interval([get(candidate,s)['metrics'][key]-get(reference,s)['metrics'][key] for s in CONFIG['seeds']]) for key in ['f1_exact','fixed05_exact','BCE','macro_AP','ECE']}
            labels=[{'label':label,**{key:interval([get(candidate,s)['labels'][j][key]-get(reference,s)['labels'][j][key] for s in CONFIG['seeds']]) for key in ['FPR','recall','weak_recall']}} for j,label in enumerate(LABELS)]
            contrasts.append({'candidate':candidate,'reference':reference,'differences':values,'labels':labels})
        splits.append({'split':split,'seeds':CONFIG['seeds'],'mean_metrics':{key:float(np.mean([r['metrics'][key] for r in control['results']])) for key in control['results'][0]['metrics']},
                       'scales_by_seed':{str(s):read(model_path(split,s))['scales'] for s in CONFIG['seeds']},'comparisons':contrasts})
    result={'role':role,'splits':splits,'decisions':10,'scope':'Same extra supervision access for a frozen single backbone; not equal total training compute.',
            'intervals':'unadjusted paired-seed t intervals conditional on fixed identity cohorts','time':stamp()}
    write(ROOT/'analysis'/f'supervision_control_{role}.json',result)
    for split in splits:
        print('SUPERVISION_COMPARE',role,split['split'],[(r['candidate'],r['reference'],100*r['differences']['f1_exact']['mean']) for r in split['comparisons']],flush=True)


def verify(role):
    checks=0;maximum=0.;rows=[]
    def equal(actual,expected):
        nonlocal checks,maximum
        error=abs(float(actual)-float(expected));checks+=1;maximum=max(maximum,error);assert error<1e-9,(actual,expected)
    for path,digest in read(STORE/'protocol.json')['sources'].items():assert sha(path)==digest,path
    for split in CONFIG['splits']:
        folder=STORE/split/(role+'_results');meta=np.load(folder/'metadata.npz');truth=meta['labels'].astype(bool)
        manifest=role_manifest(split,role)
        with h5py.File(manifest['path'],'r') as f:np.testing.assert_array_equal(np.asarray(f['known_label']).T[manifest['indices']],truth)
        cal_manifest=role_manifest(split,'calibration');cal_y=metadata(cal_manifest['path'],cal_manifest['indices'])['labels'].astype(bool)
        for seed in CONFIG['seeds']:
            decision=model_path(split,seed);record=read(decision);path=folder/f'reference_affine_{seed}.npz';score=read(path.with_suffix('.json'))
            assert score['decision_sha256']==sha(decision) and score['prediction_sha256']==sha(path)
            assert record['protocol_sha256']==sha(STORE/'protocol.json') and record['training']['optimizer_steps']==1800 and record['training']['backbone_updates']==0
            assert np.all(np.asarray(record['full_allocated_weights'])[:,1:]==0)
            raw,_=cache(split,seed,role);prediction=np.load(path);z=prediction['logits'];np.testing.assert_array_equal(z,raw*np.asarray(record['scales'])+np.asarray(record['biases']))
            np.testing.assert_array_equal(prediction['thresholds'],record['thresholds']);p=expit(z);pred=p>=prediction['thresholds']
            equal(np.all(pred==truth,1).mean(),score['metrics']['f1_exact']);equal(np.all((z>=0)==truth,1).mean(),score['metrics']['fixed05_exact'])
            equal(np.mean(np.logaddexp(0,z)-truth*z),score['metrics']['BCE']);equal(np.mean((p-truth)**2),score['metrics']['Brier'])
            aps=[];eces=[]
            for j,label in enumerate(score['labels']):
                equal((pred[:,j]&~truth[:,j]).sum(),label['FP']);equal((~pred[:,j]&truth[:,j]).sum(),label['FN'])
                equal(pred[~truth[:,j],j].mean(),label['FPR']);equal(pred[truth[:,j],j].mean(),label['recall'])
                weak=meta['weak'][:,j]
                if weak.any():equal(pred[weak,j].mean(),label['weak_recall'])
                ap=average_precision(truth[:,j],p[:,j]);equal(ap,label['AP']);aps.append(ap)
                bins=np.minimum((p[:,j]*15).astype(int),14)
                # Preserve the stored float32 label bin-mean rounding used by the scorer.
                ece=sum(np.mean(bins==k)*abs(float(p[bins==k,j].mean())-float(meta['labels'][bins==k,j].mean())) for k in range(15) if np.any(bins==k))
                equal(ece,label['ECE']);eces.append(ece)
            equal(np.mean(aps),score['metrics']['macro_AP']);equal(np.mean(eces),score['metrics']['ECE'])
            cal,_=cache(split,seed,'calibration');cal_p=expit(apply(cal,record));thresholds=[]
            for j in range(6):
                values=[];grid=np.linspace(.05,.95,181)
                for threshold in grid:
                    chosen=cal_p[:,j]>=threshold;tp=np.sum(chosen&cal_y[:,j]);fp=np.sum(chosen&~cal_y[:,j]);fn=np.sum(~chosen&cal_y[:,j])
                    values.append(2*tp/max(2*tp+fp+fn,1))
                thresholds.append(float(grid[np.argmax(values)]))
            np.testing.assert_array_equal(thresholds,record['thresholds'])
            rows.append({'split':split,'seed':seed,'all_scales_positive':bool((np.asarray(record['scales'])>0).all())})
    result={'passed':True,'role':role,'decisions':len(rows),'scalars_recomputed':checks,'maximum_error':maximum,'records':rows,'time':stamp()}
    write(ROOT/(role+'_supervision_verification.json'),result);print('SUPERVISION_VERIFIED',role,len(rows),checks,maximum,flush=True)


def runtime(role):
    from runtime import FactorialRuntime
    setup();rows=[]
    for split in CONFIG['splits']:
        manifest=role_manifest(split,role);positions=np.linspace(0,len(manifest['indices'])-1,64,dtype=int)
        x,y=base.load(manifest['path'],np.asarray(manifest['indices'])[positions]);assert len(np.unique(y.astype(int)@(1<<np.arange(6))))==63
        for seed in CONFIG['seeds']:
            model=FactorialRuntime(split,read(model_path(split,seed))).cuda().eval()
            with torch.inference_mode():z=model(x.cuda()).cpu().numpy()
            expected=np.load(STORE/split/(role+'_results')/f'reference_affine_{seed}.npz')['logits'][positions]
            np.testing.assert_allclose(z,expected,rtol=1e-6,atol=1e-6)
            rows.append({'split':split,'seed':seed,'maximum_error':float(abs(z-expected).max())});del model
    write(ROOT/(role+'_supervision_runtime_verification.json'),{'passed':True,'decisions':len(rows),'distinct_combinations':63,'records':rows,'time':stamp()})
    print('SUPERVISION_RUNTIME_VERIFIED',role,len(rows),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['fit','score','verify','runtime']);parser.add_argument('--role',default='development');args=parser.parse_args()
    if args.action=='fit':fit()
    else:globals()[args.action](args.role)
