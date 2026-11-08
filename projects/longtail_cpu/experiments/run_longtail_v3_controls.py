"""Five training controls x five seeds; paired fixed/F1 decisions per model.

Freeze first. Run the snapshot copy with snapshot/src on PYTHONPATH. Report
identities are evaluated only after all 25 training/check jobs complete.
"""
import argparse
import csv
from datetime import datetime
from html import escape
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parents[1]
METHODS={'bce':'fresh_cpu','label_sampler':'label_sampler','combo_sampler':'combo_sampler',
         'weighted_bce':'weighted_bce','logit_adjustment':'logit_adjustment'}
NAMES={'bce':'ordinary BCE','label_sampler':'label-aware sampler','combo_sampler':'combo-aware sampler',
       'weighted_bce':'weighted BCE','logit_adjustment':'Logit Adjustment'}
CORES=list(range(8,24))+list(range(40,56))


def now():return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')


def prepare(suite,report,benchmark):
    import numpy as np
    from longtail.config import resolve_config
    from longtail.data.dataset import read_metadata
    from longtail.data.splits import build_split
    from longtail.data.sampling import sampling_probabilities
    from longtail.data.manage import verify
    from longtail.io import read_json,write_json,sha256
    from longtail.models.factory import build_model
    if suite.exists():raise FileExistsError('Use a new suite directory')
    bench=read_json(benchmark);threads=bench['recommended_threads'];parallel=bench['recommended_parallel_jobs']
    slots=[CORES[i*threads:(i+1)*threads] for i in range(parallel)]
    if not set(sum(slots,[]))<=os.sched_getaffinity(0):raise ValueError('Requested cores unavailable')
    base=resolve_config(ROOT/'configs/longtail_v3_fresh_cpu.json');base['train']['cpu_threads']=threads
    manifest=read_json(base['data']['manifest']);data_root=Path(base['data']['manifest']).parent
    verified=verify(data_root);split=build_split(manifest,data_root,base['data'])
    y,ids,_=read_metadata(data_root/manifest['files']['train']['path']);n=len(y);counts=y.sum(0)
    sampling={}
    for mode in ['uniform','label_aware','combo_aware']:
        p=sampling_probabilities(y,mode)
        sampling[mode]={'replacement':mode!='uniform','expected_label_fractions':(p@y).tolist(),
            'expected_mean_cardinality':float(p@y.sum(1)),
            'expected_unique_fit_identities_per_epoch':float(n if mode=='uniform' else np.sum(1-(1-p)**n))}
    suite.mkdir(parents=True);jobs=[]
    for method,suffix in METHODS.items():
        for seed in (2050,2051,2052,2053,2054):
            c=resolve_config(ROOT/f'configs/longtail_v3_{suffix}.json');c['train']['cpu_threads']=threads
            assert c['data']==base['data'] and c['dataloader']==base['dataloader'] and c['model']==base['model']
            assert c['train']['epochs']==24 and c['train']['scheduler']=='cosine'
            c['train']['seed']=seed;job_id=f'{method}_seed{seed}';c['run']['output_dir']=str(suite/job_id)
            path=suite/'configs'/f'{job_id}.json';write_json(path,c);slot=len(jobs)%parallel
            count=sum(p.numel() for p in build_model(c['model'],num_classes=6).parameters());assert count==257511
            jobs.append({'id':job_id,'method':method,'seed':seed,'config':str(path),'output':c['run']['output_dir'],
                         'slot':slot,'cpus':slots[slot],'parameters':count})
    files=sorted((ROOT/'src').rglob('*.py'))+sorted((ROOT/'configs').glob('*.json'))+\
          sorted((ROOT/'experiments').glob('*.py'))+sorted((ROOT/'tests').glob('*.py'))+[ROOT/'pyproject.toml',ROOT/'uv.lock']
    source=[]
    for path in files:
        rel=path.relative_to(ROOT);dest=suite/'source_snapshot'/rel;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(path,dest);source.append({'path':str(rel),'sha256':sha256(path)})
    plan={'created':now(),'purpose':'Attribute long-tail gains to sampling, training loss and decision thresholds.',
        'jobs':jobs,'source_manifest':source,'report_output':str(report),'benchmark':bench,
        'protocol':{'seeds':[2050,2051,2052,2053,2054],'epochs':24,'draws_per_epoch':8000,'batch_size':32,
            'steps_per_epoch':250,'optimizer_steps_per_job':6000,'total_jobs':25,'total_training_epochs':600,
            'model':'DFMS-IQ, random initialization, 257511 parameters','device':'cpu','amp':'none','world_size':1,
            'cpu_threads_per_job':threads,'maximum_parallel_jobs':parallel,'cpu_slots':slots,
            'optimizer':'AdamW','lr':1e-4,'weight_decay':1e-4,'scheduler':'24-epoch cosine',
            'checkpoint_selection':'check Macro-AUPR, threshold-free; identical for all five methods',
            'threshold_conditions':['fixed sigmoid 0.5','per-label exact F1 over observed calibration scores, ties choose lowest threshold'],
            'calibration_is_postprocessing':True,'calibration_additional_training_jobs':0,
            'training_methods':{'bce':'uniform shuffled identities, ordinary BCE',
                'label_sampler':'uniform label mixture: probability proportional to sum(y_c/n_positive_c), replacement',
                'combo_sampler':'probability proportional to inverse combination count, replacement',
                'weighted_bce':'uniform identities; BCE pos_weight=(N-n_positive)/n_positive',
                'logit_adjustment':'uniform identities; BCE(z+logit(pi),y), tau=1; evaluate raw z, no second correction'},
            'all_priors_and_sampling_weights_from':'fit identities only',
            'data_split_sha256':split['sha256'],'data_pool':'fresh v3 same-physics distribution, Python PCG64 port, not MATLAB RNG parity',
            'report_role':'fresh predeclared reporting identities; no report metrics until every fixed training job completes',
            'report_not_used_for_epoch_threshold_or_hyperparameter_selection':True,
            'dataset_generation_plan':str(data_root/'generation_plan.json'),
            'primary_metrics':['Macro-AUPR','Macro-F1','Exact','tail-label F1/AUPR','rare-combination Exact'],
            'tail_label_definition':'lowest two fit positive counts, fixed before training',
            'rare_combo_definition':'all combinations attaining the minimum fit combination count; keep every tie',
            'effects':['method_fixed minus BCE_fixed','method_calibrated minus BCE_calibrated',
                      'calibrated minus fixed within each model','difference of calibration gains versus BCE'],
            'no_sampler_loss_stacking':True,'no_hyperparameter_sweep':True},
        'sampling_expectations':sampling,'train_label_counts':counts.astype(int).tolist(),
        'weighted_bce_pos_weight':((n-counts)/counts).tolist(),
        'logit_adjustment_bias':np.log(counts/(n-counts)).tolist(),'dataset_verification':verified}
    write_json(suite/'plan.json',plan)
    print(json.dumps({'suite':str(suite),'jobs':25,'threads':threads,'parallel':parallel,
                     'sampling':sampling,'estimated_hours':bench['estimated_wall_minutes']/60},ensure_ascii=False),flush=True)


def worker(suite,job_id,stage):
    from longtail.io import read_json,write_json,sha256
    plan=read_json(suite/'plan.json');job=next(j for j in plan['jobs'] if j['id']==job_id)
    os.sched_setaffinity(0,set(job['cpus']));os.nice(5)
    import torch
    torch.set_num_threads(plan['protocol']['cpu_threads_per_job']);torch.set_num_interop_threads(1)
    from longtail.train import run
    from longtail.evaluate import run as evaluate
    output=Path(job['output']);phase=suite/'phases'/f'{job_id}_{stage}.json'
    write_json(phase,{'phase':'running','started':now(),'pid':os.getpid(),'cpus':job['cpus']})
    try:
        if stage=='train':
            config=read_json(job['config']);status=read_json(output/'status.json') if (output/'status.json').exists() else {}
            if status.get('status')!='completed':
                run(config,resume=output/'last.pt' if (output/'last.pt').exists() else None)
            role='check'
        else:
            if not (suite/'all_training_completed.json').exists():raise RuntimeError('Report split is deferred until all 25 jobs finish')
            role='development_test'
        checkpoint=output/'best.pt';digest=sha256(checkpoint);dest=output/'evaluation'/role
        if not (dest/'metrics.json').exists():evaluate(checkpoint,split_name=role,output=dest,device_name='cpu')
        assert read_json(dest/'metrics.json')['checkpoint_sha256']==digest and sha256(checkpoint)==digest
        write_json(phase,{'phase':'completed','finished':now(),'pid':os.getpid(),'checkpoint_sha256':digest})
    except BaseException as e:
        write_json(phase,{'phase':'failed','failed':now(),'error':repr(e),'pid':os.getpid()});raise


def launch_stage(suite,stage):
    from longtail.io import read_json,write_json
    plan=read_json(suite/'plan.json');parallel=plan['protocol']['maximum_parallel_jobs']
    queues={slot:[j for j in plan['jobs'] if j['slot']==slot] for slot in range(parallel)}
    active={};finished=set();failures=[];logs=suite/'logs';logs.mkdir(exist_ok=True)
    def start(slot):
        if not queues[slot]:return
        job=queues[slot].pop(0);handle=(logs/f'{job["id"]}_{stage}.log').open('a')
        process=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--suite',str(suite),
            '--worker',job['id'],'--stage',stage],stdout=handle,stderr=subprocess.STDOUT)
        active[slot]=(job,process,handle)
    for slot in queues:start(slot)
    while active:
        for slot,(job,process,handle) in list(active.items()):
            code=process.poll()
            if code is not None:
                handle.close();del active[slot];finished.add(job['id'])
                if code:failures.append({'job':job['id'],'stage':stage,'exit_code':code})
                start(slot)
        rows=[]
        for j in plan['jobs']:
            row={'job':j['id'],'method':j['method'],'seed':j['seed'],'slot':j['slot'],'cpus':j['cpus']}
            h=Path(j['output'])/'history.json'
            if h.exists():
                history=read_json(h);row.update(epoch=history[-1]['epoch'],loss=history[-1]['loss'],
                    last_epoch_seconds=history[-1].get('epoch_seconds'),check_macro_aupr=history[-1]['validation']['macro_aupr'])
            row['phase']='completed' if j['id'] in finished else 'running' if any(x[0]['id']==j['id'] for x in active.values()) else 'queued'
            for running,p,_ in active.values():
                if running['id']==j['id']:row['pid']=p.pid
            rows.append(row)
        write_json(suite/'progress.json',{'stage':stage,'updated':now(),'completed_jobs':len(finished),'total_jobs':25,
            'completed_epochs':sum(r.get('epoch',0) for r in rows),'total_epochs':600,'failures':failures,'jobs':rows})
        if active:time.sleep(5)
    if failures:raise RuntimeError(str(failures))


def summarize(suite):
    import numpy as np
    from scipy.stats import t
    from longtail.io import read_json,write_json,sha256
    from statistics import mean,stdev
    plan=read_json(suite/'plan.json');report=Path(plan['report_output']);report.mkdir(parents=True,exist_ok=True)
    rows=[];references={};tail=np.argsort(plan['train_label_counts'],kind='stable')[:2]
    for job in plan['jobs']:
        output=Path(job['output']);digest=sha256(output/'best.pt');decision=read_json(output/'decision.json')
        h=read_json(output/'history.json');assert len(h)==24 and all(r['samples_seen']==8000 and r['sampler_padding']==0 for r in h)
        for role in ['check','development_test']:
            e=read_json(output/'evaluation'/role/'metrics.json');z=dict(np.load(output/'evaluation'/role/'predictions.npz'))
            assert e['checkpoint_sha256']==digest and e['decision']==decision and e['split_sha256']==plan['protocol']['data_split_sha256']
            if role not in references:references[role]=z
            for key in ['identities','indices','targets']:np.testing.assert_array_equal(z[key],references[role][key])
            for condition,m in e['metrics'].items():
                labels=list(m['per_label'])
                code=z['targets'].astype(np.int64)@(1<<np.arange(6))
                combo_counts=np.asarray(read_json(output/'split.json')['fit_combination_counts'])
                rare=np.flatnonzero(combo_counts==combo_counts.min())+1;mask=np.isin(code,rare)
                p=1/(1+np.exp(-np.clip(z['logits'].astype(float),-40,40)))
                predicted=p>=np.asarray(m['thresholds']);correct=(predicted==z['targets']).all(1)
                row={'method':job['method'],'seed':job['seed'],'role':role,'decision':condition,
                    'macro_f1':m['macro_f1'],'macro_aupr':m['macro_aupr'],'exact':m['subset_accuracy'],
                    'tail_f1':float(np.mean([m['per_label'][labels[c]]['f1'] for c in tail])),
                    'tail_aupr':float(np.mean([m['per_label'][labels[c]]['aupr'] for c in tail])),
                    'rare_combo_exact':float(correct[mask].mean()),
                    'mean_fpr':float(np.mean([v['negative_fpr'] for v in m['per_label'].values()])),
                    'checkpoint_sha256':digest,'best_epoch':read_json(output/'status.json')['best_epoch']}
                rows.append(row)
    keys=['macro_f1','macro_aupr','exact','tail_f1','tail_aupr','rare_combo_exact','mean_fpr']
    aggregates={role:{method:{decision:{k:{'mean':mean(r[k] for r in rows if r['role']==role and r['method']==method and r['decision']==decision),
        'std':stdev(r[k] for r in rows if r['role']==role and r['method']==method and r['decision']==decision)} for k in keys}
        for decision in ['fixed_0.5','calibrated']} for method in METHODS} for role in ['check','development_test']}
    def val(method,seed,condition,key):
        return next(r[key] for r in rows if r['role']=='development_test' and r['method']==method and r['seed']==seed and r['decision']==condition)
    effects={}
    for method in METHODS:
        effects[method]={}
        for key in keys:
            modes={'raw_training_effect':lambda s:val(method,s,'fixed_0.5',key)-val('bce',s,'fixed_0.5',key),
                   'training_effect_after_equal_calibration':lambda s:val(method,s,'calibrated',key)-val('bce',s,'calibrated',key),
                   'within_method_calibration_effect':lambda s:val(method,s,'calibrated',key)-val(method,s,'fixed_0.5',key),
                   'calibration_interaction_vs_bce':lambda s:(val(method,s,'calibrated',key)-val(method,s,'fixed_0.5',key))-(val('bce',s,'calibrated',key)-val('bce',s,'fixed_0.5',key))}
            effects[method][key]={}
            for name,fn in modes.items():
                d=np.array([fn(s) for s in [2050,2051,2052,2053,2054]]);delta=float(d.mean());radius=float(t.ppf(.975,4)*d.std(ddof=1)/np.sqrt(5))
                effects[method][key][name]={'mean':delta,'paired_seed_95_t_interval':[delta-radius,delta+radius],'per_seed':d.tolist()}
    result={'created':now(),'plan':plan,'rows':rows,'aggregate':aggregates,'effects':effects,
            'validation':{'all_25_jobs_completed_24_epochs':True,'frozen_checkpoints_and_thresholds':True,'paired_prediction_identities':True},
            'interpretation':'The paired seed interval conditions on this fixed fresh pool. It is not uncertainty over simulator implementations or new datasets.'}
    write_json(report/'results.json',result)
    with (report/'results.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    text='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>v3长尾基线归因</title><style>body{font:16px/1.7 system-ui;max-width:1150px;margin:30px auto;color:#234}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:9px;border:1px solid #ccd}th{background:#e5f0f4}code{overflow-wrap:anywhere}</style><h1>v3新身份：长尾收益来自采样、损失还是阈值</h1>'
    text+='<p>5种训练方法×5种子；统一DFMS、24轮、每轮8000次抽样和250步；按check Macro-AUPR选模。每个模型报告固定0.5与calibration逐标签F1阈值；报告身份在全部预定训练完成后才测评。数据沿用v3物理与10–30 dB独立分量分布，执行使用Python PCG64，非MATLAB逐点复现。</p>'
    for condition in ['fixed_0.5','calibrated']:
        text+='<h2>新身份报告集：'+condition+'</h2><table><tr><th>方法</th>'+''.join('<th>'+k+'</th>' for k in keys)+'</tr>'
        for method in METHODS:
            text+='<tr><td>'+NAMES[method]+'</td>'+''.join(f"<td>{aggregates['development_test'][method][condition][k]['mean']:.5f} ± {aggregates['development_test'][method][condition][k]['std']:.5f}</td>" for k in keys)+'</tr>'
        text+='</table>'
    text+='<h2>归因：同样校准后，训练方法还有多少增益</h2><table><tr><th>方法</th><th>Macro-F1差 pp</th><th>五种子配对95%区间 pp</th><th>Macro-AUPR差</th><th>本方法校准带来的F1差 pp</th></tr>'
    for method in METHODS:
        e=effects[method];f=e['macro_f1']['training_effect_after_equal_calibration'];ap=e['macro_aupr']['training_effect_after_equal_calibration'];cal=e['macro_f1']['within_method_calibration_effect']
        text+=f"<tr><td>{NAMES[method]}</td><td>{f['mean']*100:+.3f}</td><td>"+' / '.join(f'{x*100:+.3f}' for x in f['paired_seed_95_t_interval'])+f"</td><td>{ap['mean']:+.5f}</td><td>{cal['mean']*100:+.3f}</td></tr>"
    text+='</table><p>阈值变化不会改变逐类排序与Macro-AUPR；若固定0.5改善在统一校准后消失且AUPR不变，主要收益应解释为决策偏移。五种子区间仍只条件于本固定新身份池；采样改变标签共现和重复曝光，详见plan与逐轮sampling记录。</p><p><a href="results.json">完整结果和归因</a> · <a href="results.csv">逐种子CSV</a></p></html>'
    (report/'report.html').write_text(text)


def pipeline(suite):
    from longtail.io import read_json,write_json,sha256
    import longtail
    plan=read_json(suite/'plan.json');start=time.time();status=suite/'pipeline.json'
    try:
        assert Path(longtail.__file__).resolve().is_relative_to(suite/'source_snapshot/src')
        for e in plan['source_manifest']:assert sha256(ROOT/e['path'])==e['sha256']
        write_json(status,{'status':'running','phase':'training_and_check','started':now(),'controller_pid':os.getpid(),
                           'module_source':str(Path(longtail.__file__).resolve())})
        launch_stage(suite,'train');write_json(suite/'all_training_completed.json',{'finished':now(),'jobs':25})
        write_json(status,{'status':'running','phase':'deferred_fresh_report_evaluation','updated':now(),'controller_pid':os.getpid()})
        launch_stage(suite,'report');summarize(suite)
        write_json(status,{'status':'completed','finished':now(),'wall_seconds':time.time()-start,'jobs_completed':25,
            'report':str(Path(plan['report_output'])/'report.html')})
        write_json(suite/'pipeline_completed.json',read_json(status))
    except BaseException as e:
        write_json(status,{'status':'failed','failed':now(),'error':repr(e),'controller_pid':os.getpid()});raise


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--suite',required=True,type=Path)
    p.add_argument('--report',type=Path);p.add_argument('--benchmark',type=Path);p.add_argument('--prepare',action='store_true')
    p.add_argument('--run',action='store_true');p.add_argument('--worker');p.add_argument('--stage',choices=['train','report'])
    args=p.parse_args();suite=args.suite.resolve()
    if args.prepare:prepare(suite,args.report.resolve(),args.benchmark.resolve())
    elif args.worker:worker(suite,args.worker,args.stage)
    elif args.run:pipeline(suite)
    else:p.error('Choose --prepare or --run')


if __name__=='__main__':main()
