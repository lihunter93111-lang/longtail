"""Launch four evidence/readout controls and automatically finish evaluation.

--prepare freezes configurations and source. Run the snapshot copy with its src
on PYTHONPATH so later workspace edits cannot change a running experiment.
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
ARMS={'baseline':'cpu','single':'single','fused':'fused','uniform':'uniform'}
NAMES={'baseline':'DFMS基线','single':'最深层查询','fused':'融合后特征查询','uniform':'均匀汇聚容量对照'}
SLOTS=[list(range(a,a+4)) for a in (8,12,16,20,40,44,48,52)]


def now():return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')


def prepare(suite,report):
    from longtail.config import resolve_config
    from longtail.data.manage import verify
    from longtail.data.splits import build_split
    from longtail.models.factory import build_model
    from longtail.io import read_json,write_json,sha256
    if suite.exists():raise FileExistsError('Use a new suite directory')
    cpus=sum(SLOTS,[])
    if not set(cpus)<=os.sched_getaffinity(0):raise ValueError('CPU slots are unavailable')
    physical=[(Path(f'/sys/devices/system/cpu/cpu{c}/topology/physical_package_id').read_text().strip(),
               Path(f'/sys/devices/system/cpu/cpu{c}/topology/core_id').read_text().strip()) for c in cpus]
    if len(set(physical))!=32:raise ValueError('Expected 32 distinct physical cores')
    base=resolve_config(ROOT/'configs/query_evidence24_cpu.json')
    manifest=read_json(base['data']['manifest']);data_root=Path(base['data']['manifest']).parent
    verified=verify(data_root);split=build_split(manifest,data_root,base['data'])
    suite.mkdir(parents=True);jobs=[]
    for arm,suffix in ARMS.items():
        for seed in (2050,2051):
            config=resolve_config(ROOT/f'configs/query_evidence24_{suffix}.json')
            assert config['train']==base['train'] and config['data']==base['data']
            assert config['dataloader']==base['dataloader'] and config['evaluation']==base['evaluation']
            config['train']['seed']=seed;job_id=f'{arm}_seed{seed}'
            config['run']['output_dir']=str(suite/job_id);path=suite/'configs'/f'{job_id}.json'
            write_json(path,config);model=build_model(config['model'],num_classes=6)
            count=sum(p.numel() for p in model.parameters())
            assert count==(257511 if arm=='baseline' else 300211)
            jobs.append({'id':job_id,'arm':arm,'seed':seed,'config':str(path),
                         'output':config['run']['output_dir'],'cpus':SLOTS[len(jobs)],'parameters':count})
    paths=sorted((ROOT/'src').rglob('*.py'))+sorted((ROOT/'configs').glob('*.json'))+\
          sorted((ROOT/'experiments').glob('*.py'))+sorted((ROOT/'tests').glob('*.py'))+[ROOT/'pyproject.toml',ROOT/'uv.lock']
    source=[]
    for path in paths:
        rel=path.relative_to(ROOT);dest=suite/'source_snapshot'/rel
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,dest)
        source.append({'path':str(rel),'sha256':sha256(path)})
    write_json(suite/'plan.json',{'created':now(),'repository':str(ROOT),'jobs':jobs,'source_manifest':source,
        'report_output':str(report),'purpose':'Separate evidence source, learned attention and added active head capacity.',
        'protocol':{'seeds':[2050,2051],'epochs':24,'global_batch':32,'optimizer':'adamw',
            'lr':1e-4,'weight_decay':1e-4,'scheduler':'cosine over 24 epochs; declared before launch',
            'loss':'bce','initialization':'random, matched initial global branch; fresh 24-epoch runs',
            'device':'cpu','amp':'none','world_size':1,'cpu_threads_per_job':4,'parallel_jobs':8,
            'physical_cpu_slots':SLOTS,'data_split_sha256':split['sha256'],
            'checkpoint_selection':'check weak_macro_aupr','threshold_fit':'calibration per-label F1',
            'fpr_fit':'calibration negatives, targets 0.01 and 0.05',
            'evaluation_roles':['check','calibration','development_test'],
            'paired_power_views_db':[0,6,12,18,24],
            'candidate_decisions_use':'check; development_test is a consumed development report, not an independent test',
            'locked_test_used':False,'development_test_consumed_previously':True,
            'hyperparameters_selected_using_this_round_development_test':False,
            'uniform_capacity_control':'All parameters active; post-pooling MLPs, no unused Q/K tensors',
            'fused_comparison':'Only query memory source changes; 128-channel source, identical initialized state and parameter count'},
        'validation':{'full_pytest':'51 passed in 32.05s before explicit constructor/config freeze clarification; targeted rerun recorded in preflight.json',
                      'matched_head_total_parameters':300211,'distinct_physical_cores':len(set(physical))},
        'dataset_verification':verified})
    print(json.dumps({'suite':str(suite),'jobs':len(jobs),'parameters':{j['arm']:j['parameters'] for j in jobs},
                      'data_split_sha256':split['sha256']},ensure_ascii=False),flush=True)


def evaluation_worker(suite,job_id):
    from longtail.io import read_json,write_json,sha256
    # This sets affinity and torch interop threads once in the fresh process.
    from evaluate_query_route1_fpr import worker as evaluate_fpr
    evaluate_fpr(suite,job_id)
    from longtail.evaluate import run as evaluate
    from longtail.metrics import evaluate_metrics
    from longtail.weak_metrics import weak_metrics
    import numpy as np
    job=next(j for j in read_json(suite/'plan.json')['jobs'] if j['id']==job_id)
    output=Path(job['output']);checkpoint=output/'best.pt';digest=sha256(checkpoint)
    split=read_json(output/'split.json');decision=read_json(output/'decision.json')
    phase=suite/'evaluation_phases'/f'{job_id}.json'
    write_json(phase,{'phase':'running','started':now(),'pid':os.getpid()})
    targets=read_json(output/'evaluation/fpr_controlled/decisions.json');fpr_views={}
    for gap in (0,6,12,18,24):
        dest=output/'evaluation/check'/f'gap_{gap}'
        if not (dest/'metrics.json').exists():
            evaluate(checkpoint,split_name='check',output=dest,device_name='cpu',view=f'gap_{gap}')
        e=read_json(dest/'metrics.json')
        assert e['checkpoint_sha256']==digest and e['decision']==decision
        z=dict(np.load(dest/'predictions.npz'));fpr_views[f'gap_{gap}']={}
        for name,d in targets.items():
            m=evaluate_metrics(z['targets'],z['logits'],split['classes'],d['thresholds'])
            w=weak_metrics(z['targets'],z['logits'],split['classes'],d['thresholds'],
                           z['weak_mask'],z['relative_power_db'],split['fit_label_counts'])
            fpr_views[f'gap_{gap}'][name]={'metrics':m,'weak_component':w}
    assert sha256(checkpoint)==digest
    write_json(output/'evaluation/fpr_controlled/check_power_views.json',{'checkpoint_sha256':digest,
               'threshold_fit_role':'calibration','metrics':fpr_views})
    write_json(phase,{'phase':'completed','finished':now(),'pid':os.getpid(),'checkpoint_sha256':digest})


def summarize(suite):
    from longtail.io import read_json,write_json,sha256
    from statistics import mean,stdev
    import numpy as np
    plan=read_json(suite/'plan.json');report=Path(plan['report_output']);report.mkdir(parents=True,exist_ok=True)
    rows=[];reference={};power=[];fpr=[]
    for job in plan['jobs']:
        output=Path(job['output']);status=read_json(output/'status.json');h=read_json(output/'history.json')
        digest=sha256(output/'best.pt');decision=read_json(output/'decision.json')
        assert len(h)==24 and all(r['samples_seen']==8000 and r['sampler_padding']==0 for r in h)
        assert status['device']=='cpu' and status['world_size']==1
        for role in ('check','development_test'):
            for gap in (None,0,6,12,18,24):
                dest=output/'evaluation'/role
                if gap is not None:dest=dest/f'gap_{gap}'
                e=read_json(dest/'metrics.json');z=dict(np.load(dest/'predictions.npz'))
                assert e['checkpoint_sha256']==digest and e['decision']==decision
                assert e['split_sha256']==plan['protocol']['data_split_sha256']
                key=(role,gap)
                if key not in reference:reference[key]=z
                for field in ('indices','identities','targets','weak_mask','relative_power_db'):
                    np.testing.assert_array_equal(z[field],reference[key][field])
                m=e['metrics']['calibrated'];w=m['weak_component']
                record={'arm':job['arm'],'seed':job['seed'],'role':role,'gap_db':gap,
                    'exact':m['subset_accuracy'],'macro_f1':m['macro_f1'],'weak_aupr':w['weak_macro_aupr'],
                    'weak_recall':w['weak_macro_recall'],'mean_fpr':float(np.mean([v['negative_fpr'] for v in w['per_label'].values()])),
                    'parameters':status['parameters'],'best_epoch':status['best_epoch'],
                    'train_seconds':status['elapsed_seconds'],'checkpoint':str(output/'best.pt'),'checkpoint_sha256':digest}
                (rows if gap is None else power).append(record)
        fpr.append({'job':job['id'],'results':read_json(output/'evaluation/fpr_controlled/metrics.json')})
    aggregate={role:{arm:{key:{'mean':mean(r[key] for r in rows if r['role']==role and r['arm']==arm),
                        'std':stdev(r[key] for r in rows if r['role']==role and r['arm']==arm)}
                 for key in ('exact','macro_f1','weak_aupr','weak_recall','mean_fpr','train_seconds')}
              for arm in ARMS} for role in ('check','development_test')}
    result={'created':now(),'suite':str(suite),'plan':plan,'rows':rows,'aggregate':aggregate,
            'power_views':power,'fpr_controlled':fpr,'validation':{'paired_metadata':True,
            'all_8_runs_24_epochs':True,'frozen_thresholds_and_weights':True},
            'limitations':['Two seeds and consumed development data; an independent test is still required.',
                           'All four arms use the same new 24-epoch cosine schedule; old 8-epoch scores are not the control.']}
    write_json(report/'results.json',result)
    with (report/'results.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    text='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>24轮类别查询对照</title><style>body{font:16px/1.7 system-ui;max-width:1100px;margin:32px auto;color:#234}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccd;padding:10px}th{background:#e6f1f4}code{overflow-wrap:anywhere}</style><h1>24轮：证据来源与读取方式对照</h1>'
    text+='<p>4组×2种子，32个物理核，BCE、AdamW、24轮cosine。check弱AUPR选模，calibration逐类F1与1%/5%误报目标拟合阈值。check用于开发比较；development-test已消费，不是独立测试。所有模型的主权重和阈值均冻结后测评。</p>'
    for role in ('check','development_test'):
        text+='<h2>'+escape(role)+'</h2><table><tr><th>模型</th><th>参数量</th><th>Exact %</th><th>Macro-F1 %</th><th>弱AUPR</th><th>弱召回 %</th><th>平均FPR %</th></tr>'
        for arm in ARMS:
            a=aggregate[role][arm];text+='<tr><td>'+NAMES[arm]+'</td><td>'+str(257511 if arm=='baseline' else 300211)+'</td>'
            for key,factor in [('exact',100),('macro_f1',100),('weak_aupr',1),('weak_recall',100),('mean_fpr',100)]:
                text+=f"<td>{a[key]['mean']*factor:.4f} ± {a[key]['std']*factor:.4f}</td>"
            text+='</tr>'
        text+='</table><p>±为两个种子的样本标准差。</p>'
    text+='<p>逐次结果：<a href="results.csv">results.csv</a>；功率分档与误报约束：<a href="results.json">results.json</a>。</p><code>'+escape(str(suite))+'</code></html>'
    (report/'report.html').write_text(text)
    print(json.dumps({'report':str(report/'report.html'),'aggregate':aggregate},ensure_ascii=False),flush=True)


def run_pipeline(suite):
    from longtail.io import read_json,write_json,sha256
    import longtail
    plan=read_json(suite/'plan.json');started=time.time();pipeline=suite/'pipeline.json'
    try:
        expected=suite/'source_snapshot/src'
        if not Path(longtail.__file__).resolve().is_relative_to(expected):
            raise RuntimeError('Run the frozen snapshot with its src on PYTHONPATH')
        write_json(pipeline,{'status':'running','phase':'training_and_primary_evaluation','started':now(),
                           'controller_pid':os.getpid(),'module_source':str(Path(longtail.__file__).resolve())})
        from validate_query_route1_cpu import launch
        launch(suite,'train')
        write_json(pipeline,{'status':'running','phase':'fpr_and_check_power_evaluation','updated':now(),'controller_pid':os.getpid()})
        logs=suite/'post_evaluation_logs';logs.mkdir(exist_ok=True);processes=[]
        for job in plan['jobs']:
            handle=(logs/(job['id']+'.log')).open('a')
            proc=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--suite',str(suite),
                '--evaluation-worker',job['id']],stdout=handle,stderr=subprocess.STDOUT)
            processes.append((job,proc,handle))
        write_json(suite/'post_evaluation_processes.json',{'jobs':[{'id':j['id'],'pid':p.pid} for j,p,_ in processes]})
        codes=[]
        for job,p,h in processes:codes.append(p.wait());h.close()
        if any(codes):raise RuntimeError('Post-evaluation failed: '+str(codes))
        summarize(suite)
        for e in plan['source_manifest']:
            assert sha256(ROOT/e['path'])==e['sha256']
        write_json(pipeline,{'status':'completed','finished':now(),'wall_seconds':time.time()-started,
                            'jobs_completed':8,'report':str(Path(plan['report_output'])/'report.html')})
        write_json(suite/'pipeline_completed.json',read_json(pipeline))
    except BaseException as error:
        write_json(pipeline,{'status':'failed','failed':now(),'error':repr(error),'controller_pid':os.getpid()})
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--suite',required=True,type=Path)
    p.add_argument('--report',type=Path);p.add_argument('--prepare',action='store_true');p.add_argument('--run',action='store_true')
    p.add_argument('--evaluation-worker');args=p.parse_args();suite=args.suite.resolve()
    if args.prepare:prepare(suite,args.report.resolve())
    elif args.evaluation_worker:evaluation_worker(suite,args.evaluation_worker)
    elif args.run:run_pipeline(suite)
    else:p.error('Choose --prepare or --run')


if __name__=='__main__':main()
