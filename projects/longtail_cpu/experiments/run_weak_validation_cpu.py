"""CPU A/B/C/D experiment with frozen FMB experts and deferred fresh reports."""
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
SEEDS=list(range(2050,2055))
METHODS=['A','B','C','D']
NAMES={'A':'现有融合基线','B':'32k多样性','C':'32k弱难例','D':'弱难例+1.5倍弱正例'}
CORES=list(range(8,24))+list(range(40,56))
KEYS=['exact','macro_f1','macro_aupr','tail_f1','rare_exact','mean_fpr','weak_micro_recall','target_weak_recall','target_weak_aupr','nonweak_recall']


def now():return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')


def table(headers,rows):
    return '<div style="overflow-x:auto"><table><tr>'+''.join('<th>'+escape(str(v))+'</th>' for v in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+escape(str(v))+'</td>' for v in r)+'</tr>' for r in rows)+'</table></div>'


STYLE='<style>body{font:16px/1.8 system-ui;max-width:1250px;margin:30px auto;padding:20px;color:#234}table{border-collapse:collapse;width:100%;font-size:14px;margin:18px 0}th,td{padding:8px;border:1px solid #ccd}th{background:#e7f2f3}code{overflow-wrap:anywhere}a{color:#16758a}</style>'


def prepare(suite,report,data_root,previous):
    import numpy as np
    import torch
    from longtail.io import read_json,write_json,sha256
    from longtail.config import resolve_config,validate
    from longtail.data.dataset import read_metadata
    from longtail.data.manage import verify
    from longtail.data.splits import build_split
    from longtail.data.loaders import make_dataset
    from longtail.data.sampling import EpochBudgetSampler
    from longtail.models.factory import build_model
    from longtail.engine.trainer import build_loss,WeakPositiveBCE
    if suite.exists():raise FileExistsError('Use a new suite')
    ready=read_json(data_root/'generation_complete.json');assert ready['status']=='completed'
    base=resolve_config(ROOT/'configs/longtail_v3_fresh_cpu.json')
    base['train'].update(draws_per_epoch=8000,epochs=24,loss='bce',sampler='uniform',cpu_threads=4,device='cpu')
    evaluation_manifest=data_root/'evaluation/dataset.json';evaluation=read_json(evaluation_manifest)
    context_split=build_split(evaluation,evaluation_manifest.parent,base['data'])
    verify(evaluation_manifest.parent)
    manifests={'B':Path(ready['manifests']['B_diverse']),'C':Path(ready['manifests']['C_weak']),'D':Path(ready['manifests']['C_weak'])}
    for path in set(manifests.values()):verify(path.parent)
    assert read_json(manifests['C'])['files']['train']['sha256']==read_json(manifests['D'])['files']['train']['sha256']
    slots=[CORES[i:i+4] for i in range(0,32,4)];assert set(CORES)<=os.sched_getaffinity(0)
    torch.set_num_threads(4);smoke=[]
    for method,path in manifests.items():
        manifest=read_json(path);split=build_split(manifest,path.parent,base['data'])
        assert all(split['roles'][r]==context_split['roles'][r] for r in ['calibration','check','development_test'])
        fit=make_dataset(manifest,path.parent,split,'fit',base['dataloader'])
        sampler=EpochBudgetSampler(fit.labels,2050,8000);sampler.set_epoch(1);idx=list(sampler)[:32]
        batch=[fit[i] for i in idx];x=torch.stack([r['signal'] for r in batch]);y=torch.stack([r['target'] for r in batch])
        torch.manual_seed(2050);model=build_model(base['model'],num_classes=6)
        assert sum(v.numel() for v in model.parameters())==257511
        weak_counts=(fit.power_truth['weak_mask']&fit.labels.astype(bool)).sum(0)
        cfg={**base['train'],'loss':'weak_positive_bce' if method=='D' else 'bce'}
        loss_fn=build_loss(cfg,split['fit_label_counts'],len(fit),torch.device('cpu'),weak_counts)
        z=model(x);loss=(loss_fn(z,y,torch.stack([r['weak_mask'] for r in batch])) if isinstance(loss_fn,WeakPositiveBCE) else loss_fn(z,y))
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True)
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
        old=model.classifier[-1].weight.detach().clone();optimizer.step()
        assert torch.isfinite(loss) and not torch.equal(old,model.classifier[-1].weight)
        assert sampler.audit()['unique_fit_identities']==8000
        smoke.append({'method':method,'input_shape':list(x.shape),'loss':loss.item(),'gradient_norm':norm.item(),
                      'pool_size':len(fit),'epoch_draws':len(sampler)})
        fit.close()
    suite.mkdir(parents=True);report.mkdir(parents=True,exist_ok=True)
    jobs=[];references=[]
    for seed in SEEDS:
        for method in ['B','C','D']:
            cfg=json.loads(json.dumps(base));cfg['data']['manifest']=str(manifests[method])
            cfg['train'].update(seed=seed,loss='weak_positive_bce' if method=='D' else 'bce')
            ident=f'{method}_seed{seed}';cfg['run']['output_dir']=str(suite/ident);validate(cfg)
            config=suite/'configs'/f'{ident}.json';write_json(config,cfg)
            slot=len(jobs)%8
            jobs.append({'id':ident,'method':method,'seed':seed,'config':str(config),'output':cfg['run']['output_dir'],
                         'checkpoint':str(suite/ident/'best.pt'),'slot':slot,'cpus':slots[slot],'reference':False})
        for method,source in [('A','bce'),('FMB','combo_sampler')]:
            old=previous/f'{source}_seed{seed}';ident=f'{method}_seed{seed}';slot=len(references)%8
            references.append({'id':ident,'method':method,'seed':seed,'source_output':str(old),'checkpoint':str(old/'best.pt'),
                'checkpoint_sha256':sha256(old/'best.pt'),'output':str(suite/'references'/ident),
                'slot':slot,'cpus':slots[slot],'reference':True})
    evaluation_jobs=references+jobs
    for i,job in enumerate(evaluation_jobs):job['eval_slot']=i%8;job['eval_cpus']=slots[i%8]
    files=sorted((ROOT/'src').rglob('*.py'))+sorted((ROOT/'configs').glob('*.json'))+sorted((ROOT/'experiments').glob('*.py'))+sorted((ROOT/'tests').glob('*.py'))+[ROOT/'pyproject.toml',ROOT/'uv.lock']
    source=[]
    for path in files:
        relative=path.relative_to(ROOT);dest=suite/'source_snapshot'/relative;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(path,dest);source.append({'path':str(relative),'sha256':sha256(dest)})
    plan={'created':now(),'training_jobs':jobs,'references':references,'evaluation_jobs':evaluation_jobs,
        'source_manifest':source,'report_output':str(report),'data_root':str(data_root),
        'evaluation_manifest':str(evaluation_manifest),'context_split_sha256':context_split['sha256'],
        'model':base['model'],'dataloader':base['dataloader'],'data_audit':ready,'preflight':{'pytest_passed':75,'real_iq_steps':smoke},
        'protocol':{'methods':NAMES,'seeds':SEEDS,'new_models':15,'epochs':24,'draws_per_epoch':8000,'batch_size':32,
            'steps_per_job':6000,'total_new_optimizer_steps':90000,'total_epochs':360,'cpu_threads':4,'parallel_jobs':8,'slots':slots,
            'device':'cpu','amp':'none','model_parameters':257511,'optimizer':'AdamW','lr':1e-4,'weight_decay':1e-4,'scheduler':'24epoch cosine',
            'checkpoint_selection':'check Macro-AUPR for all new models, matching reused baseline',
            'training_sampler':'natural 8000 without replacement per epoch, independent of 32000 identity pool',
            'D_loss':'SawFM/MFT/ISRJ weak positives 1.5x; divide each label by fit-only expected weight mass',
            'FMB_rule':'fixed same-seed previous combo model at sigmoid0.5, identical across A/B/C/D',
            'other_labels_rule':'candidate per-label F1 thresholds fit only on unchanged calibration',
            'secondary_decision':'candidate empirical calibration FP caps matched to same-seed A F1 counts; FMB remains fixed0.5',
            'candidate_selection':'highest check mean fusion Exact among B/C/D satisfying check tail-F1 drop<=0.003 and max mean class-FPR rise<=0.01; must exceed A check Exact',
            'confirmation':'only selected-on-check candidate is primary; other contrasts exploratory',
            'success':{'minimum_exact_gain':.01,'minimum_positive_seeds':4,'paired95_interval_lower_above_zero':True,
                'maximum_tail_f1_drop':.003,'maximum_class_fpr_rise':.01},
            'report_gate':'all15 trainings complete; all25 checkpoint and decision hashes plus candidate frozen before any new report inference',
            'report_samples':4032,'calibration_samples':1008,'check_samples':1008,
            'report_distribution':'original independent10--30dB component JNR; conditional augmentation is fit-only',
            'generator':'v3 physical-equation Python PCG64 port, not bitwise MATLAB parity','power_truth_input':'training loss and evaluation only; inference only I/Q'},
        'estimated_wall_hours':[2,3]}
    write_json(suite/'context_split.json',context_split);write_json(suite/'plan.json',plan);write_json(suite/'preflight.json',plan['preflight'])
    prepare_references(suite,plan)
    text='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>弱分量CPU验证冻结方案</title>'+STYLE
    text+='<h1>弱分量CPU验证：A/B/C/D</h1><p>A复用现有融合；B保留原8000身份并增加24000新身份；C与B共享物理参数/噪声，仅改变目标难例分量功率；D与C文件完全相同，仅改变弱正例损失。所有方法固定使用相同旧FMB专家0.5决策，其余五类来自候选BCE分支。</p><p>15个新增模型，各5种子，24轮×8000抽样、6000步；8作业各4物理CPU核，预计预留2–3小时。阈值只拟合calibration，check选择一个主候选；模型、决策与主候选冻结后才评测4032个新报告身份。</p>'
    text+='<p>主决策为F1，辅助决策匹配A在calibration上的逐类误报计数。只在check选择主候选，报告主验收：Exact同池差≥1个百分点、至少4/5种子正向、条件95%区间下限>0、尾类F1下降≤0.3个百分点、最大逐类平均误报上升≤1个百分点。</p>'
    text+=table(['组','训练池','损失','新增训练'],[['A',8000,'BCE','复用'],['B',32000,'BCE',5],['C',32000,'BCE',5],['D',32000,'弱正例1.5倍、每类归一化',5]])
    text+='<p>75项测试及三方法真实I/Q优化器步通过。新报告仍为原独立10–30dB分布；条件弱难例仅用于fit。采用Python物理移植与PCG64，不声称MATLAB逐点复现。</p></html>'
    (report/'plan.html').write_text(text)
    print(json.dumps({'prepared':str(suite),'new_models':15,'references':10,'smoke':smoke},ensure_ascii=False),flush=True)


def get_predictions(suite,job,role,plan):
    import numpy as np
    from longtail.io import read_json,write_json,sha256
    from longtail.data.dataset import read_metadata
    dest=Path(job['output'])/'evaluation'/role;dest.mkdir(parents=True,exist_ok=True)
    file=dest/'predictions.npz';split=read_json(suite/'context_split.json');manifest=read_json(plan['evaluation_manifest'])
    if role=='development_test':
        frozen=read_json(suite/'frozen_ready.json')
        assert sha256(suite/'plan.json')==frozen['plan_sha256']
        hold=frozen['jobs'][job['id']]
        assert sha256(job['checkpoint'])==hold['checkpoint_sha256']
        assert sha256(Path(job['output'])/'decisions.json')==hold['decisions_sha256']
    if not file.exists():
        if job['reference'] and role!='development_test':
            origin=Path(job['source_output'])/'evaluation'/role/'predictions.npz'
            shutil.copyfile(origin,file)
        else:
            import torch
            from longtail.engine.checkpoint import load_checkpoint
            from longtail.models.factory import build_model
            from longtail.data.loaders import make_dataset,make_loader
            from longtail.engine.inference import predict
            payload,config=load_checkpoint(job['checkpoint'])
            assert config['model']==plan['model'] and config['dataloader']['preprocess']==plan['dataloader']['preprocess']
            assert payload['split']['classes']==manifest['classes']
            assert all(payload['split']['roles'][r]==split['roles'][r] for r in ['calibration','check','development_test'])
            model=build_model(config['model'],num_classes=6);model.load_state_dict(payload['model'],strict=True)
            dataset=make_dataset(manifest,Path(plan['evaluation_manifest']).parent,split,role,config['dataloader'])
            try:records=predict(model,make_loader(dataset,config['dataloader'],batch_size=64),torch.device('cpu'))
            finally:dataset.close()
            np.savez_compressed(file,**records)
    records=dict(np.load(file));y,ids,_=read_metadata(Path(plan['evaluation_manifest']).parent/manifest['files']['development']['path'])
    assert set(records['indices'].tolist())==set(split['roles'][role])
    np.testing.assert_array_equal(records['targets'],y[records['indices']])
    np.testing.assert_array_equal(records['identities'],ids[records['indices']])
    assert np.isfinite(records['logits']).all()
    write_json(dest/'source.json',{'checkpoint':job['checkpoint'],'checkpoint_sha256':sha256(job['checkpoint']),
        'context_manifest_sha256':sha256(plan['evaluation_manifest']),'context_split_sha256':split['sha256'],
        'role':role,'reused_saved_predictions':job['reference'] and role!='development_test','predictions_sha256':sha256(file)})
    return records


def fit_decisions(suite,job,plan):
    import numpy as np
    from longtail.io import read_json,write_json,sha256
    from longtail.metrics import fit_thresholds,probabilities
    from longtail.decision import fit_fpr_caps
    path=Path(job['output']);cal=dict(np.load(path/'evaluation/calibration/predictions.npz'))
    f1=fit_thresholds(cal['targets'],cal['logits'],None)
    baseline=suite/'references'/f'A_seed{job["seed"]}'
    ref=dict(np.load(baseline/'evaluation/calibration/predictions.npz'))
    ref_thresholds=fit_thresholds(ref['targets'],ref['logits'],None)
    caps=((probabilities(ref['logits'])>=ref_thresholds)&~ref['targets'].astype(bool)).sum(0)
    matched,meta=fit_fpr_caps(cal['targets'],cal['logits'],caps)
    write_json(path/'decisions.json',{'checkpoint_sha256':sha256(job['checkpoint']),'fit_role':'calibration',
        'calibration_predictions_sha256':sha256(path/'evaluation/calibration/predictions.npz'),
        'context_split_sha256':plan['context_split_sha256'],'thresholds':{'f1':f1,'matched_fpr':matched},
        'matched_fpr_fit':meta,'FMB_fusion_override':.5})


def prepare_references(suite,plan):
    from longtail.io import read_json,write_json,sha256
    for job in plan['references']:
        assert sha256(job['checkpoint'])==job['checkpoint_sha256']
        path=Path(job['output']);path.mkdir(parents=True,exist_ok=True);write_json(path/'reference.json',job)
        for role in ['calibration','check']:get_predictions(suite,job,role,plan)
    for job in plan['references']:fit_decisions(suite,job,plan)


def worker(suite,job_id,stage):
    from longtail.io import read_json,write_json,sha256
    plan=read_json(suite/'plan.json');job=next(j for j in plan['evaluation_jobs'] if j['id']==job_id)
    cpus=job['cpus'] if stage=='train' else job['eval_cpus']
    os.sched_setaffinity(0,set(cpus));os.nice(5)
    import torch
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    phase=suite/'phases'/f'{job_id}_{stage}.json';started=now()
    write_json(phase,{'status':'running','started':started,'pid':os.getpid(),'cpus':cpus})
    try:
        if stage=='train':
            from longtail.train import run
            output=Path(job['output']);status=read_json(output/'status.json') if (output/'status.json').exists() else {}
            if status.get('status')!='completed':
                run(read_json(job['config']),resume=output/'last.pt' if (output/'last.pt').exists() else None)
            for role in ['calibration','check']:get_predictions(suite,job,role,plan)
            fit_decisions(suite,job,plan)
        else:get_predictions(suite,job,'development_test',plan)
        write_json(phase,{'status':'completed','started':started,'finished':now(),'pid':os.getpid(),
            'checkpoint_sha256':sha256(job['checkpoint']),'decisions_sha256':sha256(Path(job['output'])/'decisions.json')})
    except BaseException as error:
        write_json(phase,{'status':'failed','started':started,'failed':now(),'error':repr(error)})
        raise


def launch_stage(suite,stage):
    from longtail.io import read_json,write_json
    plan=read_json(suite/'plan.json');jobs=plan['training_jobs'] if stage=='train' else plan['evaluation_jobs']
    slot_key='slot' if stage=='train' else 'eval_slot'
    queues={i:[j for j in jobs if j[slot_key]==i] for i in range(8)}
    active={};finished=set();failures=[];(suite/'logs').mkdir(exist_ok=True)
    def start(slot):
        if not queues[slot]:return
        job=queues[slot].pop(0);log=(suite/'logs'/f'{job["id"]}_{stage}.log').open('a')
        process=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--suite',str(suite),
            '--worker',job['id'],'--stage',stage],stdout=log,stderr=subprocess.STDOUT)
        active[slot]=(job,process,log)
    for slot in queues:start(slot)
    while active:
        for slot,(job,process,log) in list(active.items()):
            code=process.poll()
            if code is not None:
                log.close();del active[slot]
                if code:failures.append({'job':job['id'],'stage':stage,'exit_code':code})
                else:finished.add(job['id'])
                start(slot)
        rows=[]
        for job in jobs:
            row={'job':job['id'],'method':job['method'],'seed':job['seed'],'cpus':job['cpus'] if stage=='train' else job['eval_cpus']}
            file=Path(job['output'])/'history.json'
            if file.exists():
                h=read_json(file)[-1];row.update(epoch=h['epoch'],last_epoch_seconds=h['epoch_seconds'],check_aupr=h['validation']['macro_aupr'])
            process=next((p for j,p,_ in active.values() if j['id']==job['id']),None)
            row['state']='completed' if job['id'] in finished else 'failed' if any(f['job']==job['id'] for f in failures) else 'running' if process else 'queued'
            if process:row['pid']=process.pid
            rows.append(row)
        training_epochs=sum((read_json(Path(j['output'])/'history.json')[-1]['epoch'] if (Path(j['output'])/'history.json').exists() else 0) for j in plan['training_jobs'])
        write_json(suite/'progress.json',{'updated':now(),'stage':stage,'completed_jobs':len(finished),'total_jobs':len(jobs),
            'completed_epochs':training_epochs,'total_epochs':360,'failures':failures,'jobs':rows})
        if active:time.sleep(5)
    if failures:raise RuntimeError(str(failures))


def metric(records,logits,thresholds,plan,split):
    import numpy as np
    from longtail.metrics import evaluate_metrics,probabilities
    from longtail.weak_metrics import weak_metrics
    m=evaluate_metrics(records['targets'],logits,split['classes'],thresholds)
    w=weak_metrics(records['targets'],logits,split['classes'],thresholds,records['weak_mask'],records['relative_power_db'],split['fit_label_counts'])
    m['weak_component']=w;m['exact']=m['subset_accuracy']
    m['tail_f1']=float(np.mean([m['per_label'][c]['f1'] for c in ['AMA','SinFM']]))
    m['mean_fpr']=float(np.mean([v['negative_fpr'] for v in m['per_label'].values()]))
    m['weak_micro_recall']=w['weak_micro_recall']
    m['target_weak_recall']=float(np.mean([w['per_label'][c]['weak_recall'] for c in ['SawFM','MFT','ISRJ']]))
    m['target_weak_aupr']=float(np.mean([w['per_label'][c]['weak_aupr'] for c in ['SawFM','MFT','ISRJ']]))
    p=probabilities(logits)>=thresholds;y=records['targets'].astype(bool);strong=y&~records['weak_mask']
    m['nonweak_recall']=float(p[strong].mean())
    counts=np.array(split['fit_combination_counts']);rare=np.flatnonzero(counts==counts.min())+1
    code=y.astype(int)@(1<<np.arange(6));correct=(p==y).all(1)
    m['rare_exact']=float(correct[np.isin(code,rare)].mean())
    return m


def assemble(suite,role,plan):
    import numpy as np
    from longtail.io import read_json,write_json,sha256
    split=read_json(suite/'context_split.json')
    for seed in SEEDS:
        fjob=next(j for j in plan['references'] if j['id']==f'FMB_seed{seed}')
        fmb=dict(np.load(Path(fjob['output'])/'evaluation'/role/'predictions.npz'))
        for method in METHODS:
            job=next(j for j in plan['evaluation_jobs'] if j['id']==f'{method}_seed{seed}')
            path=Path(job['output']);records=dict(np.load(path/'evaluation'/role/'predictions.npz'))
            for k in ['targets','indices','identities']:np.testing.assert_array_equal(records[k],fmb[k])
            decisions=read_json(path/'decisions.json');fused=records['logits'].copy();fused[:,0]=fmb['logits'][:,0]
            metrics={'single':{},'fusion':{}}
            for d,t in decisions['thresholds'].items():
                metrics['single'][d]=metric(records,records['logits'],t,plan,split)
                threshold=t.copy();threshold[0]=.5
                metrics['fusion'][d]=metric(records,fused,threshold,plan,split)
            dest=suite/'views'/f'{method}_seed{seed}'/'evaluation'/role;dest.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(dest/'fusion_predictions.npz',**{**records,'logits':fused})
            write_json(dest/'metrics.json',{'role':role,'method':method,'seed':seed,
                'checkpoint_sha256':sha256(job['checkpoint']),'FMB_checkpoint_sha256':sha256(fjob['checkpoint']),
                'decisions_sha256':sha256(path/'decisions.json'),'metrics':metrics})


def select_candidate(suite,plan):
    import numpy as np
    from longtail.io import read_json,write_json
    means={}
    for method in METHODS:
        all_metrics=[read_json(suite/'views'/f'{method}_seed{s}/evaluation/check/metrics.json')['metrics']['fusion']['f1'] for s in SEEDS]
        means[method]={'exact':float(np.mean([m['exact'] for m in all_metrics])),
            'tail_f1':float(np.mean([m['tail_f1'] for m in all_metrics])),
            'class_fpr':np.mean([[m['per_label'][c]['negative_fpr'] for c in ['FMB','AMA','SawFM','SinFM','MFT','ISRJ']] for m in all_metrics],axis=0).tolist()}
    baseline=means['A'];eligible=[]
    for method in ['B','C','D']:
        m=means[method]
        if m['exact']>baseline['exact'] and m['tail_f1']>=baseline['tail_f1']-.003 and max(np.array(m['class_fpr'])-baseline['class_fpr'])<=.01:
            eligible.append(method)
    selected=max(eligible,key=lambda m:means[m]['exact']) if eligible else None
    result={'selected_on':'check only','selected_candidate':selected,'eligible':eligible,'check_means':means,
            'rule':plan['protocol']['candidate_selection'],'reporting_metrics_available_at_selection':False}
    write_json(suite/'selected_candidate.json',result);return result


def summarize(suite,plan):
    import numpy as np
    from scipy.stats import t
    from longtail.io import read_json,write_json,sha256
    frozen=read_json(suite/'frozen_ready.json');selected=read_json(suite/'selected_candidate.json')
    assert sha256(suite/'plan.json')==frozen['plan_sha256']
    assert sha256(suite/'selected_candidate.json')==frozen['selection_sha256']
    for job in plan['evaluation_jobs']:
        hold=frozen['jobs'][job['id']]
        assert sha256(job['checkpoint'])==hold['checkpoint_sha256']
        assert sha256(Path(job['output'])/'decisions.json')==hold['decisions_sha256']
    sampling={}
    for job in plan['training_jobs']:
        history=read_json(Path(job['output'])/'history.json')
        assert len(history)==24 and all(h['samples_seen']==8000 and h['sampler_padding']==0 and h['fit_pool_size']==32000 for h in history)
        assert all(h['sampling']['unique_fit_identities']==8000 and not h['sampling']['replacement'] for h in history)
        sampling[job['id']]=[{'epoch':h['epoch'],**h['sampling']} for h in history]
    rows=[]
    for role in ['calibration','check','development_test']:
        for method in METHODS:
            for seed in SEEDS:
                metrics=read_json(suite/'views'/f'{method}_seed{seed}/evaluation/{role}/metrics.json')['metrics']
                for view,ds in metrics.items():
                    for decision,m in ds.items():
                        rows.append({'method':method,'seed':seed,'role':role,'view':view,'decision':decision,**{k:m[k] for k in KEYS}})
    aggregate={role:{view:{decision:{method:{k:{'mean':float(np.mean([r[k] for r in rows if r['role']==role and r['view']==view and r['decision']==decision and r['method']==method])),
                'std':float(np.std([r[k] for r in rows if r['role']==role and r['view']==view and r['decision']==decision and r['method']==method],ddof=1))} for k in KEYS}
                for method in METHODS} for decision in ['f1','matched_fpr']} for view in ['single','fusion']}
                for role in ['calibration','check','development_test']}
    def values(method,decision,key):
        return np.array([next(r[key] for r in rows if r['role']=='development_test' and r['view']=='fusion' and r['method']==method and r['decision']==decision and r['seed']==seed) for seed in SEEDS])
    effects={}
    for decision in ['f1','matched_fpr']:
        effects[decision]={}
        for name,a,b in [('B_vs_A','B','A'),('C_vs_A','C','A'),('D_vs_A','D','A'),('C_vs_B','C','B'),('D_vs_C','D','C')]:
            effects[decision][name]={}
            for key in KEYS:
                delta=values(a,decision,key)-values(b,decision,key);radius=t.ppf(.975,4)*delta.std(ddof=1)/np.sqrt(5)
                effects[decision][name][key]={'mean':float(delta.mean()),'paired_seed_95_t_interval':[float(delta.mean()-radius),float(delta.mean()+radius)],'per_seed':delta.tolist()}
    candidate=selected['selected_candidate'];acceptance={'selected_candidate':candidate,'selected_on':'check','passed':False}
    if candidate:
        e=effects['f1'][candidate+'_vs_A'];limit=plan['protocol']['success']
        new=[read_json(suite/'views'/f'{candidate}_seed{s}/evaluation/development_test/metrics.json')['metrics']['fusion']['f1'] for s in SEEDS]
        base=[read_json(suite/'views'/f'A_seed{s}/evaluation/development_test/metrics.json')['metrics']['fusion']['f1'] for s in SEEDS]
        fpr_delta=np.mean([[n['per_label'][c]['negative_fpr']-b['per_label'][c]['negative_fpr'] for c in ['FMB','AMA','SawFM','SinFM','MFT','ISRJ']] for n,b in zip(new,base)],axis=0)
        conditions={'exact_gain_ge_1pp':e['exact']['mean']>=limit['minimum_exact_gain'],
            'positive_seeds_ge4':sum(v>0 for v in e['exact']['per_seed'])>=4,
            'paired_interval_lower_positive':e['exact']['paired_seed_95_t_interval'][0]>0,
            'tail_f1_drop_le_0.3pp':e['tail_f1']['mean']>=-limit['maximum_tail_f1_drop'],
            'class_fpr_rise_le_1pp':float(fpr_delta.max())<=limit['maximum_class_fpr_rise']}
        acceptance.update(conditions=conditions,passed=all(conditions.values()),mean_class_fpr_delta=fpr_delta.tolist(),primary_effect=e)
    out=Path(plan['report_output'])
    result={'created':now(),'plan':plan,'selection':selected,'acceptance':acceptance,'rows':rows,'aggregate':aggregate,'effects':effects,'sampling_audits':sampling,
        'validation':{'all15_jobs24epochs6000steps':True,'frozen_weights_decisions_selection':True,'fresh_report_deferred':True,'same_FMB_expert_all_methods':True},
        'limitations':'Fixed simulated data distribution and five initializations. Paired intervals condition on this pool. Only selected-on-check candidate is primary; other contrasts exploratory.'}
    write_json(out/'results.json',result)
    with (out/'results.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    text='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>弱分量CPU验证结果</title>'+STYLE
    text+='<h1>弱分量CPU验证：A/B/C/D新身份结果</h1><p>15个新模型各6000步；A与FMB专家复用已冻结模型。每组均使用同种子的FMB专家0.5输出，其余五类由该组模型预测。4032新报告身份在全部模型、决策和check候选冻结后测评。数值为五种子均值±标准差，百分数。</p>'
    text+=f'<p>check选中的主候选：{escape(str(candidate))}；预定义新报告验收通过：{acceptance["passed"]}。不根据报告结果改选主候选。</p>'
    for decision in ['f1','matched_fpr']:
        text+='<h2>新报告融合结果：'+decision+'</h2>'
        text+=table(['组']+KEYS,[[NAMES[m]]+[f"{v[k]['mean']*100:.3f} ± {v[k]['std']*100:.3f}" for k in KEYS] for m,v in aggregate['development_test']['fusion'][decision].items()])
    text+='<h2>主决策的配对归因</h2>'
    text+=table(['比较','Exact差 pp','条件95%配对区间 pp','弱目标召回差 pp'],[[name,f"{v['exact']['mean']*100:+.3f}",' ～ '.join(f'{x*100:+.3f}' for x in v['exact']['paired_seed_95_t_interval']),f"{v['target_weak_recall']['mean']*100:+.3f}"] for name,v in effects['f1'].items()])
    text+='<p>B−A衡量固定训练步数下扩大波形身份池的整体收益；C−B在成对物理参数与噪声下检验条件弱难例；D−C只改变训练损失。训练功率真值不作为推理输入。匹配FPR只在calibration满足经验预算，新报告实际FPR仍需同时检查。</p><p>单模型结果、逐类弱分量、功率差分档和全部条件在JSON与每个views目录中保留。新报告已消费，下一次调参需新身份确认；Python PCG64物理移植不代表MATLAB随机流或实测复现。</p>'
    text+='<p><a href="results.csv">全部240组逐种子CSV</a> · <a href="results.json">完整结果和验收</a> · <a href="plan.html">冻结方案</a></p></html>'
    (out/'report.html').write_text(text)


def pipeline(suite):
    from longtail.io import read_json,write_json,sha256
    import longtail
    assert Path(longtail.__file__).resolve().is_relative_to(suite/'source_snapshot/src')
    if (suite/'pipeline_completed.json').exists():raise FileExistsError('Suite already completed')
    plan=read_json(suite/'plan.json');start=time.time()
    for item in plan['source_manifest']:assert sha256(suite/'source_snapshot'/item['path'])==item['sha256']
    for job in plan['references']:assert sha256(job['checkpoint'])==job['checkpoint_sha256']
    write_json(suite/'pipeline.json',{'status':'running','stage':'training','started':now(),'pid':os.getpid()})
    try:
        launch_stage(suite,'train')
        for role in ['calibration','check']:assemble(suite,role,plan)
        if not (suite/'selected_candidate.json').exists():select_candidate(suite,plan)
        held={j['id']:{'checkpoint_sha256':sha256(j['checkpoint']),'decisions_sha256':sha256(Path(j['output'])/'decisions.json')} for j in plan['evaluation_jobs']}
        frozen={'finished':now(),'jobs':held,'plan_sha256':sha256(suite/'plan.json'),'selection_sha256':sha256(suite/'selected_candidate.json')}
        if (suite/'frozen_ready.json').exists():
            original=read_json(suite/'frozen_ready.json');assert original['jobs']==held and original['selection_sha256']==frozen['selection_sha256'];frozen=original
        else:
            write_json(suite/'frozen_ready.json',frozen);write_json(suite/'all_training_completed.json',{'finished':frozen['finished'],'jobs':15})
        write_json(suite/'pipeline.json',{'status':'running','stage':'fresh_report','updated':now(),'pid':os.getpid()})
        launch_stage(suite,'report');assemble(suite,'development_test',plan);summarize(suite,plan)
        done={'status':'completed','finished':now(),'wall_seconds':time.time()-start,'new_models_completed':15,'evaluated_models':25,'report':str(Path(plan['report_output'])/'report.html')}
        write_json(suite/'pipeline.json',done);write_json(suite/'pipeline_completed.json',done)
    except BaseException as error:
        write_json(suite/'pipeline.json',{'status':'failed','failed':now(),'error':repr(error),'pid':os.getpid()})
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--suite',required=True,type=Path)
    p.add_argument('--report',type=Path);p.add_argument('--data-root',type=Path);p.add_argument('--previous',type=Path)
    p.add_argument('--prepare',action='store_true');p.add_argument('--run',action='store_true');p.add_argument('--worker');p.add_argument('--stage',choices=['train','report'])
    a=p.parse_args();suite=a.suite.resolve()
    if a.prepare:prepare(suite,a.report.resolve(),a.data_root.resolve(),a.previous.resolve())
    elif a.worker:worker(suite,a.worker,a.stage)
    elif a.run:pipeline(suite)
    else:p.error('Choose prepare or run')
