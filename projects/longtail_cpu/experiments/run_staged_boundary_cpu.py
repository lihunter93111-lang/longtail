"""Run the ordered three-stage validation from a frozen source snapshot."""
import argparse
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
SLOTS=[list(range(8+4*i,12+4*i)) for i in range(4)]+[list(range(40+4*i,44+4*i)) for i in range(4)]
KEYS=['exact','macro_f1','macro_aupr','tail_f1','mean_fpr','saw_weak_recall','target_weak_recall','weak_micro_recall','nonweak_recall']
STYLE='<style>body{font:16px/1.8 system-ui;max-width:1200px;margin:30px auto;padding:24px;color:#234}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:8px;border:1px solid #ccd}th{background:#e7f2f3}code{overflow-wrap:anywhere}</style>'


def now():return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')


def table(headers,rows):
    return '<div style="overflow-x:auto"><table><tr>'+''.join('<th>'+escape(str(v))+'</th>' for v in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+escape(str(v))+'</td>' for v in row)+'</tr>' for row in rows)+'</table></div>'


def prepare(suite,data_root,report):
    import torch
    import numpy as np
    import h5py
    from longtail.io import read_json,write_json,sha256
    from longtail.data.splits import build_split
    from longtail.models.decision_heads import FiveLabelHead,LocalResidual,residual_logits
    from longtail.engine.checkpoint import load_checkpoint
    ready=read_json(data_root/'generation_complete.json');assert ready['status']=='completed'
    if suite.exists():raise FileExistsError('Use a new suite')
    assert set(sum(SLOTS,[]))<=os.sched_getaffinity(0)
    refs={}
    for seed in SEEDS:
        refs[str(seed)]={}
        for kind,path in [('B',ROOT/f'runs/weak_component_validation_cpu_20261103/B_seed{seed}/best.pt'),
                          ('A',ROOT/f'runs/combo_optimization_cpu_20261103/bce_seed{seed}/best.pt'),
                          ('FMB',ROOT/f'runs/combo_optimization_cpu_20261103/combo_sampler_seed{seed}/best.pt')]:
            payload,cfg=load_checkpoint(path);refs[str(seed)][kind]={'checkpoint':str(path),'sha256':sha256(path)}
            assert cfg['model']['entrypoint']=='longtail.models.dfms:build_model'
    suite.mkdir(parents=True);report.mkdir(parents=True,exist_ok=True)
    files=sorted((ROOT/'src').rglob('*.py'))+sorted((ROOT/'experiments').glob('*.py'))+sorted((ROOT/'tests').glob('*.py'))+sorted((ROOT/'configs').glob('*.json'))+[ROOT/'pyproject.toml',ROOT/'uv.lock']
    source=[]
    for file in files:
        dest=suite/'source_snapshot'/file.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(file,dest)
        source.append({'path':str(file.relative_to(ROOT)),'sha256':sha256(dest)})
    data_hashes={};contexts={}
    for stage,path in ready['stage_manifests'].items():
        manifest=read_json(path);_,cfg=load_checkpoint(refs['2050']['B']['checkpoint'])
        split=build_split(manifest,Path(path).parent,cfg['data']);write_json(suite/f'split_stage{stage}.json',split);contexts[stage]=split['sha256']
        data_hashes[path]=sha256(path)
        for f in manifest['files'].values():data_hashes[str(Path(path).parent/f['path'])]=f['sha256']
    pairs=read_json(ready['pair_manifest']);data_hashes[ready['pair_manifest']]=sha256(ready['pair_manifest'])
    edge_sets={role:{v['edge_id'] for v in pairs['edges'] if v['role']==role} for role in ['fit','check','development_test']}
    assert not edge_sets['fit']&edge_sets['check'] and not edge_sets['fit']&edge_sets['development_test'] and not edge_sets['check']&edge_sets['development_test']
    for info in pairs['files'].values():assert sha256(info['path'])==info['sha256'];data_hashes[info['path']]=info['sha256']
    # Cache-free mathematical checks; full real-I/Q smoke is performed before launch.
    torch.manual_seed(2050);head=FiveLabelHead(torch.randn(6,128),torch.randn(6));x=torch.randn(32,128);base=torch.randn(32,6)
    local=LocalResidual();assert torch.equal(residual_logits(base,local(x)),base)
    opt=torch.optim.AdamW(head.parameters(),lr=.001);loss=torch.nn.functional.binary_cross_entropy_with_logits(head(x),torch.randint(0,2,(32,5)).float())
    loss.backward();opt.step();assert all(torch.isfinite(p).all() for p in head.parameters())
    protocol={'seeds':SEEDS,'device':'CPU only','max_parallel_jobs':8,'threads_per_inference_job':4,'threads_per_cached_head_training':1,
        'fixed_B':'natural32000 BCE best checkpoint per seed','fixed_FMB':'previous combo expert sigmoid0.5, unchanged all stages',
        'stage1':{'rules':['f1','matched_fpr','exact_fpr','exact_weak'],'training':'none; same frozen logits',
                  'exact_fpr':'calibration Exact, FP<=A-F1+1pp, overall/target weak recall>=matched-FPR minus1pp',
                  'exact_weak':'same FP cap, recall and target weak recall>=B-F1 minus3pp',
                  'selection':'check maximum mean Exact subject to class FPR<=A-F1+1pp, tail-F1>=matched-FPR-0.3pp, SawFM weak recall>=B-F1-3pp, Exact>=matched-FPR-0.2pp; default matched-FPR if none'},
        'stage2':{'methods':['base','head_natural','head_mixed'],'new_models':10,'trainable_parameters':645,'optimizer':'AdamW',
                  'lr':.001,'weight_decay':.0001,'epochs':24,'steps_per_epoch':250,'batch_size':32,'steps_per_model':6000,
                  'mixed':'4000 natural without replacement +4000 combo-balanced with replacement per epoch',
                  'checkpoint':'check five-label Macro-AUPR','backbone_mode':'eval and frozen; cached128-dimensional features'},
        'stage3':{'methods':['base','local_natural','local_pair_bce','local_pair_margin'],'new_models_if_gate_passes':15,
                  'steps_per_model':6000,'epochs':24,'batch_size':32,'lr':.001,'weight_decay':.0001,
                  'local_head':'128->32 GELU->3 zero-initialized residuals for SawFM/MFT/ISRJ; previous backbone/head frozen',
                  'paired_batch':'24 natural views+4 one-bit pairs=32 views; natural control32 views; fixed192000 views/model',
                  'margin':'BCE +0.1*relu(0.5-(target_logit_positive-target_logit_negative))',
                  'gate':'per target max(physical single-view logistic-probe check AUROC, mean frozen-model check AUROC)>=0.60; failed target disabled; no active target skips local training',
                  'FMB_pairs':'observability control only, FMB expert remains frozen',
                  'probe_fit':'pair fit only; probes see one signal, never pair differences or latent components',
                  'edge_holdout':'19 fit/6 check/6 report edges per four diagnostic targets; pretrained natural backbone has seen all63 combinations'},
        'stage2_3_selection':'check best mean Exact above current-stage base, tail-F1 drop<=0.3pp, class FPR<=A-F1+1pp, SawFM weak recall>=B-F1-3pp; otherwise retain base',
        'next_stage_choice':'previous stage check selection only; report values never drive later choices',
        'calibration_rule_infeasible_on_new_pool':'record fallback to matched-FPR',
        'success':{'exact_gain':.005,'positive_seeds':4,'paired95_lower_positive':True,'max_tail_drop':.003,'max_class_fpr_rise':.01,'max_saw_weak_drop':.03},
        'data_roles_per_stage':{'fit':32000,'calibration':1008,'check':1008,'new_report':4032},
        'report_gate':'freeze source weights, trained heads, calibration decisions and check selection before each stage report',
        'generator':'original independent10--30dB v3 physics, Python PCG64; not bitwise MATLAB parity',
        'new_training_steps':150000,'limitations':'Five initializations on fixed simulated pools; exploratory secondary comparisons have no multiplicity correction.'}
    plan={'created':now(),'references':refs,'data':ready,'data_hashes':data_hashes,'contexts':contexts,'source_manifest':source,
          'protocol':protocol,'slots':SLOTS,'report_output':str(report),'data_root':str(data_root)}
    write_json(suite/'plan.json',plan)
    html='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>三阶段边界验证方案</title>'+STYLE+'<h1>按顺序验证：决策→分类头→one-bit局部头</h1><p>复用B自然32k backbone与FMB组合专家，五种子。每阶段均重新生成calibration/check/report；后续阶段仅采用前阶段check选择，不读取report做决策。</p>'
    html+=table(['阶段','对照','新增训练'],[['1 决策','F1 /误报预算/误报约束Exact/弱召回约束Exact',0],['2 冻结特征分类头','原头/自然重训/温和组合重训',10],['3 one-bit局部头','原模型/自然局部头/配对BCE/配对边界损失','通过可观测性门控后15']])
    html+='<p>训练部分每模型6000步、192000个样本曝光。分类头645参数；局部残差仅作用SawFM/MFT/ISRJ。FMB配对只用于诊断。原始配对差分用于生成审计；检测器与最终模型只接收单条信号。</p><p>第一阶段候选由check Exact、误报和SawFM弱召回共同选择；后续分类头/局部头没有合格增益则保留基线。各阶段报告前冻结全部权重、阈值和选择。正式成功要求同池Exact增益≥0.5pp、4/5种子正向、配对95%区间下限>0，并满足尾类/FPR/SawFM弱召回门槛。约束只在拟合角色保证，报告仍独立检验。</p><pre>'+escape(json.dumps(protocol,ensure_ascii=False,indent=2))+'</pre></html>'
    (report/'plan.html').write_text(html);print(json.dumps({'prepared':str(suite),'conditional_new_models':25,'data':ready},ensure_ascii=False),flush=True)


def worker(suite,action,seed,slot):
    from longtail.io import read_json,write_json
    from longtail import staged_training as training
    import torch
    plan=read_json(suite/'plan.json');os.sched_setaffinity(0,set(plan['slots'][slot]));os.nice(5)
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    phase=suite/'phases'/f'{action}_seed{seed}.json'
    if phase.exists() and read_json(phase)['status']=='completed':return
    started=now();write_json(phase,{'status':'running','started':started,'pid':os.getpid(),'cpus':plan['slots'][slot]})
    try:
        if action=='stage1_dev':training.phase1(suite,seed)
        elif action=='stage1_report':training.phase1(suite,seed,True)
        elif action=='stage2_dev':training.phase2(suite,seed)
        elif action=='stage2_report':training.phase2(suite,seed,True)
        elif action=='stage3_probe':training.probe_cache(suite,seed)
        elif action=='stage3_dev':training.phase3(suite,seed)
        elif action=='stage3_report':training.phase3(suite,seed,True)
        else:raise ValueError(action)
        write_json(phase,{'status':'completed','started':started,'finished':now(),'pid':os.getpid()})
    except BaseException as error:
        write_json(phase,{'status':'failed','started':started,'failed':now(),'error':repr(error)});raise


def launch(suite,action):
    from longtail.io import read_json,write_json
    (suite/'logs').mkdir(exist_ok=True);jobs=[]
    for slot,seed in enumerate(SEEDS):
        log=(suite/'logs'/f'{action}_seed{seed}.log').open('a')
        proc=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--suite',str(suite),'--worker',action,'--seed',str(seed),'--slot',str(slot)],stdout=log,stderr=subprocess.STDOUT)
        jobs.append((seed,proc,log))
    while True:
        rows=[];active=0;failures=[]
        for seed,proc,log in jobs:
            code=proc.poll();phase=suite/'phases'/f'{action}_seed{seed}.json'
            row={'seed':seed,'pid':proc.pid,'state':'running' if code is None else 'completed' if code==0 else 'failed'}
            if phase.exists():row['phase']=read_json(phase)
            if code is None:active+=1
            elif code:failures.append({'seed':seed,'exit_code':code})
            rows.append(row)
        write_json(suite/'progress.json',{'updated':now(),'action':action,'active_jobs':active,'completed_jobs':sum(r['state']=='completed' for r in rows),'total_jobs':5,'failures':failures,'jobs':rows})
        if not active:break
        time.sleep(3)
    for _,_,log in jobs:log.close()
    if failures:raise RuntimeError(str(failures))


def stage_metrics(suite,stage,role):
    from longtail.io import read_json
    return [read_json(suite/f'stage{stage}'/f'seed{seed}'/f'metrics_{role}.json')['metrics'] for seed in SEEDS]


def select(suite,stage):
    import numpy as np
    from longtail.io import write_json
    data=stage_metrics(suite,stage,'check');available=sorted(set.intersection(*[set(row) for row in data]))
    baseline='matched_fpr' if stage==1 else 'base';guard='f1' if stage==1 else 'B_f1_guard'
    names=[x for x in available if x not in ['A_f1','B_f1_guard']]
    means={m:{k:float(np.mean([r[m][k] for r in data])) for k in KEYS} for m in available}
    eligibility={}
    for name in names:
        fpr={c:float(np.mean([r[name]['per_label'][c]['negative_fpr']-r['A_f1']['per_label'][c]['negative_fpr'] for r in data])) for c in ['FMB','AMA','SawFM','SinFM','MFT','ISRJ']}
        criteria={'tail_f1':means[name]['tail_f1']>=means[baseline]['tail_f1']-.003,
                  'class_fpr':max(fpr.values())<=.01,'saw_weak':means[name]['saw_weak_recall']>=means[guard]['saw_weak_recall']-.03,
                  'exact':means[name]['exact']>=means[baseline]['exact']-.002 if stage==1 else means[name]['exact']>means[baseline]['exact']}
        eligibility[name]={'eligible':all(criteria.values()),'criteria':criteria,'class_fpr_delta_vs_A':fpr}
    eligible=[m for m in names if eligibility[m]['eligible']]
    selected=max(eligible,key=lambda m:(means[m]['exact'],m==baseline)) if eligible else baseline
    decision={'stage':stage,'selected':selected,'selected_on':'check only','selected_at':now(),'baseline':baseline,'eligible':eligible,
              'fallback_no_eligible':not eligible,'check_means':means,'eligibility':eligibility,'report_values_used':False}
    write_json(suite/f'stage{stage}'/'selection.json',decision);return decision


def freeze(suite,stage):
    from longtail.io import read_json,write_json,sha256
    plan=read_json(suite/'plan.json');files={}
    for refs in plan['references'].values():
        for v in refs.values():assert sha256(v['checkpoint'])==v['sha256'];files[v['checkpoint']]=v['sha256']
    for earlier in range(1,stage):
        held=read_json(suite/f'stage{earlier}'/'frozen.json')
        for file,digest in held['files'].items():assert sha256(file)==digest
    for st in range(1,stage+1):
        for path in (suite/f'stage{st}').rglob('*'):
            if path.name=='decisions.json' or path.suffix=='.pt' and not path.name.startswith('last_') or path.name in ['selection.json','observability.json']:
                files[str(path)]=sha256(path)
    for path in (suite/'cache').rglob('*.npz'):
        if 'development_test' not in path.name:files[str(path)]=sha256(path)
    frozen={'stage':stage,'frozen_at':now(),'plan_sha256':sha256(suite/'plan.json'),'selection_sha256':sha256(suite/f'stage{stage}'/'selection.json'),'files':files}
    write_json(suite/f'stage{stage}'/'frozen.json',frozen)


def observability(suite):
    import numpy as np
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from longtail.io import read_json,write_json
    from longtail.staged_training import head_logits,previous_head
    fit=dict(np.load(suite/'cache/seed2050/pair_fit.npz'));check=dict(np.load(suite/'cache/seed2050/pair_check.npz'))
    rows={};enabled=[]
    for c in [0,2,4,5]:
        train=fit['pair_target']==c;test=check['pair_target']==c
        probe=make_pipeline(StandardScaler(),LogisticRegression(C=1.,max_iter=1000,random_state=2050))
        probe.fit(fit['physical_descriptor'][train],fit['targets'][train,c])
        raw_auc=float(roc_auc_score(check['targets'][test,c],probe.decision_function(check['physical_descriptor'][test])))
        frozen=[]
        for seed in SEEDS:
            records=dict(np.load(suite/'cache'/f'seed{seed}'/'pair_check.npz'));mask=records['pair_target']==c
            z=records['FMB_logits'][:,0] if c==0 else head_logits(records,previous_head(suite,seed))[:,c]
            frozen.append(float(roc_auc_score(records['targets'][mask,c],z[mask])))
        passed=max(raw_auc,float(np.mean(frozen)))>=.60
        if c in [2,4,5] and passed:enabled.append(c)
        rows[str(c)]={'physical_single_view_probe_auroc':raw_auc,'frozen_single_view_auroc_per_seed':frozen,'passed':passed,'fit_edges':19,'check_edges':6}
    audit={'created':now(),'enabled_targets':enabled,'per_target':rows,'status':'passed' if enabled else 'no_target_passed_skip_local_training',
        'fit_role':'pair fit only','gate_role':'pair check only','input':'single-view amplitude-normalized spectral/autocorrelation features or frozen I/Q model',
        'not_an_information_theoretic_claim':True,'differences_and_clean_components_not_probe_inputs':True,'pair_report_opened':False}
    write_json(suite/'stage3/observability.json',audit);return audit


def summarize(suite,stage):
    import numpy as np
    from scipy.stats import t
    from longtail.io import read_json,write_json
    plan=read_json(suite/'plan.json');choice=read_json(suite/f'stage{stage}/selection.json');data=stage_metrics(suite,stage,'development_test')
    names=sorted(set.intersection(*[set(r) for r in data]));aggregate={m:{k:{'mean':float(np.mean([r[m][k] for r in data])),'std':float(np.std([r[m][k] for r in data],ddof=1))} for k in KEYS} for m in names}
    effects={};baseline=choice['baseline']
    for method in names:
        if method in ['A_f1','B_f1_guard']:continue
        effects[method]={}
        for key in KEYS:
            delta=np.array([r[method][key]-r[baseline][key] for r in data]);radius=t.ppf(.975,4)*delta.std(ddof=1)/np.sqrt(5)
            effects[method][key]={'mean':float(delta.mean()),'paired95':[float(delta.mean()-radius),float(delta.mean()+radius)],'per_seed':delta.tolist()}
    selected=choice['selected'];d=effects[selected];limits=plan['protocol']['success']
    classes=['FMB','AMA','SawFM','SinFM','MFT','ISRJ']
    fpr={c:float(np.mean([r[selected]['per_label'][c]['negative_fpr']-r[baseline]['per_label'][c]['negative_fpr'] for r in data])) for c in classes}
    conditions={'exact_gain_ge0.5pp':d['exact']['mean']>=limits['exact_gain'],'positive_seeds_ge4':sum(v>0 for v in d['exact']['per_seed'])>=4,
        'paired_interval_lower_positive':d['exact']['paired95'][0]>0,'tail_f1_drop_le0.3pp':d['tail_f1']['mean']>=-limits['max_tail_drop'],
        'class_fpr_rise_le1pp':max(fpr.values())<=limits['max_class_fpr_rise'],'saw_weak_recall_drop_le3pp':d['saw_weak_recall']['mean']>=-limits['max_saw_weak_drop']}
    result={'stage':stage,'finished':now(),'selection':choice,'aggregate':aggregate,'effects_vs_stage_baseline':effects,
        'acceptance':{'selected':selected,'baseline':baseline,'passed':all(conditions.values()),'conditions':conditions,'class_fpr_delta':fpr},
        'per_seed_metrics':data,'limitations':plan['protocol']['limitations']}
    write_json(suite/f'stage{stage}/results.json',result);out=Path(plan['report_output']);write_json(out/f'stage{stage}_results.json',result)
    html='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>三阶段验证结果</title>'+STYLE+f'<h1>阶段{stage}：新身份报告结果</h1>'
    html+=f'<p>check选中：{escape(selected)}；同池参考：{escape(baseline)}；预定义验收通过：{all(conditions.values())}。五种子均值，百分数；本阶段报告不用于下一阶段选择。</p>'
    html+=table(['方法']+KEYS,[[m]+[f"{v[k]['mean']*100:.3f} ± {v[k]['std']*100:.3f}" for k in KEYS] for m,v in aggregate.items()])
    html+=table(['条件','通过'],[[k,v] for k,v in conditions.items()])+'<p>逐类与逐种子结果见同名JSON；配对区间以固定模拟池为条件，未做多重比较校正。</p></html>'
    (out/f'stage{stage}.html').write_text(html)


def pipeline(suite):
    from longtail.io import read_json,write_json,sha256
    import longtail
    assert Path(longtail.__file__).resolve().is_relative_to(suite/'source_snapshot/src')
    plan=read_json(suite/'plan.json');start=time.time()
    for item in plan['source_manifest']:assert sha256(suite/'source_snapshot'/item['path'])==item['sha256']
    for file,digest in plan['data_hashes'].items():assert sha256(file)==digest
    if (suite/'pipeline_completed.json').exists():raise FileExistsError('Pipeline already completed')
    write_json(suite/'pipeline.json',{'status':'running','started':now(),'pid':os.getpid()})
    try:
        for stage in range(1,4):
            if (suite/f'stage{stage}'/'results.json').exists():continue
            (suite/f'stage{stage}').mkdir(exist_ok=True)
            write_json(suite/'pipeline.json',{'status':'running','stage':stage,'updated':now(),'pid':os.getpid()})
            if stage==3:
                launch(suite,'stage3_probe');observability(suite)
            launch(suite,f'stage{stage}_dev');select(suite,stage);freeze(suite,stage)
            launch(suite,f'stage{stage}_report');summarize(suite,stage)
        done={'status':'completed','started':read_json(suite/'launch.json')['started'],'finished':now(),'wall_seconds':time.time()-start,
              'stages_completed':3,'new_heads':10,'new_local_models':15 if read_json(suite/'stage3/observability.json')['enabled_targets'] else 0,
              'report_output':plan['report_output']}
        write_json(suite/'pipeline.json',done);write_json(suite/'pipeline_completed.json',done)
        out=Path(plan['report_output']);(out/'report.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8">'+STYLE+'<h1>三阶段训练和测试已完成</h1><p>各阶段均使用独立的新开发与报告身份。后阶段只采用前阶段check选择。请结合验收状态阅读各阶段结果。</p>'+''.join(f'<p><a href="stage{i}.html">阶段{i}报告</a></p>' for i in range(1,4))+'</html>')
    except BaseException as error:
        write_json(suite/'pipeline.json',{'status':'failed','failed':now(),'error':repr(error),'pid':os.getpid()});raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--suite',type=Path,required=True);p.add_argument('--data-root',type=Path);p.add_argument('--report',type=Path)
    p.add_argument('--prepare',action='store_true');p.add_argument('--run',action='store_true');p.add_argument('--worker');p.add_argument('--seed',type=int);p.add_argument('--slot',type=int,default=0)
    a=p.parse_args();suite=a.suite.resolve()
    if a.prepare:prepare(suite,a.data_root.resolve(),a.report.resolve())
    elif a.worker:worker(suite,a.worker,a.seed,a.slot)
    elif a.run:pipeline(suite)
    else:p.error('Choose prepare/run/worker')
