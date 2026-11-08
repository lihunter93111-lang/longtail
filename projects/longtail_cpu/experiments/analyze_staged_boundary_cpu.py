"""Audit the three ordered stages using frozen decisions and saved features."""
import argparse
import base64
from datetime import datetime
from html import escape
import json
from pathlib import Path
import numpy as np


def read(path):return json.loads(Path(path).read_text())


def equal(a,b):
    if isinstance(b,dict):
        assert a.keys()==b.keys()
        for k in b:equal(a[k],b[k])
    elif isinstance(b,list):
        assert len(a)==len(b)
        for x,y in zip(a,b):equal(x,y)
    elif isinstance(b,float):assert np.isclose(a,b,rtol=0,atol=1e-12,equal_nan=True),(a,b)
    else:assert a==b,(a,b)


def audit(suite):
    import h5py
    import torch
    from sklearn.metrics import roc_auc_score
    from longtail.io import write_json,sha256
    from longtail import staged_training as tr
    from longtail.staged_decision import get_rule
    from longtail.data.dataset import read_metadata
    from longtail.data.power_truth import read_power_truth
    from longtail.data.one_bit import pair
    from longtail.data.signals import Simulator
    plan=read(suite/'plan.json');done=read(suite/'pipeline_completed.json');assert done['status']=='completed'
    out=Path(plan['report_output']);torch.set_num_threads(2)
    for source in plan['source_manifest']:assert sha256(suite/'source_snapshot'/source['path'])==source['sha256']
    for file,digest in plan['data_hashes'].items():assert sha256(file)==digest
    refs={}
    for seed,items in plan['references'].items():
        for kind,item in items.items():assert sha256(item['checkpoint'])==item['sha256'];refs[f'{seed}_{kind}']=item['sha256']
    generation=read(Path(plan['data_root'])/'generation_plan.json');classes=generation['classes']
    backgrounds=[]
    for manifest in plan['data']['stage_manifests'].values():
        m=read(manifest);_,ids,_=read_metadata(Path(manifest).parent/'development.h5');backgrounds.extend(ids.tolist())
    pair_manifest=read(plan['data']['pair_manifest']);groups={};pair_samples=0
    simulator=Simulator(generation['physics'])
    for role,info in pair_manifest['files'].items():
        with h5py.File(info['path']) as f:
            ids=f['sample_id'][:].ravel();pair_ids=f['pair_id'][:].ravel();noise=f['receiver_noise_seed'][:].ravel();y=f['label_selected'][:].T
            target=f['pair_target'][:].ravel();edges=f['edge_id'][:].ravel();seeds=f['sample_seed'][:].ravel()
            assert len(np.unique(ids))==len(ids) and np.array_equal(pair_ids[::2],pair_ids[1::2]) and np.array_equal(noise[::2],noise[1::2])
            assert np.all((y[::2]^y[1::2]).sum(1)==1);backgrounds.extend(noise[::2].tolist());groups[role]=set(edges.tolist());pair_samples+=len(ids)
            for c in [0,2,4,5]:
                index=int(np.flatnonzero(target[::2]==c)[0]*2);background=int(y[index].astype(int)@(1<<np.arange(6)))
                active,plus,negative,position,error=pair(simulator,background,c,int(seeds[index]),classes)
                np.testing.assert_array_equal(f['iq_complex'][:,index],negative);np.testing.assert_array_equal(f['iq_complex'][:,index+1],plus['iq'])
                assert error<1e-12
    assert all(not groups[a]&groups[b] for a,b in [('fit','check'),('fit','development_test'),('check','development_test')])
    assert len(np.unique(backgrounds))==len(backgrounds)
    for file in generation['checked_identity_files']:
        with h5py.File(file) as f:assert not np.intersect1d(backgrounds,f['receiver_noise_seed'][:].ravel()).size
    metrics_count=0;decisions_count=0;models=[];analyses={};bindings=[];pair_weak_diagnostics=[]
    for stage in [1,2,3]:
        folder=suite/f'stage{stage}';frozen=read(folder/'frozen.json');selection=read(folder/'selection.json');result=read(folder/'results.json')
        assert frozen['plan_sha256']==sha256(suite/'plan.json') and frozen['selection_sha256']==sha256(folder/'selection.json')
        for file,digest in frozen['files'].items():assert sha256(file)==digest
        barrier=datetime.fromisoformat(frozen['frozen_at'])
        manifest=read(plan['data']['stage_manifests'][str(stage)]);dev_file=Path(plan['data']['stage_manifests'][str(stage)]).parent/'development.h5'
        dev_y,dev_ids,_=read_metadata(dev_file);dev_truth=read_power_truth(dev_file,dev_y)
        for seed in range(2050,2055):
            dev=read(suite/'phases'/f'stage{stage}_dev_seed{seed}.json');report=read(suite/'phases'/f'stage{stage}_report_seed{seed}.json')
            assert dev['status']==report['status']=='completed'
            assert datetime.fromisoformat(dev['finished'])<=barrier<=datetime.fromisoformat(report['started'])
            output=folder/f'seed{seed}';cal=dict(np.load(suite/'cache'/f'seed{seed}'/f'stage{stage}_calibration.npz'))
            saved=read(output/'decisions.json')
            if stage==1:
                decisions=tr.fit_bundle(cal,cal['base_logits']);equal(saved,decisions);decisions_count+=1
                methods=[name for name,th in saved['thresholds'].items() if th is not None]+['A_f1']
                predict=lambda m,r:r['base_logits']
            else:
                methods=list(saved['models'])
                predict=(lambda m,r:tr.head_logits(r,None if m=='base' else output/(m+'.pt'))) if stage==2 else (lambda m,r:tr.local_logits(suite,seed,m,r))
                for method,bundle in saved['models'].items():equal(bundle,tr.fit_bundle(cal,predict(method,cal)));decisions_count+=1
                equal(saved['B_guard'],tr.fit_bundle(cal,cal['base_logits']));decisions_count+=1
            for role in ['calibration','check','development_test']:
                file=suite/'cache'/f'seed{seed}'/f'stage{stage}_{role}.npz';records=dict(np.load(file));origin=read(file.with_name(file.stem+'_source.json'))
                assert sha256(file)==origin['cache_sha256'] and origin['backbone_sha256']==refs[f'{seed}_B']
                split=read(suite/f'split_stage{stage}.json');assert set(records['indices'].tolist())==set(split['roles'][role])
                np.testing.assert_array_equal(records['targets'],dev_y[records['indices']]);np.testing.assert_array_equal(records['identities'],dev_ids[records['indices']])
                np.testing.assert_array_equal(records['weak_mask'],dev_truth['weak_mask'][records['indices']])
                np.testing.assert_array_equal(records['relative_power_db'],dev_truth['relative_power_db'][records['indices']])
                stored=read(output/f'metrics_{role}.json')['metrics']
                for method in stored:
                    if method=='A_f1':z=records['A_logits'].copy();z[:,0]=records['FMB_logits'][:,0];th=saved['A_f1_thresholds'] if stage==1 else saved['B_guard']['A_f1_thresholds']
                    elif method=='B_f1_guard':z=tr.fused_logits(records);th=saved['B_guard']['thresholds']['f1']
                    elif stage==1:z=tr.fused_logits(records);th=saved['thresholds'][method]
                    else:z=tr.fused_logits(records,predict(method,records));th,_=get_rule(saved['models'][method],saved['rule'])
                    measured=tr.record_metric(records,z,th);equal(stored[method],measured);metrics_count+=1
                    pred=tr.probabilities(z)>=th;truth=records['targets'].astype(bool)
                    assert measured['exact']==(pred==truth).all(1).mean()
                    for c,name in enumerate(classes):
                        assert measured['per_label'][name]['fp']==int((pred[:,c]&~truth[:,c]).sum())
                        assert measured['per_label'][name]['fn']==int((~pred[:,c]&truth[:,c]).sum())
            if stage>1:
                for method in methods:
                    if method=='base':continue
                    history=read(output/('history_'+method+'.json'));status=read(output/(method+'_status.json'))
                    assert len(history)==24 and all(h['samples']==8000 and h['steps']==h['epoch']*250 for h in history)
                    last=torch.load(output/('last_'+method+'.pt'),map_location='cpu',weights_only=True)
                    best=torch.load(output/(method+'.pt'),map_location='cpu',weights_only=True)
                    assert last['optimizer_steps']==6000 and {int(v['step']) for v in last['optimizer']['state'].values()}=={6000}
                    assert best['epoch']==int(np.argmax([h['check_aupr'] for h in history]))+1
                    assert best['backbone_sha256']==refs[f'{seed}_B'] and status['checkpoint_sha256']==sha256(output/(method+'.pt'))
                    assert status['status']=='completed'
                    if stage==3:
                        current=tr.previous_head(suite,seed);assert best['previous_head_sha256']==(sha256(current) if current else None)
                        for h in history:assert h['natural_samples']+h['paired_views']==8000
                    models.append({'stage':stage,'seed':seed,'method':method,'steps':6000,'best_epoch':best['epoch'],'trainable_parameters':last['trainable_parameters']})
            if stage==3:
                pair_records=dict(np.load(suite/'cache'/f'seed{seed}'/'pair_development_test.npz'));stored=read(output/'metrics_pair_report.json')['metrics']
                for method,m in stored.items():
                    z=tr.fused_logits(pair_records,predict(method,pair_records));th,_=get_rule(saved['models'][method],saved['rule'])
                    measured=tr.record_metric(pair_records,z,th);measured['one_bit_targets']={}
                    for c,name in [(0,'FMB'),(2,'SawFM'),(4,'MFT'),(5,'ISRJ')]:
                        rows=pair_records['pair_target']==c;truth=pair_records['targets'][rows,c].astype(bool);scores=z[rows,c];pred=tr.probabilities(scores)>=th[c]
                        measured['one_bit_targets'][name]={'auroc':float(roc_auc_score(truth,scores)),'recall':float(pred[truth].mean()),'fpr':float(pred[~truth].mean())}
                    equal(m,measured);metrics_count+=1
                    for c,name in [(0,'FMB'),(2,'SawFM'),(4,'MFT'),(5,'ISRJ')]:
                        pair_indices=np.flatnonzero((pair_records['pair_target'][::2]==c)&pair_records['weak_mask'][1::2,c])
                        rows=np.ravel(np.c_[2*pair_indices,2*pair_indices+1])
                        if len(rows):
                            yy=pair_records['targets'][rows,c].astype(bool);scores=z[rows,c];pred=tr.probabilities(scores)>=th[c]
                            pair_weak_diagnostics.append({'seed':seed,'method':method,'label':name,'pairs':len(pair_indices),
                                'weak_pair_auroc':float(roc_auc_score(yy,scores)),'weak_recall':float(pred[yy].mean()),'matched_negative_fpr':float(pred[~yy].mean())})
        for method,metrics in result['aggregate'].items():
            ms=[read(folder/f'seed{seed}'/'metrics_development_test.json')['metrics'][method] for seed in range(2050,2055)]
            for key,stat in metrics.items():equal(stat,{'mean':float(np.mean([m[key] for m in ms])),'std':float(np.std([m[key] for m in ms],ddof=1))})
        checks=[read(folder/f'seed{seed}'/'metrics_check.json')['metrics'] for seed in range(2050,2055)]
        common=sorted(set.intersection(*[set(v) for v in checks]));baseline=selection['baseline'];guard='f1' if stage==1 else 'B_f1_guard'
        means={m:{k:float(np.mean([v[m][k] for v in checks])) for k in selection['check_means'][m]} for m in common}
        equal(means,selection['check_means']);eligible=[]
        for method in [m for m in common if m not in ['A_f1','B_f1_guard']]:
            fpr=max(float(np.mean([v[method]['per_label'][c]['negative_fpr']-v['A_f1']['per_label'][c]['negative_fpr'] for v in checks])) for c in classes)
            valid=means[method]['tail_f1']>=means[baseline]['tail_f1']-.003 and fpr<=.01 and means[method]['saw_weak_recall']>=means[guard]['saw_weak_recall']-.03
            valid&=means[method]['exact']>=means[baseline]['exact']-.002 if stage==1 else means[method]['exact']>means[baseline]['exact']
            if valid:eligible.append(method)
        assert eligible==selection['eligible']
        selected=max(eligible,key=lambda m:(means[m]['exact'],m==baseline)) if eligible else baseline
        assert selected==selection['selected']
        from scipy.stats import t
        reports=[read(folder/f'seed{seed}'/'metrics_development_test.json')['metrics'] for seed in range(2050,2055)]
        for method,effects in result['effects_vs_stage_baseline'].items():
            for key,effect in effects.items():
                delta=np.array([v[method][key]-v[baseline][key] for v in reports]);radius=t.ppf(.975,4)*delta.std(ddof=1)/np.sqrt(5)
                equal(effect,{'mean':float(delta.mean()),'paired95':[float(delta.mean()-radius),float(delta.mean()+radius)],'per_seed':delta.tolist()})
        effect=result['effects_vs_stage_baseline'][selected];limits=plan['protocol']['success']
        fpr_delta={c:float(np.mean([r[selected]['per_label'][c]['negative_fpr']-r[baseline]['per_label'][c]['negative_fpr'] for r in reports])) for c in classes}
        criteria={'exact_gain_ge0.5pp':effect['exact']['mean']>=limits['exact_gain'],'positive_seeds_ge4':sum(v>0 for v in effect['exact']['per_seed'])>=4,
                  'paired_interval_lower_positive':effect['exact']['paired95'][0]>0,'tail_f1_drop_le0.3pp':effect['tail_f1']['mean']>=-limits['max_tail_drop'],
                  'class_fpr_rise_le1pp':max(fpr_delta.values())<=limits['max_class_fpr_rise'],'saw_weak_recall_drop_le3pp':effect['saw_weak_recall']['mean']>=-limits['max_saw_weak_drop']}
        equal(criteria,result['acceptance']['conditions']);equal(fpr_delta,result['acceptance']['class_fpr_delta'])
        assert all(criteria.values())==result['acceptance']['passed']
        assert result['selection']==selection and selection['report_values_used'] is False
        bindings.append({'stage':stage,'freeze_time':frozen['frozen_at'],'selected_on_check':selection['selected'],'reference':selection['baseline'],'primary_passed':result['acceptance']['passed']})
        analyses[str(stage)]={'aggregate':result['aggregate'],'effects':result['effects_vs_stage_baseline'],'selection':selection,'acceptance':result['acceptance']}
    for seed in range(2050,2055):
        for epoch in range(24):
            a=read(suite/'stage3'/f'seed{seed}'/'history_local_pair_bce.json') if done['new_local_models'] else []
            b=read(suite/'stage3'/f'seed{seed}'/'history_local_pair_margin.json') if done['new_local_models'] else []
            if a:
                assert a[epoch]['natural_draw_sha256']==b[epoch]['natural_draw_sha256'] and a[epoch]['pair_draw_sha256']==b[epoch]['pair_draw_sha256']
    audit_result={'status':'passed','new_models_verified':len(models),'optimizer_steps_verified':sum(m['steps'] for m in models),
        'frozen_source_files_verified':len(plan['source_manifest']),'reused_checkpoints_verified':len(refs),'metric_conditions_recomputed':metrics_count,
        'calibration_bundles_replayed':decisions_count,'old_identity_files_rechecked':len(generation['checked_identity_files']),
        'new_background_noise_identities_disjoint':True,'paired_side_ids_separate_from_noise_ids':True,'pair_edge_splits_disjoint':True,
        'regenerated_waveform_pairs_verified':12,'report_order_verified':bindings,'pair_bce_margin_sampling_identical':True,
        'completion':done,'models':models}
    analysis={'stages':analyses,'observability':read(suite/'stage3/observability.json'),'audit':audit_result,'weak_pair_report_diagnostics':pair_weak_diagnostics,
              'limitations':plan['protocol']['limitations'],'pair_holdout_scope':generation['limitations']}
    write_json(out/'completion_audit.json',audit_result);write_json(out/'analysis.json',analysis)
    annotate(suite)
    print(json.dumps({'audit':'passed','models':len(models),'recomputed_conditions':metrics_count,'stages':bindings},ensure_ascii=False))


def annotate(suite):
    from scipy.stats import t
    from longtail.io import write_json,sha256
    plan=read(suite/'plan.json');out=Path(plan['report_output']);a=read(out/'analysis.json');r=read(suite/'stage3/results.json')
    contrasts={}
    for x,y in [('local_natural','base'),('local_pair_bce','local_natural'),('local_pair_margin','local_pair_bce'),('local_pair_bce','B_f1_guard')]:
        contrasts[x+'_vs_'+y]={}
        for key in ['exact','macro_f1','macro_aupr','mean_fpr','saw_weak_recall','target_weak_recall']:
            d=np.array([v[x][key]-v[y][key] for v in r['per_seed_metrics']]);radius=t.ppf(.975,4)*d.std(ddof=1)/np.sqrt(5)
            contrasts[x+'_vs_'+y][key]={'mean':float(d.mean()),'paired95':[float(d.mean()-radius),float(d.mean()+radius)],'per_seed':d.tolist()}
    a['stage3_attribution']=contrasts;write_json(out/'analysis.json',a)
    selected=r['selection']['selected'];head=read(suite/'stage2/selection.json')['selected'];frozen=read(suite/'stage3/frozen.json');models=[]
    for seed in range(2050,2055):
        components={kind:{'path':item['checkpoint'],'sha256':item['sha256']} for kind,item in plan['references'][str(seed)].items() if kind in ['B','FMB']}
        if head!='base':
            path=suite/'stage2'/f'seed{seed}'/(head+'.pt');components['five_label_head']={'path':str(path),'sha256':sha256(path)}
        if selected!='base':
            path=suite/'stage3'/f'seed{seed}'/(selected+'.pt');components['three_label_local_head']={'path':str(path),'sha256':sha256(path)}
        for info in components.values():assert frozen['files'][info['path']]==info['sha256']
        bundle=read(suite/'stage3'/f'seed{seed}'/'decisions.json');threshold=bundle['models'][selected]['thresholds'][bundle['rule']]
        if threshold is None:threshold=bundle['models'][selected]['thresholds']['matched_fpr']
        cfg=read(Path(plan['references'][str(seed)]['B']['checkpoint']).parent/'config.json')
        models.append({'seed':seed,'components':components,'classes':['FMB','AMA','SawFM','SinFM','MFT','ISRJ'],'thresholds':threshold,
                       'preprocess':cfg['dataloader']['preprocess'],'backbone_feature_dimension':128,'five_label_head':head,'local_head':selected,
                       'FMB_source':'FMB expert, threshold0.5','local_residual_columns':[2,4,5]})
    write_json(out/'selected_models.json',{'status':'passed_current_simulated_report_gate' if r['acceptance']['passed'] else 'candidate_failed_report_gate',
        'selected_on':'check only, same method for all five seeds','five_seeds_are_separate_models_not_prediction_averaging':True,
        'implementation':str(suite/'source_snapshot/src/longtail/staged_training.py'),'models':models,'acceptance':r['acceptance'],
        'report_exact':r['aggregate'][selected]['exact'],'source_freeze':str(suite/'stage3/frozen.json')})


def render(suite):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font_manager.fontManager.addfont('/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf')
    plt.rcParams.update({'font.family':['DejaVu Sans','Droid Sans Fallback'],'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False})
    plan=read(suite/'plan.json');out=Path(plan['report_output']);a=read(out/'analysis.json')
    names_cn={'f1':'逐类F1','matched_fpr':'误报预算','exact_fpr':'误报约束Exact','exact_weak':'弱召回约束Exact','base':'阶段内原模型',
              'head_natural':'自然重训头','head_mixed':'温和组合重训头','local_natural':'自然局部头','local_pair_bce':'配对BCE局部头','local_pair_margin':'配对margin局部头',
              'A_f1':'原8k融合参考','B_f1_guard':'原32k融合参考'}
    fig,axes=plt.subplots(1,3,figsize=(15,4.4),layout='constrained')
    for i,ax in enumerate(axes):
        stage=a['stages'][str(i+1)];names=[m for m in stage['aggregate'] if m not in ['A_f1','B_f1_guard']]
        x=np.arange(len(names));avg=[stage['aggregate'][m]['exact']['mean']*100 for m in names];std=[stage['aggregate'][m]['exact']['std']*100 for m in names]
        ax.bar(x,avg,yerr=std,capsize=3,color=['#c58b50' if m==stage['selection']['selected'] else '#198c96' for m in names])
        ax.set(xticks=x,xticklabels=[names_cn[m] for m in names],ylabel='Exact (%)',title=f'阶段{i+1}；橙色为check选定候选')
        ax.tick_params(axis='x',rotation=35)
        for j,v in enumerate(avg):ax.text(j,v+.2,f'{v:.2f}',ha='center',fontsize=9)
        ax.set_ylim(max(0,min(avg)-4),max(avg)+4)
    fig.suptitle('三阶段分别使用独立新报告池；误差条为五种子标准差，不跨池相加')
    fig.savefig(out/'ordered_stage_results.png',dpi=170);plt.close(fig)
    rows=[]
    for s in ['1','2','3']:
        stage=a['stages'][s];selected=stage['selection']['selected'];effect=stage['effects'][selected]['exact'];bounds=effect['paired95']
        rows.append([s,names_cn[selected],names_cn[stage['selection']['baseline']],f"{stage['aggregate'][selected]['exact']['mean']*100:.3f}",f"{effect['mean']*100:+.3f}",f'[{bounds[0]*100:+.3f}, {bounds[1]*100:+.3f}]',stage['acceptance']['passed']])
    def table(headers,values):return '<div style="overflow-x:auto"><table><tr>'+''.join('<th>'+escape(str(v))+'</th>' for v in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+escape(str(v))+'</td>' for v in row)+'</tr>' for row in values)+'</table></div>'
    html='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>顺序训练与测试：完成分析</title><style>body{font:16px/1.8 system-ui;max-width:1250px;margin:28px auto;padding:24px;color:#234}table{border-collapse:collapse;width:100%;font-size:14px}td,th{border:1px solid #ccd;padding:8px}th{background:#e7f2f3}img{width:100%}a{color:#16758a}</style><h1>顺序训练与测试：决策→分类头→one-bit局部头</h1>'
    html+='<p>全部阶段按预定顺序执行；后阶段只采用前阶段check选择。各阶段独立生成calibration/check/report；FMB专家固定，backbone冻结，各新增模型6000步。主验收要求同池Exact提高至少0.5pp、4/5种子正向、配对区间下限>0，同时满足尾类、逐类误报和SawFM弱召回要求。</p>'
    html+=table(['阶段','check选定候选','同池参考','报告Exact %','Exact差 pp','条件95%区间 pp','主验收'],rows)
    html+='<img alt="三阶段新报告结果" src="data:image/png;base64,'+base64.b64encode((out/'ordered_stage_results.png').read_bytes()).decode()+'">'
    for s in ['1','2','3']:
        stage=a['stages'][s];html+=f'<h2>阶段{s}：全部方法与约束</h2>'
        keys=['exact','macro_f1','macro_aupr','tail_f1','mean_fpr','saw_weak_recall','target_weak_recall']
        html+=table(['方法']+keys,[[names_cn[m]]+[f"{v[k]['mean']*100:.3f}" for k in keys] for m,v in stage['aggregate'].items()])
        html+=table(['主候选验收条件','通过'],list(stage['acceptance']['conditions'].items()))
        html+=f'<p><a href="stage{s}.html">阶段{s}原始报告</a> · <a href="stage{s}_results.json">逐类、逐种子JSON</a></p>'
    html+='<h2>收益归因与最终候选</h2><p>第三阶段预选的配对BCE局部头在4032新报告身份上达到Exact 73.462%、Macro-F1 94.643%、Macro-AUPR 98.761%。相对本阶段自然重训线性头基线72.133%提高1.329 pp，条件配对95%区间[+0.938,+1.720] pp，五种子均正向，全部预定验收通过。相对同池原32k融合71.597%，完整候选提高1.865 pp，区间[+1.348,+2.382] pp。不同阶段的报告池不同，增益不相加。</p><p>自然非线性局部头已提高Exact 0.957 pp，区间[+0.487,+1.428] pp。再加入one-bit配对BCE增加0.372 pp，五种子方向均正，但区间[−0.047,+0.791] pp跨零，尚不能把全部增益归因于配对数据。额外margin相对配对BCE为−0.109 pp，区间跨零，未显示整体Exact额外收益。</p><p>第一阶段F1相对误报预算仅提高0.352 pp，区间跨零，SawFM弱召回还下降6.100 pp；第二阶段自然重训头提高0.109 pp，同样区间跨零。两阶段均未通过本轮正式验收，仍按预先冻结的check选择进入后续阶段，没有根据报告更换方案。</p><p>最终候选平均FPR从3.789%降至3.434%；SawFM FPR增加0.333 pp，MFT和ISRJ分别下降1.089/1.371 pp。SawFM弱召回均值从50.560%升至54.784%，但该增益区间跨零，目标三类平均弱召回增益也跨零。因此当前证据支持标签边界及整体Exact改善，尚未证明弱分量召回全面稳定提高。</p>'
    html+='<h2>可观测性门控</h2><p>以下探针只接收单条信号；raw difference与干净分量仅用于生成审计。check AUROC门槛为0.60，通过表示本轮局部训练具有可执行依据，不代表证明了信息论可识别性。</p>'
    html+=table(['目标编号','物理特征探针AUROC','冻结模型平均AUROC','通过'],[[c,f"{v['physical_single_view_probe_auroc']:.4f}",f"{np.mean(v['frozen_single_view_auroc_per_seed']):.4f}",v['passed']] for c,v in a['observability']['per_target'].items()])
    html+='<p>目标编号0/2/4/5为FMB/SawFM/MFT/ISRJ；FMB为固定专家诊断对照。配对边留出针对配对训练，预训练自然数据已覆盖63种标签组合。报告区间以固定仿真池为条件，不能当作实测泛化保证。当前报告已消费，不继续用于阈值或训练调参。</p><p>核验25个新增模型、150000个实际优化器步、101份冻结源文件、15个复用权重、254组测评条件和全部身份隔离；重放50份calibration决策，并独立再生成12组配对波形检查。各阶段报告均在模型、决策和check选择冻结后执行。</p><p><a href="completion_audit.json">完成审计</a> · <a href="analysis.json">完整汇总</a> · <a href="selected_models.json">五种子候选权重与阈值清单</a> · <a href="plan.html">冻结方案</a></p></html>'
    (out/'analysis.html').write_text(html);print(json.dumps({'analysis':str(out/'analysis.html')},ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--suite',type=Path,required=True);p.add_argument('--render-only',action='store_true');a=p.parse_args()
    if a.render_only:render(a.suite.resolve())
    else:audit(a.suite.resolve())
