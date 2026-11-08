"""Read-only completion review of calibration and low-dose gap-stratified pairs."""
import argparse
from datetime import datetime
from html import escape
import json
from pathlib import Path

import numpy as np

CLASSES = ['FMB','AMA','SawFM','SinFM','MFT','ISRJ']
KEYS = ['exact','macro_f1','macro_aupr','mean_fpr','focus_fpr','saw_weak_recall','mft_weak_recall','focus_weak_recall','focus_weak_aupr','nonweak_recall']


def read(path):
    return json.loads(Path(path).read_text())


def average(rows):
    result = {k:float(np.mean([v[k] for v in rows])) for k in KEYS}
    result['per_label'] = {}
    for c in CLASSES:
        result['per_label'][c] = {k:float(np.mean([v['per_label'][c][k] for v in rows])) for k in ['fp','fn','negative_fpr','recall','aupr']}
        result['per_label'][c].update({k:float(np.mean([v['weak_component']['per_label'][c][k] for v in rows])) for k in ['weak_recall','weak_aupr','weak_auroc']})
    return result


def analyze(suite):
    from longtail.io import sha256,write_json
    from longtail.targeted_validation import metric,KEYS as FULL_KEYS
    from longtail.metrics import probabilities
    from sklearn.metrics import roc_auc_score
    plan = read(suite/'plan.json')
    status = read(suite/'pipeline_status.json')
    audit = read(suite/'completion_audit.json')
    assert status['status'] == 'completed' and audit['status'] == 'passed'
    checked = {}
    for stage in ['calibration','targeted']:
        frozen = read(suite/stage/'frozen.json')
        assert frozen['plan_sha256'] == sha256(suite/'plan.json')
        checked.update(frozen['files'])
    for path,digest in checked.items():
        assert sha256(path) == digest
    for source in plan['source_manifest']:
        assert sha256(suite/'source_snapshot'/source['path']) == source['sha256']
    for path,digest in plan['data_hashes'].items():
        assert sha256(path) == digest
    contexts = [('calibration','development_test'),('targeted','development_test'),('targeted','ood_wide'),('targeted','pair_report')]
    outcomes,focus_gaps,pair_bins = {},{},{}
    replayed = 0
    for stage,role in contexts:
        rows = [read(suite/stage/f'seed{seed}'/f'metrics_{role}.json')['metrics'] for seed in plan['seeds']]
        means = {}
        for model in rows[0]:
            common = set.intersection(*(set(r[model]) for r in rows))
            means[model] = {rule:average([r[model][rule] for r in rows]) for rule in sorted(common)}
        key = 'calibration' if stage == 'calibration' else role
        outcomes[key] = means
        gap_rows,pair_rows = {},{}
        for seed in plan['seeds']:
            folder = suite/stage/f'seed{seed}'
            pred = dict(np.load(folder/f'predictions_{role}.npz'))
            decisions = read(folder/'decisions.json')
            for model,rules in means.items():
                th = decisions['models'][model]['thresholds']['locked']
                z = pred['logits_'+model]
                recalculated = metric(pred,z,th)
                expected = rows[plan['seeds'].index(seed)][model]['locked']
                assert all(np.isclose(recalculated[k],expected[k],atol=1e-12,rtol=0) for k in FULL_KEYS)
                replayed += 1
                if stage == 'calibration':
                    continue
                pp = probabilities(z)
                if role != 'pair_report':
                    labels = pred['targets'][:,[2,4]].astype(bool)
                    gap = -pred['relative_power_db'][:,[2,4]]
                    detected = (pp >= th)[:,[2,4]]
                    values = []
                    for lo,hi in [(6,10),(10,15),(15,20),(20,30)]:
                        selected = labels & (gap >= lo) & (gap < hi)
                        values.append({'gap':[lo,hi],'positives':int(selected.sum()),'recall':float(detected[selected].mean()) if selected.any() else None})
                    gap_rows.setdefault(model,[]).append(values)
                else:
                    values = {}
                    for c in [2,4]:
                        values[CLASSES[c]] = []
                        for b in range(3):
                            mask = (pred['pair_target'][::2] == c) & (pred['target_gap_bin'][::2] == b)
                            index = np.flatnonzero(mask)
                            negative,positive = 2*index,2*index+1
                            both = np.ravel(np.c_[negative,positive])
                            values[CLASSES[c]].append({'gap':[[6,10],[10,15],[15,20]][b], 'pairs':len(index),
                                'single_view_auroc':float(roc_auc_score(pred['targets'][both,c],z[both,c])),
                                'positive_recall':float((pp[positive,c] >= th[c]).mean()),
                                'matched_negative_fpr':float((pp[negative,c] >= th[c]).mean())})
                    pair_rows.setdefault(model,[]).append(values)
        if gap_rows:
            focus_gaps[role] = {m:[{**v,'recall':float(np.mean([r[i]['recall'] for r in rr])) if v['positives'] else None} for i,v in enumerate(rr[0])] for m,rr in gap_rows.items()}
        if pair_rows:
            pair_bins = {m:{c:[{**v,**{k:float(np.mean([r[c][i][k] for r in rr])) for k in ['single_view_auroc','positive_recall','matched_negative_fpr']}} for i,v in enumerate(rr[0][c])] for c in ['SawFM','MFT']} for m,rr in pair_rows.items()}
    stability = {}
    for version in ['new','old']:
        deviations = []
        for seed in plan['seeds']:
            decision = read(suite/'calibration'/f'seed{seed}/decisions.json')
            deviations.append(np.std(np.array(decision['models'][version]['small_block_thresholds'])[:,[2,4]],axis=0).tolist())
        stability[version] = {'per_seed_small_block_threshold_std':deviations,'mean_std_Saw_MFT':np.mean(deviations,axis=0).tolist(),
                              'scope':'Across8 disjoint1008 calibration blocks; no best-block selection or estimate of repeated-large-pool variance.'}
    histories = {}
    for method in ['natural','pair1','pair2']:
        histories[method] = {'best_epochs':[],'epoch18_to24_weak_AP_change':[]}
        for seed in plan['seeds']:
            h = read(suite/'targeted'/f'seed{seed}'/f'history_{method}.json')
            assert len(h) == 24 and h[-1]['optimizer_steps'] == 6000
            histories[method]['best_epochs'].append(max(h,key=lambda v:v['check_focus_weak_aupr'])['epoch'])
            histories[method]['epoch18_to24_weak_AP_change'].append(h[-1]['check_focus_weak_aupr']-h[17]['check_focus_weak_aupr'])
    result = {'review_only':True,'training_weights_thresholds_or_selection_modified':False,'status':status,
              'wall_seconds':(datetime.fromisoformat(status['finished'])-datetime.fromisoformat(status['started'])).total_seconds(),
              'audit':audit,'source_files_verified':len(plan['source_manifest']),'frozen_files_reverified':len(checked),
              'additional_main_and_supplementary_locked_metrics_replayed':replayed,'outcomes':outcomes,
              'stage_summaries':{s:read(suite/s/'summary.json') for s in ['calibration','targeted']},
              'small_calibration_stability':stability,'training_curves':histories,
              'Saw_MFT_gap_recalls':focus_gaps,'paired_edge_gap_diagnostics':pair_bins,
              'assessment':{'complete':True,'calibration_new_shared_policy_accepted':False,
                  'selected_method':'pair1','selected_pair_view_fraction':.0625,'selected_rule':'locked',
                  'overall_Exact_gate_passed':False,'weak_tradeoff_vs_frozen_new_gate_passed':True,
                  'weak_tradeoff_vs_natural_continuation_gate_passed':False,
                  'limitations':'fixed simulator/pools, five conditional model seeds; secondary contrasts not multiplicity corrected; calibration FP budgets do not enforce equal test FPR'}}
    dest = Path(plan['report_output'])
    write_json(dest/'review.json',result)
    write_json(dest/'candidate_assessment.json',result['assessment'])
    render(result,dest)
    page = dest/'analysis.html'
    html = page.read_text()
    marker = '<!-- completion-review-link -->'
    if marker not in html:
        html = html.replace('<h1>',marker+'<p><a href="review.html">完成统计、收益归因与功率差诊断</a></p><h1>',1)
        page.write_text(html)
    return result


def table(headers,rows):
    return '<table><tr>'+''.join('<th>'+escape(str(v))+'</th>' for v in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+escape(str(v))+'</td>' for v in row)+'</tr>' for row in rows)+'</table>'


def render(d,dest):
    o = d['outcomes']
    html = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>大校准池与分层配对完成分析</title><style>body{font:16px/1.8 system-ui;max-width:1250px;margin:24px auto;padding:24px;color:#234}table{border-collapse:collapse;width:100%;font-size:14px}th,td{border:1px solid #ccd;padding:7px}th{background:#eaf3f4}img{max-width:100%}</style><h1>完成统计与收益归因</h1>'
    html += '<p>完成时间2026-11-04 19:45:19（北京时间）；耗时15分51秒。15个头副本全部完成360轮、90000实际优化器步，无运行失败。审计通过：1068组指标、10份校准决策、18个独立重生成的原始配对波形；原始参考权重和其他四类输出/阈值保持不变。</p>'
    html += '<p><b>6.25%定向配对改善了弱分量召回，但整体Exact提升尚不显著。大校准池主要改变误报—漏检权衡，没有得到可替代原阈值的共同决策。</b></p><p>Exact为整组标签完全正确率；五种子均值，非模型集成。不同阶段使用不同报告池，不能跨池相减。</p><p><a href="plan.html">冻结方案</a> · <a href="completion_audit.json">完成审计</a> · <a href="review.json">详细统计</a></p>'
    html += '<h2>1 固定权重的大校准池</h2>'
    rows = []
    for rule in ['locked','f1_1008','f1_8064','fpr03','fpr05','fpr07','weak_guard']:
        if rule not in o['calibration']['new']:
            continue
        m = o['calibration']['new'][rule]
        rows.append([rule]+[f'{100*m[k]:.3f}%' for k in ['exact','saw_weak_recall','mft_weak_recall','focus_fpr']])
    html += table(['新固定候选规则','Exact','SawFM弱召回','MFT弱召回','两目标平均FPR'],rows)
    html += '<p>8064条大池F1相对预先指定的1008条小池，Exact只提高0.027pp（条件95%区间[−0.511,+0.565]）；两目标平均弱召回提高4.833pp，同时平均FPR增加1.127pp。MFT弱召回63.50%→72.39%，体现决策点移动，不是排序能力提高：各阈值下弱AP完全相同。</p><p>check阶段仅locked满足两个固定候选的全部共同保护条件；大池F1、5%及7%预算违反新候选逐类FPR增量上限，3%预算则损失弱召回。旧候选weak_guard在4/5种子不可行，按预定规则不参加共同选型。没有使用报告结果重新挑阈值。</p><p>八份1008条校准子池的阈值有波动（新候选SawFM/MFT平均标准差约0.070/0.106），说明原小池有抽样不确定性；本轮没有重复多个独立8064池，不能量化大池方差降低多少。</p>'
    html += '<h2>2 同一8064身份报告上的分层配对对照</h2>'
    names = {'new':'固定新候选','natural':'0%自然继续训练','pair1':'6.25%分层配对（check预选）','pair2':'12.5%分层配对','old':'固定旧候选'}
    rows = []
    for model,label in names.items():
        m = o['development_test'][model]['locked']
        rows.append([label]+[f'{100*m[k]:.3f}%' for k in ['exact','saw_weak_recall','mft_weak_recall','focus_fpr','focus_weak_aupr']])
    html += table(['方法','Exact','Saw弱召回','MFT弱召回','两目标FPR','两目标弱AP'],rows)
    html += '<p>6.25%相对固定新候选：Exact提高0.181pp，条件95%区间[−0.035,+0.397]，4/5种子提高，未通过整体Exact门槛。SawFM弱召回提高2.180pp、MFT提高2.225pp，五种子均提高；两者平均提高2.202pp，区间[+1.474,+2.931]。SawFM/MFT FPR分别增加0.459/0.292pp，按五种子均值通过预设弱召回权衡门槛；此门槛允许小幅误报增加，不代表等误报率改善。</p>'
    html += '<p>相对0%继续训练，6.25%的配对额外贡献为：Exact +0.079pp，区间[−0.100,+0.258]；平均弱召回+1.679pp，区间[+0.713,+2.645]；弱AP +0.177pp，区间[+0.087,+0.266]。存在小幅排序收益，不能把相对固定权重的全部改善归因于配对。相对继续训练的SawFM FPR增加0.549pp，超出辅助权衡门槛0.5pp，未通过这项更严格对照。</p>'
    html += '<p>12.5%使平均弱召回提高3.184pp，但SawFM FPR增加0.781pp，超过弱召回权衡的0.5pp上限；Exact也未优于6.25%。当前不支持扩大配对比例。原固定参考权重和其他四类输出保持不变。</p><p>本轮按功率差分层，使配对目标正侧中2/3属于至少10dB弱分量，明显高于此前未分层配对约22%的水平。但与上一轮25%方案相比，还同时改变了初始化、学习率、训练标签范围和配对筛选，不能把跨轮结果差异只归因于降低配对比例。本轮0%、6.25%、12.5%的共同对照才用于收益归因。</p>'
    html += '<p>大池F1辅助规则下，6.25%与12.5%均约74.983% Exact；该规则没有通过第一阶段共同选型，因此属于预设辅助结果，不能替代主决策locked的验收，也不能据此宣称达到稳定75%。相对固定旧候选，预选6.25%主决策Exact提高0.647pp，区间[−0.007,+1.302]，仍跨零。</p>'
    html += '<h2>3 功率差及配对边留出</h2>'
    rows = []
    for role,label in [('ood_wide','宽功率10～40dB自然报告'),('pair_report','功率差分层的配对边留出报告')]:
        for model in ['new','natural','pair1','pair2']:
            m = o[role][model]['locked']
            rows.append([label,names[model]]+[f'{100*m[k]:.3f}%' for k in ['exact','saw_weak_recall','mft_weak_recall']])
    html += table(['补充报告','方法','Exact','Saw弱召回','MFT弱召回'],rows)
    html += '<p>宽功率报告Exact仅53.24%→53.53%，改善较小。两目标在原分布15～20dB差距的召回32.38%→35.25%；宽功率20～30dB差距仅4.63%→5.37%。本轮训练只覆盖20dB以下差距，没有解决更极端功率差。</p>'
    html += '<p>配对边留出报告Exact56.50%→56.57%，无明显提升。15～20dB差距中，SawFM单条信号AUROC约74.46%→74.60%、正侧召回21.46%→22.50%；MFT AUROC78.31%→78.52%、召回38.96%→39.90%。说明对最难边界的排序改进仍很小；这些结果不能证明信息论不可观测，也不能称完全未见组合泛化，原backbone已见过全部63种组合。</p>'
    html += '<h2>后续优先级</h2><p>将6.25%保留为弱分量候选，冻结其权重、决策和比例，在新的独立身份池与固定新候选、0%继续训练共同复现；不要继续用本轮报告调参。若核心目标是提升整体Exact，应先验证15～20dB困难样本在全局128维特征中是否丢失区分信息，再做信号长度或局部时频证据的受控消融。现阶段没有证据支持增加到12.5%或单纯延长头训练。</p>'
    html += '<p>限制：固定仿真池上的五种子条件区间，次要比较未校正多重检验；统一校准预算不保证测试FPR相同；使用同一Python物理仿真器和固定FMB，缺少实测信号验证。本复核只读取结果，未修改训练、阈值或候选选择。</p><p><img src="outcome_overview.png" alt="校准权衡、Exact区间、分层配对弱召回与功率差结果"></p></html>'
    (dest/'review.html').write_text(html)


def plot(d,dest):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    o = d['outcomes']
    fig,axes = plt.subplots(2,2,figsize=(12,8.6),constrained_layout=True)
    ax = axes[0,0]
    for model,label,color in [('new','Fixed new weights','#177d83'),('old','Fixed old weights','#b77d31')]:
        rules = [o['calibration'][model][r] for r in ['fpr03','fpr05','fpr07']]
        ax.plot([100*r['focus_fpr'] for r in rules],[100*r['focus_weak_recall'] for r in rules],'-o',label=label,color=color)
    for rule,label,marker in [('locked','New locked','s'),('f1_8064','New large F1','*')]:
        r = o['calibration']['new'][rule]
        ax.scatter(100*r['focus_fpr'],100*r['focus_weak_recall'],s=80,marker=marker,label=label,color='#333')
    ax.set_xlabel('Observed report FPR, SawFM/MFT mean (%)')
    ax.set_ylabel('Weak recall, SawFM/MFT mean (%)')
    ax.set_title('A. Calibration changes the operating point')
    ax.legend(fontsize=8)
    ax.grid(alpha=.2)
    ax = axes[0,1]
    summary = d['stage_summaries']['targeted']['comparisons']
    for i,(key,label) in enumerate([('natural_vs_frozen_new','0% continuation'),('pair1_vs_frozen_new','6.25% paired'),('pair2_vs_frozen_new','12.5% paired')]):
        v = summary[key]['deltas']['exact'];mean = v['mean']*100;lo,hi = np.array(v['paired95'])*100
        ax.errorbar(mean,i,xerr=np.array([[mean-lo],[hi-mean]]),fmt='o',capsize=4,color=['#80969f','#177d83','#b77d31'][i])
    ax.axvline(0,color='#888',ls='--',lw=1)
    ax.set_yticks(range(3),['0% continuation','6.25% paired','12.5% paired'])
    ax.invert_yaxis()
    ax.set_xlabel('Exact gain vs frozen new (pp), conditional 95% CI')
    ax.set_title('B. Overall accuracy gain remains uncertain')
    ax.grid(axis='x',alpha=.2)
    ax = axes[1,0]
    x = np.arange(2)
    for i,(m,label,color) in enumerate([('new','Frozen new','#abbcc3'),('natural','0% continuation','#63839a'),('pair1','6.25% paired','#177d83'),('pair2','12.5% paired','#b77d31')]):
        r = o['development_test'][m]['locked']
        ax.bar(x+(i-1.5)*.19,[100*r['saw_weak_recall'],100*r['mft_weak_recall']],width=.19,label=label,color=color)
    ax.set_xticks(x,['SawFM','MFT'])
    ax.set_ylim(0,100)
    ax.set_ylabel('Weak-component recall (%)')
    ax.set_title('C. Small targeted doses help weak recall')
    ax.legend(fontsize=8)
    ax = axes[1,1]
    for role,m,label,color,style in [('development_test','new','Original, frozen','#80969f','-'),('development_test','pair1','Original, 6.25%','#177d83','-'),('ood_wide','new','Wide power, frozen','#c5a677','--'),('ood_wide','pair1','Wide power, 6.25%','#b77d31','--')]:
        bins = d['Saw_MFT_gap_recalls'][role][m]
        xx = [i for i,b in enumerate(bins) if b['recall'] is not None]
        yy = [100*b['recall'] for b in bins if b['recall'] is not None]
        ax.plot(xx,yy,style+'o',label=label,color=color)
    ax.set_xticks(range(4),['6-10','10-15','15-20','20-30'])
    ax.set_xlabel('Gap to strongest component (dB)')
    ax.set_ylabel('SawFM/MFT positive recall (%)')
    ax.set_ylim(0,105)
    ax.set_title('D. Large power gaps remain the bottleneck')
    ax.legend(fontsize=8)
    fig.suptitle('Calibration and targeted-pair CPU validation: five seeds, fixed simulated pools',fontsize=14)
    fig.savefig(dest/'outcome_overview.png',dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite',type=Path,required=True)
    p.add_argument('--plot-only',action='store_true')
    a = p.parse_args()
    suite = a.suite.resolve()
    dest = Path(read(suite/'plan.json')['report_output'])
    if a.plot_only:
        plot(read(dest/'review.json'),dest)
    else:
        result = analyze(suite)
        print(json.dumps({'status':'review_completed','wall_seconds':result['wall_seconds'],'report':str(dest/'review.html')},ensure_ascii=False))
