"""Summarize the frozen CPU ablation; makes no training/selection decisions."""
import argparse
import base64
import csv
from datetime import datetime
from html import escape
import json
from pathlib import Path
from statistics import mean, stdev
from zoneinfo import ZoneInfo
import numpy as np


NAMES = {'baseline': 'DFMS基线', 'single': '最深层查询',
         'capacity': '最深层查询·容量对照', 'multiscale': '第2/3/4层查询'}
ARMS = list(NAMES)


def read(path):
    return json.loads(Path(path).read_text())


def paired_exact_bootstrap(first, second, bootstrap=2000):
    """Condition on the trained seeds; resample paired IDs within combinations."""
    differences = []
    for a, b in zip(first, second):
        for key in ('identities', 'targets', 'indices'):
            np.testing.assert_array_equal(a[key], b[key])
        differences.append(a['correct'].astype(float) - b['correct'].astype(float))
    delta_per_identity = np.mean(differences, axis=0)
    y = first[0]['targets']
    code = y.astype(np.int64) @ (1 << np.arange(6))
    strata = [np.flatnonzero(code == c) for c in np.unique(code)]
    rng = np.random.default_rng(20261102)
    results = []
    for start in range(0, bootstrap, 100):
        size = min(100, bootstrap - start)
        index = np.concatenate([rng.choice(g, (size, len(g)), replace=True) for g in strata], axis=1)
        results.extend(delta_per_identity[index].mean(1).tolist())
    return {'mean_delta': float(delta_per_identity.mean()),
            'per_seed_delta': [float(x.mean()) for x in differences],
            'identity_conditional_95_ci': np.quantile(results, [.025, .975]).tolist(),
            'bootstrap_replicates': bootstrap,
            'interpretation': 'Paired sample-ID uncertainty conditional on these two trained seeds; not five-seed confirmation.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args();suite = args.suite.resolve();out = args.output.resolve()
    if not (suite / 'completion.json').exists():
        raise RuntimeError('Wait for every frozen job and power-view evaluation to finish')
    out.mkdir(parents=True, exist_ok=True)
    plan = read(suite / 'plan.json');rows = [];predictions = {a: [] for a in ARMS}
    power = {a: [] for a in ARMS};histories = {};diagnostics = {};fpr_results = {};per_label = {}
    reference = None
    for job in plan['jobs']:
        directory = Path(job['output']);status = read(directory / 'status.json')
        phase = read(suite / 'phases' / (job['id'] + '.json'))
        if status['device'] != 'cpu' or status['world_size'] != 1 or phase['phase'] != 'completed':
            raise ValueError('Incomplete or non-CPU job')
        e = read(directory / 'evaluation/development_test/metrics.json')
        if e['split_sha256'] != plan['protocol']['data_split_sha256']:
            raise ValueError('Different dataset split')
        fixed, cal = e['metrics']['fixed_0.5'], e['metrics']['calibrated'];w = cal['weak_component']
        per_label[job['id']] = w['per_label']
        row = {'arm': job['arm'], 'seed': job['seed'], 'parameters': status['parameters'],
               'best_epoch': status['best_epoch'], 'train_seconds': status['elapsed_seconds'],
               'selection_check_weak_aupr': status['best_selection_score'],
               'worker_seconds': phase['worker_elapsed_seconds'], 'fixed_exact': fixed['subset_accuracy'],
               'fixed_macro_f1': fixed['macro_f1'], 'exact': cal['subset_accuracy'],
               'macro_f1': cal['macro_f1'], 'macro_aupr': cal['macro_aupr'],
               'weak_aupr': w['weak_macro_aupr'], 'weak_recall': w['weak_macro_recall'],
               'tail_weak_recall': w['tail_weak_recall'],
               'mean_negative_fpr': float(np.mean([x['negative_fpr'] for x in w['per_label'].values()])),
               'checkpoint': str(directory / 'best.pt'), 'checkpoint_sha256': e['checkpoint_sha256'],
               'thresholds': e['decision']['thresholds']}
        rows.append(row)
        z = dict(np.load(directory / 'evaluation/development_test/predictions.npz'))
        if reference is None:reference = z
        else:
            for key in ('identities', 'targets', 'indices'):
                np.testing.assert_array_equal(z[key], reference[key])
        probs = 1 / (1 + np.exp(-np.clip(z['logits'].astype(np.float64), -40, 40)))
        z['correct'] = np.all((probs >= np.asarray(row['thresholds'])) == z['targets'], axis=1)
        z['prediction'] = probs >= np.asarray(row['thresholds'])
        if abs(z['correct'].mean() - row['exact']) > 1e-12:
            raise ValueError('Saved exact disagrees with frozen predictions')
        predictions[job['arm']].append(z)
        for gap in (0, 6, 12, 18, 24):
            p = read(directory / f'evaluation/development_test/gap_{gap}/metrics.json')
            if p['decision'] != e['decision'] or p['checkpoint_sha256'] != e['checkpoint_sha256']:
                raise ValueError('Power-view decision/checkpoint was not frozen')
            power[job['arm']].append({'seed': job['seed'], 'gap': gap, 'metrics': p['metrics']['calibrated']})
        histories[job['id']] = read(directory / 'history.json')
        if (directory / 'query_diagnostics.json').exists():
            diagnostics[job['id']] = read(directory / 'query_diagnostics.json')
        if (directory / 'evaluation/fpr_controlled/metrics.json').exists():
            fpr_results[job['id']] = read(directory / 'evaluation/fpr_controlled/metrics.json')
    fields = ['exact', 'macro_f1', 'weak_aupr', 'weak_recall', 'tail_weak_recall',
              'mean_negative_fpr', 'train_seconds', 'worker_seconds', 'fixed_exact', 'selection_check_weak_aupr']
    aggregate = {}
    for arm in ARMS:
        selected = [r for r in rows if r['arm'] == arm]
        aggregate[arm] = {'n_seeds': len(selected), 'parameters': selected[0]['parameters'],
                          **{key: {'mean': mean([r[key] for r in selected]),
                                   'std': stdev([r[key] for r in selected])} for key in fields}}
    comparisons = {f'multiscale_minus_{arm}': paired_exact_bootstrap(predictions['multiscale'], predictions[arm])
                   for arm in ('baseline', 'single', 'capacity')}
    comparisons['single_minus_baseline'] = paired_exact_bootstrap(predictions['single'], predictions['baseline'])
    comparisons['capacity_minus_baseline'] = paired_exact_bootstrap(predictions['capacity'], predictions['baseline'])
    transitions = []
    for seed, multi, base in zip((2050,2051), predictions['multiscale'], predictions['baseline']):
        rescued = multi['correct'] & ~base['correct'];spoiled = base['correct'] & ~multi['correct']
        base_weak_miss = (base['weak_mask'] & ~base['prediction']).any(1)
        new_fp = (multi['prediction'] & ~multi['targets'].astype(bool)).any(1)
        transitions.append({'seed':seed,'rescued_samples':int(rescued.sum()),'spoiled_samples':int(spoiled.sum()),
                            'rescued_with_baseline_weak_miss':int((rescued & base_weak_miss).sum()),
                            'spoiled_with_new_false_positive':int((spoiled & new_fp).sum())})
    summary = {'created': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds'),
               'suite': str(suite), 'plan': plan, 'completion': read(suite / 'completion.json'),
               'rows': rows, 'aggregate': aggregate, 'paired_exact': comparisons,
               'power_views': power, 'query_diagnostics': diagnostics,
               'per_label_weak': per_label, 'error_transitions_multiscale_vs_baseline':transitions,
               'fpr_controlled': fpr_results,
               'memory_ablation': [read(p) for p in sorted((suite / 'diagnostics').glob('memory_ablation_seed*.json'))],
               'limitations': ['Two seeds, eight epochs, consumed development data.',
                              'CPU world_size=1 controls are newly trained; old two-GPU scores are not the control.',
                              'Identity bootstrap is conditional on the trained seeds.',
                              'Attention is diagnostic, not a causal proof.',
                              'No hyperparameters or thresholds were selected on development_test.']}
    (out / 'results.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
    with (out / 'results.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    draw_figures(out, aggregate, power, histories)
    report(out, summary)
    print(json.dumps({'report': str(out / 'report.html'), 'aggregate': aggregate,
                      'paired_exact': comparisons, 'wall_seconds': summary['completion']['wall_seconds']}, ensure_ascii=False, indent=2))


def draw_figures(out, aggregate, power, histories):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font_manager.fontManager.addfont('/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf')
    plt.rcParams.update({'font.family': ['DejaVu Sans', 'Droid Sans Fallback'],
                         'axes.unicode_minus': False, 'axes.spines.top': False,
                         'axes.spines.right': False, 'font.size': 10})
    colors = ['#65758b', '#4391a0', '#8d72a9', '#d46a45'];x = np.arange(4)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), layout='constrained')
    for ax, key, factor, title in [(axes[0], 'exact', 100, '六标签全部正确率（校准后）'),
                                    (axes[1], 'weak_aupr', 1, '弱分量 Macro-AUPR')]:
        values = [aggregate[a][key]['mean'] * factor for a in ARMS]
        errors = [aggregate[a][key]['std'] * factor for a in ARMS]
        ax.bar(x, values, yerr=errors, color=colors, capsize=4)
        ax.set_xticks(x, [NAMES[a] for a in ARMS], rotation=15);ax.set_title(title)
        ax.set_ylim(0, max(v+e for v,e in zip(values, errors))*1.18)
        for i, v in enumerate(values):ax.text(i, v + errors[i] + max(values)*.025, f'{v:.2f}' if factor==100 else f'{v:.4f}', ha='center')
    fig.suptitle('同一CPU协议，两个训练种子；误差线为种子标准差')
    fig.savefig(out / 'exact_weak.png', dpi=170);plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), layout='constrained')
    for arm, color in zip(ARMS, colors):
        for seed, style in [(2050, '-'), (2051, '--')]:
            h = histories[f'{arm}_seed{seed}']
            axes[0].plot([v['epoch'] for v in h], [v['validation']['weak_component']['weak_macro_aupr'] for v in h],
                         color=color, ls=style, label=f'{NAMES[arm]} {seed}')
        gaps = [0, 6, 12, 18, 24]
        scores = [mean([v['metrics']['subset_accuracy'] for v in power[arm] if v['gap']==g])*100 for g in gaps]
        axes[1].plot(gaps, scores, 'o-', color=color, label=NAMES[arm])
    axes[0].set_xlabel('epoch');axes[0].set_ylabel('check 弱 Macro-AUPR');axes[0].legend(fontsize=7,ncol=2)
    axes[1].set_xlabel('相对功率差（dB）');axes[1].set_ylabel('配对视图 Exact（%）');axes[1].set_xticks([0,6,12,18,24]);axes[1].legend(fontsize=8)
    fig.savefig(out / 'learning_power.png', dpi=170);plt.close(fig)


def report(out, summary):
    def table(headers, rows):
        return '<table><thead><tr>' + ''.join('<th>'+escape(str(x))+'</th>' for x in headers) + '</tr></thead><tbody>' + ''.join('<tr>'+''.join('<td>'+escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows) + '</tbody></table>'
    def pic(name):
        return '<img alt="实验图" src="data:image/png;base64,' + base64.b64encode((out/name).read_bytes()).decode()+'">'
    agg = summary['aggregate'];comp=summary['paired_exact'];wall=summary['completion']['wall_seconds']/60
    def number(arm, key, factor=100):
        a=agg[arm][key];return f"{a['mean']*factor:.2f} ± {a['std']*factor:.2f}"
    delta = comp['multiscale_minus_baseline'];capacity=comp['multiscale_minus_capacity']
    if (delta['mean_delta'] > 0 and comp['multiscale_minus_single']['mean_delta'] < 0
            and capacity['mean_delta'] < 0):
        conclusion='最深层类别查询获得初步收益；多层查询虽然平均Exact略高于CPU基线，但低于单层查询及容量对照，且弱AUPR低于基线。本轮未验证额外多层证据的收益，暂不采用多层拼接作为性能改进。'
    elif all(d>0 for d in delta['per_seed_delta']) and all(d>0 for d in capacity['per_seed_delta']):
        conclusion='多层查询在这两个种子上均超过CPU基线与容量对照，方向值得扩展验证；尚不足以宣称稳定泛化增益。'
    elif delta['mean_delta']>0:
        conclusion='多层查询的平均Exact高于CPU基线，但种子一致性或相对容量对照的证据不足，暂不能确认收益来自多层证据。'
    else:
        conclusion='当前8轮、双种子协议没有验证多层查询提升整体Exact；不应将结构接入成功写成性能提升。'
    text='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>多层类别查询CPU验证</title><style>
body{font:16px/1.7 system-ui,sans-serif;background:#f5f8fa;color:#223448;margin:0}main{max-width:1100px;margin:30px auto;padding:36px;background:white;border-radius:14px}h1,h2{color:#155e75}table{border-collapse:collapse;width:100%;margin:18px 0;font-size:14px;table-layout:fixed}td,th{border:1px solid #d7e1e8;padding:9px;vertical-align:top;overflow-wrap:anywhere}th{background:#155e75;color:white}tr:nth-child(even){background:#f0f6f7}img{width:100%;height:auto}.note{padding:18px;background:#edf5f6;border-left:5px solid #197c89}code{font-size:13px;overflow-wrap:anywhere}a{color:#16728c}</style><main>'''
    text+='<h1>第一条路线：多层复数特征与类别查询的CPU验证</h1>'
    text+=f'<p>报告生成时间：{escape(summary["created"])}；8个作业，32个物理核，主训练与测评墙钟时间{wall:.1f}分钟，包含选模、阈值拟合和五档视图。补充FPR、删除诊断与报告生成另计。</p>'
    text+='<p class="note">'+conclusion+'</p>'
    text+='<h2>对照是否合理</h2><p>四组各两个种子2050/2051；相同fit身份、8轮、全局batch32、BCE、AdamW、FP32、CPU world_size=1。所有query模型保留相同初始化的DFMS全局分支，类别查询只接收时序证据。单层读取第4层，多层读取第2/3/4层，时间中心统一到同一PRI；保留实虚部，未强制不同类注意力互斥。容量对照与多层模型仅差176个参数。</p>'
    text+='<p>check弱AUPR选择checkpoint；calibration逐类F1拟合阈值；同一冻结checkpoint/阈值评估混合开发池和五档配对视图。数据身份、行索引、标签及划分哈希均一致，测评未修改checkpoint。旧GPU成绩不承担本次CPU对照。</p>'
    text+=table(['模型','参数','Exact %','Macro-F1 %','弱AUPR','弱宏召回 %','平均FPR %','训练分钟'],[
        [NAMES[a],agg[a]['parameters'],number(a,'exact'),number(a,'macro_f1'),number(a,'weak_aupr',1),number(a,'weak_recall'),number(a,'mean_negative_fpr'),number(a,'train_seconds',1/60)] for a in ARMS])
    text+='<p>±为两个种子的样本标准差。平均FPR按六类缺席误报率等权汇总，不能替代逐类误报检查。</p>'+pic('exact_weak.png')
    text+='<h2>多层查询的配对差值</h2>'
    text+=table(['比较','平均Exact差（百分点）','各种子差（百分点）','条件于已训练模型的身份95%区间'],[
        ['多层 − '+NAMES[a],f"{comp['multiscale_minus_'+a]['mean_delta']*100:+.3f}",', '.join(f'{v*100:+.3f}' for v in comp['multiscale_minus_'+a]['per_seed_delta']),', '.join(f'{v*100:+.3f}' for v in comp['multiscale_minus_'+a]['identity_conditional_95_ci'])] for a in ['baseline','single','capacity']])
    text+='<p>区间按组合分层、基础身份成对重采样2000次，并对两种子结果平均；它只描述给定这两组训练模型时的样本不确定性，不代表训练随机性已充分验证。</p>'+pic('learning_power.png')
    text+='<p>补充参照：最深层查询相对基线的平均Exact差为'+f"{comp['single_minus_baseline']['mean_delta']*100:+.3f}"+'个百分点；其双种子check选模弱AUPR均值为'+f"{agg['single']['selection_check_weak_aupr']['mean']:.5f}"+'，基线为'+f"{agg['baseline']['selection_check_weak_aupr']['mean']:.5f}"+'，多层为'+f"{agg['multiscale']['selection_check_weak_aupr']['mean']:.5f}"+'。这仍属于开发比较，不按报告集最高单个seed选择部署权重。</p>'
    text+=table(['功率差 dB']+[NAMES[a]+' Exact %' for a in ARMS],[
        [gap]+[f"{mean(v['metrics']['subset_accuracy'] for v in summary['power_views'][a] if v['gap']==gap)*100:.3f}" for a in ARMS]
        for gap in (0,6,12,18,24)])
    def power_exact(arm, gap):
        return mean(v['metrics']['subset_accuracy'] for v in summary['power_views'][arm] if v['gap']==gap)*100
    lower_gaps = [str(g) for g in (0,6,12,18,24) if power_exact('single',g)<power_exact('baseline',g)]
    text+='<p class="note">最深层查询低于基线的配对功率差档位为：'+('/'.join(lower_gaps) or '无')+' dB。24 dB时基线Exact为'+f"{power_exact('baseline',24):.2f}%"+'，最深层查询为'+f"{power_exact('single',24):.2f}%"+'。默认F1阈值下，弱宏召回分别为'+f"{agg['baseline']['weak_recall']['mean']*100:.2f}% / {agg['single']['weak_recall']['mean']*100:.2f}%"+'。整体Exact提升不能写成极弱分量问题已解决；FPR约束测评只提供另一决策工作点的比较。</p>'
    text+='<h2>逐类弱识别与错误转移</h2>'
    text+=table(['类别']+[NAMES[a]+'：弱AUPR / 召回% / FPR%' for a in ARMS],[
        [label]+[f"{mean(summary['per_label_weak'][f'{arm}_seed{s}'][label]['weak_aupr'] for s in (2050,2051)):.4f} / "+
                  f"{mean(summary['per_label_weak'][f'{arm}_seed{s}'][label]['weak_recall'] for s in (2050,2051))*100:.2f} / "+
                  f"{mean(summary['per_label_weak'][f'{arm}_seed{s}'][label]['negative_fpr'] for s in (2050,2051))*100:.2f}"
                  for arm in ARMS] for label in ['FMB','AMA','SawFM','SinFM','MFT','ISRJ']])
    text+=table(['种子','多层修复的错误样本','多层新增错误样本','修复中包含原弱漏检','新增错误中包含误报'],[
        [v['seed'],v['rescued_samples'],v['spoiled_samples'],v['rescued_with_baseline_weak_miss'],v['spoiled_with_new_false_positive']]
        for v in summary['error_transitions_multiscale_vs_baseline']])
    text+='<h2>逐种子原始结果</h2>'+table(['模型','种子','最佳轮','固定0.5 Exact %','校准Exact %','弱AUPR','checkpoint'],[
        [NAMES[r['arm']],r['seed'],r['best_epoch'],f"{r['fixed_exact']*100:.3f}",f"{r['exact']*100:.3f}",f"{r['weak_aupr']:.5f}",r['checkpoint']] for r in summary['rows']])
    text+='<h2>查询分支诊断（仅check角色）</h2>'
    for job,d in summary['query_diagnostics'].items():
        text+='<h3>'+escape(job)+'</h3>'
        text+='<p>实际残差系数：'+escape(', '.join(f'{x:.3f}' for x in d['gate_tanh']))+'；token数量：'+escape(str(d['memory_lengths']))+'。</p>'
        text+=table(['类别','弱正例的各层注意力权重占比','相对均匀token分配的富集倍数'],[
            [c,', '.join(f'{v:.3f}' for v in d['weak_attention_mass'][i]),', '.join(f'{v:.3f}' for v in d['weak_enrichment_over_uniform'][i])] for i,c in enumerate(d['classes'])])
    text+='<p>浅层token更多，不能仅凭注意力权重占比较大断言重要；富集倍数以各层token占比归一化。多数类别的富集倍数接近1，尚无清晰的尺度选择证据；这只能提出后续假设，不能据此确定性能瓶颈。注意力仍属于诊断性证据，不能独立证明机制。</p>'
    if summary['fpr_controlled']:
        text+='<h2>按校准集FPR目标拟合的补充测评</h2><p>只在calibration负例上拟合1%/5%目标的阈值；下表同时给出development-test实际FPR。这里没有改动主checkpoint、主F1阈值或选模规则。</p>'
        text+=table(['模型','种子','校准FPR目标','报告集实际平均FPR %','弱宏召回 %','Exact %'],[
            [NAMES[arm],seed,name,f"{np.mean([p['negative_fpr'] for p in value['development_test']['per_label'].values()])*100:.3f}",
             f"{value['development_test']['weak_macro_recall']*100:.3f}",f"{value['development_test']['exact']*100:.3f}"]
            for arm in ARMS for seed in (2050,2051)
            for name,value in summary['fpr_controlled'][f'{arm}_seed{seed}']['metrics'].items()])
    if summary['memory_ablation']:
        text+='<h2>冻结模型的查询记忆删除诊断（check）</h2><p>全局分支、模型权重、阈值均固定，仅删除查询分支读取的部分层token。该操作改变了查询上下文，只说明模型对某层的依赖；独立训练的单层对照才是架构增益比较。</p>'
        text+=table(['种子','保留的层','check Exact %','弱AUPR','弱宏召回 %'],[
            [d['seed'],str(v['kept_levels']),f"{v['exact']*100:.3f}",f"{v['weak_aupr']:.5f}",f"{v['weak_recall']*100:.3f}"]
            for d in summary['memory_ablation'] for v in d['outcomes'].values()])
    text+='<h2>边界与后续决策</h2><p>本轮只有两个种子、8个epoch，并复用已消费开发数据；没有新锁定测试。若结果不一致，应如实记录；若方向入围，再另做五种子与新身份确认。任何后续训练、阈值和结构选择不得使用本轮development-test继续调参。</p>'
    text+='<p>后续优先保留最深层查询作为候选，验证五种子和新身份上的整体Exact与大功率差分层收益。若继续研究多层结构，可在训练/check协议内检验逐层归一化权重或层间门控，避免token数量直接决定层贡献；本轮没有验证这些改动有效，不据报告集结果追加调参。</p>'
    text+='<p>源代码快照、配置、数据划分、预估耗时、训练计时、模型权重、冻结阈值与全部预测位于：</p><code>'+escape(summary['suite'])+'</code>'
    text+='<p>机器可读附件：<a href="results.json">results.json</a> · <a href="results.csv">results.csv</a></p></main></html>'
    (out/'report.html').write_text(text)


if __name__=='__main__':main()
