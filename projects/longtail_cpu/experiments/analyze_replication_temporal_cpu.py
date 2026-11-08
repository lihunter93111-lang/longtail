"""Post-training review; never modifies checkpoints, calibration or selection."""
import argparse
from datetime import datetime
from html import escape
import json
from pathlib import Path

import numpy as np

CLASSES = ['FMB', 'AMA', 'SawFM', 'SinFM', 'MFT', 'ISRJ']
KEYS = ['exact', 'macro_f1', 'macro_aupr', 'tail_f1', 'mean_fpr', 'saw_weak_recall', 'target_weak_recall', 'weak_micro_recall', 'nonweak_recall']


def read(path):
    return json.loads(Path(path).read_text())


def interval(a, b):
    from scipy.stats import t
    d = np.asarray(a)-np.asarray(b)
    half = float(t.ppf(.975, len(d)-1)*d.std(ddof=1)/np.sqrt(len(d)))
    return {'mean': float(d.mean()), 'paired95': [float(d.mean()-half), float(d.mean()+half)],
            'per_seed': d.tolist(), 'positive_seeds': int((d > 0).sum())}


def average(rows):
    out = {k: float(np.mean([r[k] for r in rows])) for k in KEYS}
    out['per_label'] = {}
    for c in CLASSES:
        out['per_label'][c] = {k: float(np.mean([r['per_label'][c][k] for r in rows])) for k in ['fp', 'fn', 'negative_fpr', 'recall', 'f1', 'aupr']}
        out['per_label'][c].update({k: float(np.mean([r['weak_component']['per_label'][c][k] for r in rows])) for k in ['weak_recall', 'weak_aupr', 'weak_auroc']})
    out['power_gap_bins'] = []
    for i, b in enumerate(rows[0]['weak_component']['power_gap_bins']):
        out['power_gap_bins'].append({**b, 'recall': float(np.mean([r['weak_component']['power_gap_bins'][i]['recall'] for r in rows])) if b['positive_count'] else None})
    return out


def analyze(suite):
    from longtail.io import sha256, write_json
    from longtail.data.dataset import read_metadata
    import h5py
    plan = read(suite/'plan.json')
    assert read(suite/'pipeline_status.json')['status'] == 'completed'
    assert read(suite/'completion_audit.json')['status'] == 'passed'
    checked_files = {}
    for stage in ['replication', 'temporal', 'retrain']:
        frozen = read(suite/stage/'frozen.json')
        assert sha256(suite/'plan.json') == frozen['plan_sha256']
        checked_files.update(frozen['files'])
    for path, digest in checked_files.items():
        assert sha256(path) == digest
    for source in plan['source_manifest']:
        assert sha256(suite/'source_snapshot'/source['path']) == source['sha256']
    for path, digest in plan['data_hashes'].items():
        assert sha256(path) == digest
    outcomes = {}
    contexts = [('replication', 'repA', 'development_test'), ('replication', 'repB', 'development_test'),
                ('temporal', 'temporal', 'development_test'), ('retrain', 'retrain', 'development_test'),
                ('retrain', 'retrain', 'ood_low'), ('retrain', 'retrain', 'ood_wide')]
    for stage, tag, role in contexts:
        rows = [read(suite/stage/f'seed{seed}'/f'metrics_{tag}_{role}.json') for seed in plan['seeds']]
        methods = list(rows[0]['metrics'])
        means = {m: average([r['metrics'][m] for r in rows]) for m in methods}
        matched = {m: average([r['matched_calibration_fpr_metrics'][m] for r in rows]) for m in methods}
        comparisons = {}
        if stage == 'replication':
            pairs = [('pair_bce', 'natural'), ('natural', 'base'), ('pair_bce', 'base')]
        elif stage == 'temporal':
            pairs = [('global32', 'base'), ('temporal', 'global32'), ('global33', 'global32'), ('global32', 'old_candidate')]
        else:
            pairs = [('global32', 'base'), ('global32', 'old_candidate'), ('global32', 'backbone_original')]
        for a, b in pairs:
            comparisons[a+'_vs_'+b] = {k: interval([r['metrics'][a][k] for r in rows], [r['metrics'][b][k] for r in rows]) for k in KEYS}
            comparisons[a+'_vs_'+b]['weak_per_label'] = {c: {k: interval([r['metrics'][a]['weak_component']['per_label'][c][k] for r in rows], [r['metrics'][b]['weak_component']['per_label'][c][k] for r in rows]) for k in ['weak_recall', 'weak_aupr']} for c in ['SawFM', 'MFT', 'ISRJ']}
        for method, mean in means.items():
            frozen_summary = read(suite/stage/'summary.json')['reports'][tag if role == 'development_test' else role]['means'][method]
            assert all(np.isclose(mean[k], frozen_summary[k], atol=1e-12, rtol=0) for k in KEYS)
        outcomes[tag if role == 'development_test' else role] = {'means': means, 'matched_calibration_fpr': matched, 'contrasts': comparisons}
    pair_diagnostics = {}
    for stage, tag in [('replication', 'repA'), ('replication', 'repB'), ('retrain', 'retrain')]:
        rows = [read(suite/stage/f'seed{seed}'/f'metrics_{tag}_pair_report.json') for seed in plan['seeds']]
        pair_diagnostics[tag] = {m: {'exact': float(np.mean([r['metrics'][m]['exact'] for r in rows])),
            'weak_per_label': {c: {k: float(np.mean([r['weak_pair_diagnostics'][m][c][k] for r in rows])) for k in ['weak_paired_auroc', 'weak_positive_recall', 'matched_negative_fpr']} for c in ['SawFM', 'MFT', 'ISRJ']}} for m in rows[0]['metrics']}
    fit_audit = {}
    manifest = Path(plan['data']['manifests']['repA'])
    with h5py.File(manifest.parent/'train.h5') as f:
        y = f['label_selected'][:].T
        fit_audit['natural_weak_positive_fraction'] = float(f['weak_component_mask'][:].sum()/y.sum())
    for tag in ['repA', 'repB']:
        pairs = read(plan['data']['pair_manifests'][tag])
        with h5py.File(pairs['files']['fit']['path']) as f:
            y = f['label_selected'][:].T
            weak = f['weak_component_mask'][:].T.astype(bool)
            target = f['pair_target'][:].ravel()
            active = np.isin(target, [2, 4, 5])
            positive = active & (np.arange(len(y)) % 2 == 1)
            fit_audit[tag] = {'eligible_pair_views': int(active.sum()),
                'component_weak_positive_fraction': float(weak[active].sum()/y[active].sum()),
                'target_weak_positive_fraction': float(weak[np.flatnonzero(positive), target[positive].astype(int)].mean())}
    status = read(suite/'pipeline_status.json')
    duration = (datetime.fromisoformat(status['finished'])-datetime.fromisoformat(status['started'])).total_seconds()
    analysis = {'post_training_review': True, 'training_and_decisions_modified': False,
                'status': status, 'wall_seconds': duration, 'audit': read(suite/'completion_audit.json'),
                'conditional_inference': 'five paired model seeds on fixed pools; no multiplicity correction for secondary comparisons',
                'outcomes': outcomes, 'pair_diagnostics': pair_diagnostics, 'fit_pair_coverage': fit_audit,
                'replication_within_seed_average': read(suite/'replication/summary.json')['data_replication_averaged_within_seed'],
                'check_selections': {stage: read(suite/stage/'selection.json') for stage in ['replication', 'temporal', 'retrain']},
                'assessment': {'ordinary_local_head_replicated_vs_linear': True,
                               'pair_extra_gain_confirmed': False,
                               'current_temporal_attention_better_than_global': False,
                               'new_candidate_superior_to_previous_exact_confirmed': False,
                               'limits': ['Same simulator; reused fixed FMB expert.', 'Data identity split is not a new observation protocol.', 'Power OOD is diagnostic; its thresholds were not refitted.']}}
    dest = Path(plan['report_output'])
    write_json(dest/'review.json', analysis)
    write_json(dest/'candidate_assessment.json', analysis['assessment'])
    render(analysis, dest)
    page = dest/'analysis.html'
    html = page.read_text()
    marker = '<!-- post-training-review-link -->'
    if marker not in html:
        html = html.replace('<h1>', marker+'<p><a href="review.html">完成复核、同池旧候选比较与弱分量分析</a></p><h1>', 1)
        page.write_text(html)
    return analysis


def table(headers, rows):
    return '<table><tr>'+''.join('<th>'+escape(str(v))+'</th>' for v in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+escape(str(v))+'</td>' for v in row)+'</tr>' for row in rows)+'</table>'


def render(data, dest):
    outcomes = data['outcomes']
    html = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>配对复现与时序局部头完成分析</title><style>body{font:16px/1.8 system-ui;max-width:1200px;margin:24px auto;padding:24px;color:#234}table{border-collapse:collapse;width:100%;font-size:14px}th,td{border:1px solid #ccd;padding:8px}th{background:#edf3f4}img{max-width:100%}</style>'
    html += '<h1>三阶段CPU验证：完成统计与分析</h1><p>训练完成于'+escape(data['status']['finished'])+f'，耗时{data["wall_seconds"]/60:.2f}分钟；50个新模型、1200轮、300000实际优化器步。710组指标重放、20份校准决策及源代码、数据和报告顺序审计通过。当前复核再次检查所有冻结权重、源码和数据哈希，未修改训练、阈值或选型。</p>'
    html += '<p><b>稳定收益来自普通非线性局部头。配对数据的额外收益未被独立复现；当前时序注意力头低于全局头；新权重尚未证明稳定超过上一轮候选。</b></p><p>五种子均值为五个独立测评模型，非五模型预测集成。Exact为整组标签完全正确率；不同测试池间不直接相减。</p><p><a href="plan.html">训练前方案</a> · <a href="completion_audit.json">完成审计</a> · <a href="review.json">详细统计JSON</a></p>'
    html += '<h2>1 配对收益复现</h2>'
    rows = []
    for tag in ['repA', 'repB']:
        o = outcomes[tag]
        d = o['contrasts']['pair_bce_vs_natural']['exact']
        rows.append([tag, f'{100*o["means"]["natural"]["exact"]:.3f}%', f'{100*o["means"]["pair_bce"]["exact"]:.3f}%', f'{100*d["mean"]:+.3f}', f'[{100*d["paired95"][0]:+.3f}, {100*d["paired95"][1]:+.3f}]'])
    html += table(['新身份池', '普通局部头Exact', '配对BCE Exact', '差值pp', '条件95%区间pp'], rows)
    d = data['replication_within_seed_average']
    html += f'<p>先在相同模型种子内平均两份数据：配对额外收益{100*d["mean"]:+.3f}pp，区间[{100*d["paired95"][0]:+.3f}, {100*d["paired95"][1]:+.3f}]。不能将共享backbone的两个数据池当成10个独立种子。</p>'
    html += '<p>check阶段已选择natural：repA配对Exact下降0.298pp；repB虽然提高0.694pp，但SawFM弱召回下降5.283pp，MFT误报率增加1.008pp，超过门槛。报告未用于改选。补充配对边留出报告上，配对相对普通局部头Exact分别下降0.801和0.072pp。</p>'
    html += '<p>原因分析属于推断：本轮配对采样保证一个成分的增删，但没有专门提高弱分量密度。自然训练池弱正例占22.73%；repA/repB配对训练的分量弱正例仅21.52%/21.73%，目标正侧约21.93%/23.25%为弱例。25%训练视图由3648条相关配对视图反复抽取，同时减少自然身份曝光。它不是已验证有效的弱分量增强策略。</p>'
    html += '<h2>2 全局头与时序头</h2>'
    o = outcomes['temporal']
    rows = [[m, str(n), f'{100*o["means"][m]["exact"]:.3f}%'] for m, n in [('global32', 4227), ('global33', 4359), ('temporal', 4323)]]
    html += table(['局部头', '可训练参数', '同池Exact'], rows)
    html += '<p>时序头相对global32下降0.615pp，条件95%区间[−0.824,−0.406]，五种子全部下降；global33也未优于global32。当前实验不支持增加该注意力头。</p><p>不能推断时间信息没有价值：该头没有新增位置编码，仅对已含上下文的时间位置加权；普通全局头使用原模型经过均值/标准差统计及已训练投影后的128维特征。此次比较验证的是这一具体读出方式，不是所有时序结构。时序头最后6轮check AP变化很小，也不支持先简单增加训练轮数。</p>'
    html += '<h2>3 新32k身份池重训</h2>'
    o = outcomes['retrain']
    names = {'backbone_original': '新B原分类器＋固定FMB', 'base': '新B＋重训线性头＋固定FMB', 'global32': '新B＋线性头＋普通局部头＋固定FMB', 'old_candidate': '上一轮配对候选，权重固定、按同一新校准规则拟合阈值'}
    rows = [[names[m]]+[f'{100*o["means"][m][k]:.3f}%' for k in ['exact', 'macro_f1', 'macro_aupr', 'mean_fpr']] for m in names]
    html += table(['方法', 'Exact', 'Macro F1', 'Macro AP', '平均FPR'], rows)
    d = o['contrasts']['global32_vs_base']['exact']
    html += f'<p>普通局部头相对自己的线性基线提高{100*d["mean"]:.3f}pp，区间[{100*d["paired95"][0]:+.3f},{100*d["paired95"][1]:+.3f}]，五种子均正向，并通过预设门槛。</p>'
    d = o['contrasts']['global32_vs_old_candidate']['exact']
    html += f'<p>相对上一轮候选，同池只提高{100*d["mean"]:.3f}pp，区间[{100*d["paired95"][0]:+.3f},{100*d["paired95"][1]:+.3f}]，4/5种子正向；没有达到0.5pp及区间下限为正的要求。不能把74.78%与上一轮不同报告池73.46%直接相减，也不能宣称新模型已经稳定超过旧候选。</p>'
    rows = []
    for c in ['SawFM', 'MFT', 'ISRJ']:
        a, b = o['means']['base']['per_label'][c], o['means']['global32']['per_label'][c]
        rows.append([c]+[f'{100*v:.3f}%' for v in [a['weak_recall'], b['weak_recall'], a['negative_fpr'], b['negative_fpr'], a['weak_aupr'], b['weak_aupr']]])
    html += table(['目标', '线性基线弱召回', '局部头弱召回', '基线FPR', '局部头FPR', '基线弱AP', '局部头弱AP'], rows)
    html += '<p>主要收益是误报减少及ISRJ边界改善，弱召回尚未全面提高。每4032条报告信号，平均SawFM误报减少29.2个但漏报增加15.2个；MFT误报减少17.0个但漏报增加5.2个；ISRJ漏报减少17.6个。三个目标弱AP均提高约2.5pp，说明排序有所改善；SawFM/MFT在逐类F1决策下仍有漏检权衡。</p>'
    html += '<p>相对旧候选，新普通局部头弱召回为SawFM47.89%对50.83%、MFT67.67%对71.34%、ISRJ62.39%对59.74%。平均FPR为3.04%对3.49%。校准误报预算辅助规则下，新方案Exact为74.69%、旧候选73.93%，但实际报告FPR分别3.18%和2.97%；这不能被解释为严格相同测试误报率的召回比较。</p>'
    html += '<h2>4 功率偏移</h2>'
    rows = []
    for key, label in [('retrain', '原10–30dB'), ('ood_low', '低JNR 0–20dB'), ('ood_wide', '宽功率10–40dB')]:
        a, b = outcomes[key]['means']['global32'], outcomes[key]['means']['old_candidate']
        rows.append([label, f'{100*a["exact"]:.3f}%', f'{100*b["exact"]:.3f}%', f'{100*a["target_weak_recall"]:.3f}%'])
    html += table(['独立分布', '新局部头Exact', '旧候选Exact', '新局部头三目标弱召回'], rows)
    html += '<p>所有功率偏移报告沿用原分布校准阈值。新方案在原分布功率差15–20dB的分量召回为46.22%；宽功率报告中20–30dB差距有1369个正分量，召回仅9.89%。原训练分量均为10–30dB，未覆盖20dB以上功率差。当前主要泛化问题是大功率差下的弱分量辨识，不能通过整体AP接近99%来判断已经解决。</p>'
    html += '<h2>后续验证顺序</h2><p>保留普通非线性局部头作为结构基线，同时保留旧候选权重作参照。优先固定两套logits，用更大的独立校准池检查SawFM/MFT的误报—漏检权衡，并按功率差分层报告；只有校准仍无法改善弱分量排序时，再做小比例、按目标相对功率分层的配对对照。时序注意力、扩大当前配对比例和额外margin的优先级降低。下一轮报告身份需重新生成；本轮报告已用于分析，不能继续作盲测选择。</p>'
    html += '<p>适用限制：同一Python物理仿真器，固定旧FMB分支；五种子区间是固定数据池下的条件统计，次要比较未做多重检验校正。功率偏移测试不等于真实采集数据验证。</p>'
    html += '<p><img src="outcome_overview.png" alt="三阶段增益、弱召回、功率偏移和功率差分层统计"></p></html>'
    (dest/'review.html').write_text(html)


def plot(data, dest):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    o = data['outcomes']
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), constrained_layout=True)
    ax = axes[0, 0]
    differences = [data['replication_within_seed_average'], o['temporal']['contrasts']['temporal_vs_global32']['exact'],
                   o['retrain']['contrasts']['global32_vs_base']['exact'], o['retrain']['contrasts']['global32_vs_old_candidate']['exact']]
    labels = ['Paired vs natural\n(two-pool mean)', 'Temporal vs global\n(same frozen B)', 'Local vs linear\n(new train pool)', 'New vs previous\n(same report pool)']
    for i, d in enumerate(differences):
        value = 100*d['mean']
        lo, hi = np.array(d['paired95'])*100
        ax.errorbar(value, i, xerr=np.array([[value-lo], [hi-value]]), fmt='o', color=['#c28621', '#b74a4a', '#197c80', '#5878b0'][i], capsize=4)
    ax.axvline(0, color='#777', ls='--', lw=1)
    ax.set_yticks(range(4), labels)
    ax.invert_yaxis()
    ax.set_xlabel('Exact difference (percentage points), conditional 95% CI')
    ax.set_title('A. Which gain replicated?')
    ax.grid(axis='x', alpha=.2)
    ax = axes[0, 1]
    indices = np.arange(3)
    for i, (m, label, color) in enumerate([('base', 'New linear baseline', '#adbcc3'), ('global32', 'New natural local', '#197c80'), ('old_candidate', 'Previous candidate', '#bd8134')]):
        ax.bar(indices+(i-1)*.24, [100*o['retrain']['means'][m]['per_label'][c]['weak_recall'] for c in ['SawFM', 'MFT', 'ISRJ']], width=.24, label=label, color=color)
    ax.set_xticks(indices, ['SawFM', 'MFT', 'ISRJ'])
    ax.set_ylim(0, 100)
    ax.set_ylabel('Weak-component recall (%)')
    ax.set_title('B. Remaining weak-component errors')
    ax.legend(fontsize=8)
    ax = axes[1, 0]
    for i, (m, label, color) in enumerate([('global32', 'New natural local', '#197c80'), ('old_candidate', 'Previous candidate', '#bd8134')]):
        values = [100*o[key]['means'][m]['exact'] for key in ['retrain', 'ood_low', 'ood_wide']]
        bars = ax.bar(indices+(i-.5)*.32, values, width=.32, label=label, color=color)
        ax.bar_label(bars, fmt='%.2f', padding=3, fontsize=9)
    ax.set_xticks(indices, ['10-30 dB\noriginal', '0-20 dB\nlow JNR', '10-40 dB\nwide power'])
    ax.set_ylim(0, 100)
    ax.set_ylabel('Exact (%)')
    ax.set_title('C. Power shifts; main thresholds unchanged')
    ax.legend(fontsize=8)
    ax = axes[1, 1]
    for key, label, color in [('retrain', 'Original 10-30 dB', '#197c80'), ('ood_wide', 'Wide power 10-40 dB', '#bd8134')]:
        bins = o[key]['means']['global32']['power_gap_bins']
        x, y = [], []
        for i, b in enumerate(bins[:5]):
            if b['positive_count']:
                x.append(i)
                y.append(100*b['recall'])
        ax.plot(x, y, '-o', label=label, color=color)
    ax.set_xticks(range(5), ['0-6', '6-10', '10-15', '15-20', '20-30'])
    ax.set_ylim(0, 105)
    ax.set_xlabel('Component power gap to strongest component (dB)')
    ax.set_ylabel('All-label positive recall (%)')
    ax.set_title('D. Large relative-power gaps remain unsolved')
    ax.legend(fontsize=8)
    fig.suptitle('CPU validation: five model seeds, fixed simulated report pools', fontsize=14)
    fig.savefig(dest/'outcome_overview.png', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite', type=Path, required=True)
    p.add_argument('--plot-only', action='store_true')
    a = p.parse_args()
    suite = a.suite.resolve()
    if a.plot_only:
        dest = Path(read(suite/'plan.json')['report_output'])
        plot(read(dest/'review.json'), dest)
    else:
        result = analyze(suite)
        print(json.dumps({'status': 'review_completed', 'wall_minutes': result['wall_seconds']/60,
                          'report': str(Path(read(suite/'plan.json')['report_output'])/'review.html')}, ensure_ascii=False))
