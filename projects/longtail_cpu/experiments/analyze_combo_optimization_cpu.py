"""Audit and render the completed 2026-11-03 CPU optimization experiment.

Only replay saved predictions and fixed calibration algorithms. No training,
new report inference, threshold tuning or modification of frozen decisions.
"""
import argparse
import base64
from datetime import datetime
from html import escape
import json
from pathlib import Path
import numpy as np

LABELS = ['FMB', 'AMA', 'SawFM', 'SinFM', 'MFT', 'ISRJ']
METHODS = ['bce', 'combo_sampler', 'mixed_sampler']
NAMES = {'bce': '自然采样BCE', 'combo_sampler': '纯组合采样', 'mixed_sampler': '50/50混合采样', 'hybrid': '预定义按类融合'}


def read(p):
    return json.loads(Path(p).read_text())


def write(p, v):
    Path(p).write_text(json.dumps(v, ensure_ascii=False, indent=2) + '\n')


def audit(suite):
    import h5py
    from longtail.io import sha256, fingerprint
    from longtail.engine.checkpoint import load_checkpoint
    from longtail.metrics import evaluate_metrics, probabilities, fit_thresholds
    from longtail.decision import constrained_exact_thresholds
    plan = read(suite / 'plan.json'); out = Path(plan['report_output'])
    r = read(out / 'results.json'); frozen = read(suite / 'frozen_decisions_ready.json')
    completion = read(suite / 'pipeline_completed.json')
    assert completion['status'] == 'completed' and completion['jobs_completed'] == 15
    assert sha256(suite / 'plan.json') == frozen['plan_sha256']
    for f in plan['source_manifest']:
        assert sha256(suite / 'source_snapshot' / f['path']) == f['sha256'], f['path']
    barrier = datetime.fromisoformat(frozen['finished'])
    manifest = read(plan['dataset']); data_root = Path(plan['dataset']).parent
    generation = read(data_root / 'generation_plan.json')
    split = read(suite / 'split.json'); roles = split['roles']
    for i, role in enumerate(['calibration', 'check', 'development_test']):
        for other in ['calibration', 'check', 'development_test'][i + 1:]:
            assert not set(roles[role]) & set(roles[other])
    for f in manifest['files'].values():
        assert sha256(data_root / f['path']) == f['sha256']
    with h5py.File(data_root / 'development.h5') as f:
        dev_ids = f['receiver_noise_seed'][:].ravel()
        dev_y = f['label_selected'][:].T
    report_ids = dev_ids[roles['development_test']]
    for p in generation['checked_identity_files']:
        with h5py.File(p) as f:
            assert not np.intersect1d(report_ids, f['receiver_noise_seed'][:].ravel()).size, p
    old_manifest_path = Path(generation['source_dataset_manifest'])
    old_manifest = read(old_manifest_path)
    assert manifest['files']['train']['sha256'] == old_manifest['files']['train']['sha256']
    with h5py.File(data_root / 'development.h5') as new, h5py.File(old_manifest_path.parent / old_manifest['files']['development']['path']) as old:
        for role in ['calibration', 'check']:
            for name in old:
                np.testing.assert_array_equal(new[name][:, roles[role]], old[name][:, roles[role]])
    counts = np.array(plan['train_combo_counts']); rare = np.flatnonzero(counts == counts.min()) + 1
    histories, selection, thresholds, per_label, coverage, violations = {}, {}, {}, {}, {}, []
    transfer = {m: {role: [] for role in ['calibration', 'check', 'development_test']} for m in METHODS}
    audit_jobs, reference, recomputed = [], {}, 0
    for job in plan['jobs']:
        path = Path(job['output']); m, seed = job['method'], job['seed']
        hold = frozen['jobs'][job['id']]
        assert sha256(path / 'best.pt') == hold['checkpoint_sha256']
        assert sha256(path / 'decisions_v2.json') == hold['decisions_sha256']
        assert read(path / 'split.json') == split
        config = read(path / 'config.json')
        assert config == read(job['config'])
        assert config['train']['device'] == 'cpu' and config['train']['loss'] == 'bce'
        assert config['train']['epochs'] == 24 and config['dataloader']['batch_size'] == 32
        assert config['train']['lr'] == 1e-4 and config['train']['weight_decay'] == 1e-4
        history = read(path / 'history.json'); status = read(path / 'status.json')
        assert len(history) == 24 and all(h['samples_seen'] == 8000 and h['sampler_padding'] == 0 for h in history)
        assert status['status'] == 'completed' and status['parameters'] == 257511
        last, _ = load_checkpoint(path / 'last.pt')
        assert last['epoch'] == 24 and {int(v['step']) for v in last['optimizer']['state'].values()} == {6000}
        del last
        best, _ = load_checkpoint(path / 'best.pt')
        best_epoch = int(np.argmax([h['validation']['macro_aupr'] for h in history])) + 1
        assert best['epoch'] == status['best_epoch'] == best_epoch
        assert best['decision'] == read(path / 'decision.json')
        del best
        for stage in ['train', 'report']:
            phase = read(suite / 'phases' / f'{job["id"]}_{stage}.json')
            assert phase['status'] == 'completed'
            assert phase['checkpoint_sha256'] == hold['checkpoint_sha256'] and phase['decisions_sha256'] == hold['decisions_sha256']
            if stage == 'train':
                assert datetime.fromisoformat(phase['finished']) <= barrier
            else:
                assert datetime.fromisoformat(phase['started']) >= barrier
        histories[job['id']] = [{'epoch': h['epoch'], 'loss': h['loss'], 'check_aupr': h['validation']['macro_aupr'],
                                 'epoch_seconds': h['epoch_seconds'], 'unique_identities': h.get('sampling', {}).get('unique_fit_identities', 8000)} for h in history]
        selection[job['id']] = {'best_epoch': best_epoch, 'elapsed_seconds': status['elapsed_seconds']}
        decisions = read(path / 'decisions_v2.json'); thresholds[job['id']] = decisions['thresholds']
        cal_path = path / 'evaluation/calibration/predictions.npz'
        assert sha256(cal_path) == decisions['calibration_predictions_sha256']
        cal = dict(np.load(cal_path))
        assert decisions['thresholds']['f1'] == fit_thresholds(cal['targets'], cal['logits'], None)
        fitted, meta = constrained_exact_thresholds(cal['targets'], cal['logits'])
        assert fitted == decisions['thresholds']['exact_constrained'] and meta == decisions['exact_fit']
        for role in ['calibration', 'check', 'development_test']:
            folder = path / 'evaluation' / role; z = dict(np.load(folder / 'predictions.npz'))
            assert np.isfinite(z['logits']).all()
            assert set(z['indices'].tolist()) == set(roles[role])
            np.testing.assert_array_equal(z['targets'], dev_y[z['indices']])
            np.testing.assert_array_equal(z['identities'], dev_ids[z['indices']])
            if role not in reference:
                reference[role] = z
            for k in ['targets', 'indices', 'identities']:
                np.testing.assert_array_equal(z[k], reference[role][k])
            result = read(folder / 'decision_metrics.json')
            assert result['checkpoint_sha256'] == hold['checkpoint_sha256'] and result['decisions_sha256'] == hold['decisions_sha256']
            standard = read(folder / 'metrics.json')
            assert standard['checkpoint_sha256'] == hold['checkpoint_sha256']
            assert standard['config_sha256'] == fingerprint(config)
            assert standard['split_sha256'] == plan['split_sha256'] and standard['split_role'] == role
            per_label.setdefault(role, {}).setdefault(m, {})
            for d, th in decisions['thresholds'].items():
                observed = result['metrics'][d]
                measured = evaluate_metrics(z['targets'], z['logits'], LABELS, th)
                for k in ['subset_accuracy', 'macro_f1', 'macro_aupr', 'hamming_loss']:
                    assert abs(observed[k] - measured[k]) < 1e-12
                for c in LABELS:
                    assert observed['per_label'][c] == measured['per_label'][c]
                code = z['targets'].astype(int) @ (1 << np.arange(6))
                correct = np.all((probabilities(z['logits']) >= th) == z['targets'], axis=1)
                assert abs(observed['rare_combo_exact'] - correct[np.isin(code, rare)].mean()) < 1e-12
                assert abs(observed['tail_f1'] - np.mean([measured['per_label'][c]['f1'] for c in ['AMA', 'SinFM']])) < 1e-12
                per_label[role][m].setdefault(d, []).append(measured['per_label'])
                if role != 'calibration':
                    row = next(row for row in r['rows'] if row['method'] == m and row['seed'] == seed and row['role'] == role and row['decision'] == d)
                    assert row['exact'] == measured['subset_accuracy'] and row['macro_f1'] == measured['macro_f1']
                recomputed += 1
            a, b = result['metrics']['f1'], result['metrics']['exact_constrained']
            transfer[m][role].append(b['subset_accuracy'] - a['subset_accuracy'])
            for c in LABELS:
                drop = a['per_label'][c]['recall'] - b['per_label'][c]['recall']
                rise = b['per_label'][c]['negative_fpr'] - a['per_label'][c]['negative_fpr']
                if drop > .03 + 1e-12 or rise > .02 + 1e-12:
                    violations.append({'method': m, 'seed': seed, 'role': role, 'label': c, 'recall_drop': drop, 'fpr_rise': rise})
        audit_jobs.append({'job': job['id'], 'optimizer_steps': 6000, 'best_epoch': best_epoch, **hold})
    assert not any(v['role'] == 'calibration' for v in violations)
    hybrid_changes = []
    for role in ['check', 'development_test']:
        per_label[role]['hybrid'] = {'hybrid_predeclared': []}
        for seed in range(2050, 2055):
            b = dict(np.load(suite / f'bce_seed{seed}/evaluation/{role}/predictions.npz'))
            c = dict(np.load(suite / f'combo_sampler_seed{seed}/evaluation/{role}/predictions.npz'))
            h = dict(np.load(suite / f'hybrid_seed{seed}/evaluation/{role}/predictions.npz'))
            np.testing.assert_array_equal(h['logits'][:, 0], c['logits'][:, 0])
            np.testing.assert_array_equal(h['logits'][:, 1:], b['logits'][:, 1:])
            for k in ['targets', 'indices', 'identities']:
                np.testing.assert_array_equal(h[k], reference[role][k])
            th = thresholds[f'bce_seed{seed}']['f1'].copy(); th[0] = .5
            metric = evaluate_metrics(h['targets'], h['logits'], LABELS, th)
            stored = read(suite / f'hybrid_seed{seed}/evaluation/{role}/decision_metrics.json')
            assert stored['source_checkpoints']['bce'] == frozen['jobs'][f'bce_seed{seed}']['checkpoint_sha256']
            assert stored['source_checkpoints']['combo'] == frozen['jobs'][f'combo_sampler_seed{seed}']['checkpoint_sha256']
            for key in ['subset_accuracy', 'macro_f1', 'macro_aupr']:
                assert stored['metrics']['hybrid_predeclared'][key] == metric[key]
            assert stored['metrics']['hybrid_predeclared']['per_label'] == metric['per_label']
            row = next(row for row in r['rows'] if row['method'] == 'hybrid' and row['seed'] == seed and row['role'] == role)
            assert row['exact'] == metric['subset_accuracy'] and row['macro_f1'] == metric['macro_f1']
            per_label[role]['hybrid']['hybrid_predeclared'].append(metric['per_label'])
            base_correct = ((probabilities(b['logits']) >= thresholds[f'bce_seed{seed}']['f1']) == b['targets']).all(1)
            hybrid_correct = ((probabilities(h['logits']) >= th) == h['targets']).all(1)
            hybrid_changes.append({'role': role, 'seed': seed, 'corrected_fraction': float((~base_correct & hybrid_correct).mean()),
                                   'lost_correct_fraction': float((base_correct & ~hybrid_correct).mean())})
            recomputed += 1
    for method in ['combo_sampler', 'mixed_sampler']:
        a = [v for v in r['sampling_audits'] if v['method'] == method]
        assert len(a) == 120 and len({v['draw_order_sha256'] for v in a}) == 120
        assert len({v['rng_seed'] for v in a}) == 120
        assert all(v['draws'] == 8000 and sum(v['combination_exposures']) == 8000 for v in a)
        if method == 'mixed_sampler':
            assert all(v['natural_draws_without_replacement'] == v['combo_draws_with_replacement'] == 4000 for v in a)
        coverage[method] = {'mean_unique': float(np.mean([v['unique_fit_identities'] for v in a])),
            'unique_range': [min(v['unique_fit_identities'] for v in a), max(v['unique_fit_identities'] for v in a)],
            'mean_label_fractions': (np.mean([v['label_exposures'] for v in a], axis=0) / 8000).tolist(),
            'distinct_draw_orders': 120, 'distinct_rng_seeds': 120}
    means = {role: {m: {d: {c: {k: float(np.mean([v[c][k] for v in records])) for k in ['f1', 'aupr', 'recall', 'negative_fpr']}
                           for c in LABELS} for d, records in ds.items()} for m, ds in methods.items()} for role, methods in per_label.items()}
    audit_result = {'status': 'passed', 'jobs': 15, 'epochs': 360, 'optimizer_steps': 90000,
        'frozen_source_files_verified': len(plan['source_manifest']), 'recomputed_metric_conditions': recomputed,
        'calibration_decisions_replayed': 15, 'fresh_report_identity_files_rechecked': len(generation['checked_identity_files']),
        'fresh_report_identity_overlap': 0, 'preserved_calibration_check_arrays_verified': True,
        'report_started_after_freeze': True, 'freeze_time': frozen['finished'], 'completion': completion,
        'all_calibration_constraints_satisfied': True, 'constraint_violations_on_held_out_roles': violations, 'job_checks': audit_jobs}
    analysis = {'suite': str(suite), 'aggregate': r['aggregate'], 'effects': r['effects'], 'per_label': means,
                'histories': histories, 'selection': selection, 'thresholds': thresholds, 'coverage': coverage,
                'exact_calibration_transfer': transfer, 'hybrid_error_changes': hybrid_changes,
                'audit': audit_result, 'limitations': r['limitations']}
    write(out / 'completion_audit.json', audit_result); write(out / 'analysis.json', analysis)
    print(json.dumps({'audit': 'passed', 'metric_conditions': recomputed, 'report': str(out), 'held_out_constraint_violations': violations}, ensure_ascii=False))


def render(suite):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font_manager.fontManager.addfont('/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf')
    plt.rcParams.update({'font.family': ['DejaVu Sans', 'Droid Sans Fallback'], 'axes.unicode_minus': False,
                         'axes.spines.top': False, 'axes.spines.right': False, 'font.size': 10})
    out = Path(read(suite / 'plan.json')['report_output']); a = read(out / 'analysis.json')
    agg = a['aggregate']['development_test']
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), layout='constrained')
    for ax, key, title, limits in zip(axes, ['exact', 'macro_f1'], ['完整六标签正确率 Exact (%)', 'Macro-F1 (%)'], [(54, 72), (90, 95)]):
        for d, off, color, name in [('fixed_0.5', -.24, '#64798f', '固定0.5'), ('f1', 0, '#198c96', 'F1校准'), ('exact_constrained', .24, '#c58b50', '受约束Exact校准')]:
            ax.bar(np.arange(3) + off, [agg[m][d][key]['mean'] * 100 for m in METHODS], width=.23, color=color, label=name,
                   yerr=[agg[m][d][key]['std'] * 100 for m in METHODS], capsize=3)
        h = agg['hybrid']['hybrid_predeclared'][key]
        ax.bar(3, h['mean'] * 100, width=.4, color='#925e9c', yerr=h['std'] * 100, capsize=3, label='冻结按类融合')
        ax.set_xticks(range(4), ['自然BCE', '纯组合', '50/50混合', '双模型融合']); ax.set_ylim(*limits); ax.set_title(title)
    axes[0].legend(fontsize=8, ncol=2)
    fig.suptitle('4032个新报告身份；五种子均值 ± 样本标准差')
    fig.savefig(out / 'decision_comparison.png', dpi=170); plt.close(fig)
    choices = [('combo_sampler', 'fixed_0.5'), ('mixed_sampler', 'f1'), ('hybrid', 'hybrid_predeclared')]
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.6), layout='constrained')
    for ax, metric, title in zip(axes, ['f1', 'aupr'], ['逐类F1差', '逐类AUPR差（阈值无关）']):
        values = np.array([[100 * (a['per_label']['development_test'][m][d][c][metric] - a['per_label']['development_test']['bce']['f1'][c][metric]) for c in LABELS] for m, d in choices])
        im = ax.imshow(values, cmap='RdBu', vmin=-9, vmax=9, aspect='auto')
        ax.set_xticks(range(6), LABELS); ax.set_yticks(range(3), ['纯组合+0.5', '混合+F1', '按类融合']); ax.set_title(title)
        for i in range(3):
            for j in range(6):
                ax.text(j, i, f'{values[i,j]:+.2f}', ha='center', va='center', color='white' if abs(values[i,j]) > 5 else '#123')
    fig.colorbar(im, ax=axes, label='相对BCE+F1的百分点差', shrink=.9)
    fig.savefig(out / 'per_label_tradeoffs.png', dpi=170); plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 3.8), layout='constrained')
    for j, (role, name, color) in enumerate([('calibration', 'calibration拟合池', '#bf8b54'), ('check', 'check开发池', '#70879d'), ('development_test', '新报告池', '#198c96')]):
        values = [a['exact_calibration_transfer'][m][role] for m in METHODS]
        ax.bar(np.arange(3) + (j - 1) * .24, [100*np.mean(v) for v in values], width=.23, color=color, label=name,
               yerr=[100*np.std(v, ddof=1) for v in values], capsize=3)
    ax.axhline(0, color='#9aa'); ax.set_xticks(range(3), ['自然BCE', '纯组合', '50/50混合'])
    ax.set(ylabel='Exact相对F1校准的变化 pp', title='受约束Exact校准：额外收益有限；误差条为五种子标准差'); ax.legend(fontsize=8)
    fig.savefig(out / 'calibration_transfer.png', dpi=170); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout='constrained')
    for m, color in zip(METHODS, ['#64798f', '#c08455', '#198c96']):
        curves = np.array([[h['check_aupr'] for h in a['histories'][f'{m}_seed{s}']] for s in range(2050, 2055)]) * 100
        avg, std = curves.mean(0), curves.std(0, ddof=1)
        axes[0].plot(range(1,25), avg, label=NAMES[m], color=color); axes[0].fill_between(range(1,25), avg-std, avg+std, alpha=.15, color=color)
        unique = np.array([[h['unique_identities'] for h in a['histories'][f'{m}_seed{s}']] for s in range(2050,2055)])
        axes[1].plot(range(1,25), unique.mean(0), label=NAMES[m], color=color)
    axes[0].set(xlabel='epoch', ylabel='check Macro-AUPR (%)', ylim=(90,99)); axes[1].set(xlabel='epoch', ylabel='每轮独立身份数', ylim=(3500,8500))
    axes[0].legend(fontsize=8); axes[1].legend(fontsize=8)
    fig.savefig(out / 'training_coverage.png', dpi=170); plt.close(fig)
    def table(headers, rows):
        return '<div class="scroll"><table><tr>'+''.join('<th>'+escape(str(v))+'</th>' for v in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+escape(str(v))+'</td>' for v in row)+'</tr>' for row in rows)+'</table></div>'
    def pic(name):
        return '<img alt="实验统计图" src="data:image/png;base64,'+base64.b64encode((out/name).read_bytes()).decode()+'">'
    def stat(m,d,k):
        v=agg[m][d][k];return f"{v['mean']*100:.3f} ± {v['std']*100:.3f}"
    text='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>CPU优化验证：结果审计与分析</title><style>body{font:16px/1.8 system-ui;color:#234;background:#f0f4f6;margin:0}main{max-width:1180px;margin:28px auto;padding:32px;background:white}h1,h2{color:#155e75}h2{margin-top:30px}table{border-collapse:collapse;width:100%;font-size:14px;margin:18px 0}td,th{border:1px solid #d6e0e6;padding:8px}th{background:#155e75;color:white}.scroll{overflow-x:auto}.note{background:#edf6f6;border-left:5px solid #248a94;padding:16px}img{width:100%}a{color:#126f84}</style><main>'
    text+='<h1>CPU优化验证：新身份结果与性能归因</h1><p>2026-11-03，11:40:09启动，13:31:02完成，耗时1小时50分53秒。15/15模型完成24轮，共360轮、90,000优化器步；5个配对种子。</p>'
    text+='<div class="note">预定义双模型融合是本轮最明确的收益：新报告Exact 68.423%，比BCE+F1高5.427个百分点、比纯组合+0.5高3.204个百分点，五种子均改善。混合采样提高单模型排序质量，受约束Exact校准的额外收益很小。</div>'
    text+=pic('decision_comparison.png')
    text+=table(['方法','决策','Exact %','Macro-F1 %','Macro-AUPR %','尾类F1 %','稀缺组合Exact %'],[[NAMES[m],d]+[stat(m,d,k) for k in ['exact','macro_f1','macro_aupr','tail_f1','rare_combo_exact']] for m,ds in agg.items() for d in ds])
    text+='<p>以上均为同一4032身份新报告池的五种子均值±样本标准差。尾类为AMA/SinFM；稀缺组合为训练计数最低的全部36个并列组合。此前70.10%来自check开发诊断；本轮68.42%来自新报告身份。应使用本轮同池基线作比较。</p>'
    text+='<h2>按类融合：互补性在新身份上成立</h2><p>融合规则在训练前固定：FMB使用同种子纯组合模型的0.5决策，其余五类使用自然BCE的F1决策。相对BCE+F1，Exact差的条件配对95%区间为[4.875, 5.979]个百分点；相对纯组合+0.5为[1.307, 5.101]。这是新身份上的确认，仍条件于同一模拟分布。</p>'
    text+='<p>融合把FMB召回从85.54%提高到95.63%，误报从7.75%降至2.22%，AUPR从96.34%升至99.49%；其余五类决策与BCE完全一致，尾类F1保持97.053%。平均纠正5.957%的原错误样本，同时使0.531%的原正确样本出错，净增5.427个百分点Exact。收益是两模型表征和分类器互补的整体结果，不能据此认定只增加一个分类头便可复制。</p>'
    text+='<p>部署需要两个DFMS模型，合计约515,022参数，相对单模型257,511参数增加一倍；尚未测量实际部署时延。下一步优先将这条已确认的双模型规则作为参照，研究按类蒸馏或分阶段分类器训练，并继续使用新报告身份验证。</p>'
    text+='<h2>混合采样：单模型更均衡，排序改善比Exact证据更稳定</h2><p>相对纯组合采样，混合采样Macro-AUPR提高0.240个百分点，五个种子均为正，条件配对95%区间为[0.132, 0.347]。F1校准后Macro-F1提高0.438个百分点、尾类F1提高0.645个百分点；说明其作用涉及类别区分，不能只解释为阈值变化。</p>'
    text+='<p>固定0.5时，混合采样Exact从65.218%提高到66.190%，差值+0.972个百分点，但条件区间[-0.682, 2.626]跨零。相同Exact校准下差值+1.925个百分点，区间[0.101, 3.748]；这依赖具体决策条件，不能笼统宣称混合采样全面更优。</p>'
    text+=pic('per_label_tradeoffs.png')
    text+='<p>混合采样F1校准后，MFT F1从纯组合的88.357%恢复到90.165%，SinFM从94.113%到95.096%，ISRJ从88.496%到89.215%；FMB从96.715%降至95.394%，体现优化重点的再分配。尾类F1为96.742%，仍低于自然BCE的97.053%。固定0.5下稀缺组合Exact也从70.382%降至68.698%，需保留这一权衡。</p>'
    text+=table(['采样','每轮独立身份均值','范围','FMB曝光 %','AMA曝光 %','120轮不同抽样顺序'],[[NAMES[m],f"{v['mean_unique']:.1f}",str(v['unique_range']),f"{v['mean_label_fractions'][0]*100:.2f}",f"{v['mean_label_fractions'][1]*100:.2f}",v['distinct_draw_orders']] for m,v in a['coverage'].items()])
    text+='<p>混合采样每轮覆盖约5315身份，纯组合约3965；FMB曝光由约50.73%调整到67.45%。覆盖与共现分布同时改变，本轮不能分别量化两者各贡献多少；若要定位原因，需要同分布、不同重复覆盖策略的独立对照。</p>'
    text+=pic('training_coverage.png')
    text+='<h2>受约束Exact校准：本轮不是主要增益来源</h2><p>相对F1校准，新报告Exact平均增加：BCE +0.283、纯组合+0.094、混合+0.060个百分点，三者的五种子配对区间均跨零。混合的Exact 66.215%与固定0.5的66.190%几乎相同，不能把最高小数点当作稳定改进。AUPR保持不变，因为后处理未改变逐类排序。</p>'
    text+=pic('calibration_transfer.png')
    text+='<p>该搜索使用固定网格、双起点、最多4轮坐标更新，并相对F1限制召回下降3个百分点、误报上升2个百分点。约束与有限calibration身份、局部搜索均可能限制收益；本轮没有做独立消融，不能断言是哪项因素主导，也不能推广为所有Exact决策方法无效。</p>'
    text+='<p>15个模型的calibration约束全部满足。约束不自动保证新数据上成立：BCE种子2054在新报告中ISRJ召回下降3.662个百分点，超过拟合阶段3个百分点限制；check还出现另外3个类/种子违约。部署若有硬召回要求，需要独立验证及余量，不能把calibration经验约束当作新数据保证。</p>'
    text+='<h2>可信度审计与剩余限制</h2><p>87份冻结文件、两份数据文件哈希复核通过；4032新报告身份重新与54份旧数据文件核对无交集；calibration/check各字段与原始数据逐项一致。15个last检查点均为24轮、6000步，best检查点符合check Macro-AUPR选模。15套F1与Exact校准从保存的calibration预测重放一致；145组指标条件复算通过。所有报告阶段均在13:29:55模型与决策冻结标记之后开始。</p>'
    text+='<p>随机流修复有效：纯组合和混合分别120轮的抽样顺序与派生种子均不同。五个种子仍共享固定训练/calibration/check池，区间不代表新数据集、模拟器变化或真实设备的不确定性，也没有多重比较校正。数据使用v3物理公式Python PCG64移植、独立10–30 dB分量JNR；没有MATLAB逐点复现或真实测量验证。</p>'
    at_end=sum(v['best_epoch']==24 for v in a['selection'].values())
    text+=f'<p>有{at_end}/15个模型的最佳check AUPR出现在预算末轮，本轮结论限定24轮训练预算。当前报告池已经消费，后续不能继续用它挑蒸馏超参数或阈值。</p>'
    text+='<h2>下一步优先级</h2><ol><li>以冻结双模型按类融合为当前准确率参照，优先验证其向单模型的按类蒸馏；教师只为训练身份生成软目标，保留BCE硬标签对照，避免报告信息进入训练。</li><li>部署只允许单模型时，优先保留50/50混合方案；固定0.5便于控制误报，F1校准可作为已报告的另一决策条件。两者Exact接近，选择应依据预先确定的召回/误报要求。</li><li>暂不扩大Exact阈值搜索预算；先验证主干/分类器分阶段训练能否保留FMB收益并恢复其他类别。结构实验仍需同预算与容量对照。</li><li>最终方案冻结后，增加新的报告身份及不同功率/更重长尾分布验证，避免只适配均衡组合模拟评测。</li></ol>'
    text+='<p><a href="results.csv">逐种子CSV</a> · <a href="analysis.json">逐类与约束诊断</a> · <a href="completion_audit.json">完成审计</a> · <a href="report.html">原始汇总</a> · <a href="plan.html">冻结方案</a></p></main></html>'
    (out/'analysis.html').write_text(text)
    print(out/'analysis.html')


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--suite',required=True,type=Path);p.add_argument('--render',action='store_true')
    args=p.parse_args();(render if args.render else audit)(args.suite.resolve())
