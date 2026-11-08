"""Audit the frozen 2026-11-02 v3 comparison without fitting models.

Run with the training environment first; use --render with a matplotlib
environment to render the saved statistics. Neither mode changes decisions.
"""
import argparse
import base64
from datetime import datetime
from html import escape
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
METHODS = ['bce', 'label_sampler', 'combo_sampler', 'weighted_bce', 'logit_adjustment']
NAMES = ['BCE', '标签采样', '组合采样', 'Weighted BCE', 'Logit Adjustment']
LABELS = ['FMB', 'AMA', 'SawFM', 'SinFM', 'MFT', 'ISRJ']


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def audit(suite):
    import h5py
    from longtail.engine.checkpoint import load_checkpoint
    from longtail.io import fingerprint, sha256
    from longtail.metrics import evaluate_metrics, probabilities

    plan = read(suite / 'plan.json')
    out = Path(plan['report_output'])
    results = read(out / 'results.json')
    done = read(suite / 'pipeline_completed.json')
    barrier = datetime.fromisoformat(read(suite / 'all_training_completed.json')['finished'])
    assert done['status'] == 'completed' and done['jobs_completed'] == 25
    for entry in plan['source_manifest']:
        assert sha256(suite / 'source_snapshot' / entry['path']) == entry['sha256'], entry['path']
    split = read(Path(plan['jobs'][0]['output']) / 'split.json')
    counts = np.array(split['fit_combination_counts'])
    rare = np.flatnonzero(counts == counts.min()) + 1
    tail = np.argsort(plan['train_label_counts'], kind='stable')[:2]
    roles = split['roles']
    dev_roles = ['calibration', 'check', 'development_test']
    for i, role in enumerate(dev_roles):
        for other in dev_roles[i + 1:]:
            assert not set(roles[role]) & set(roles[other])
    manifest_path = Path(read(plan['jobs'][0]['config'])['data']['manifest'])
    for name, digest in split['source_sha256'].items():
        assert sha256(manifest_path.parent / f'{name}.h5') == digest
    with h5py.File(manifest_path.parent / 'train.h5') as f:
        train_ids = f['receiver_noise_seed'][:].ravel()
        train_labels = f['label_selected'][:].T
    with h5py.File(manifest_path.parent / 'development.h5') as f:
        dev_ids = f['receiver_noise_seed'][:].ravel()
        dev_labels = f['label_selected'][:].T
    assert len(np.unique(train_ids)) == 8000 and len(np.unique(dev_ids)) == 6048
    assert not np.intersect1d(train_ids, dev_ids).size
    np.testing.assert_array_equal(train_labels.sum(0), plan['train_label_counts'])
    references, histories, sampling = {}, {}, {m: [] for m in METHODS}
    detail = {role: {m: {} for m in METHODS} for role in ['check', 'development_test']}
    strata = {m: {} for m in METHODS}
    selection, job_audits, thresholds = {}, [], {}
    for job in plan['jobs']:
        path = Path(job['output'])
        method, seed = job['method'], job['seed']
        history = read(path / 'history.json')
        config = read(path / 'config.json')
        decision = read(path / 'decision.json')
        status = read(path / 'status.json')
        assert read(path / 'split.json') == split
        assert status['status'] == 'completed' and status['parameters'] == 257511
        assert len(history) == 24 and [h['epoch'] for h in history] == list(range(1, 25))
        assert all(h['samples_seen'] == 8000 and h['sampler_padding'] == 0 for h in history)
        assert all(h['selection_split'] == 'check' for h in history)
        lr = [h['lr'] for h in history]
        if references.get('lr') is None:
            references['lr'] = lr
        assert references['lr'] == lr
        assert config['train']['epochs'] == 24 and config['dataloader']['batch_size'] == 32
        assert config['train']['lr'] == 1e-4 and config['train']['weight_decay'] == 1e-4
        assert decision['fit_role'] == 'calibration' and decision['method'] == 'f1_exact'
        assert decision['split_sha256'] == split['sha256']
        last, _ = load_checkpoint(path / 'last.pt')
        assert last['epoch'] == 24
        steps = [int(s['step']) for s in last['optimizer']['state'].values()]
        assert steps and set(steps) == {6000}
        del last
        best, _ = load_checkpoint(path / 'best.pt')
        best_epoch = int(np.argmax([h['validation']['macro_aupr'] for h in history])) + 1
        assert best['epoch'] == best_epoch == status['best_epoch']
        assert best['decision'] == decision
        del best
        digest = sha256(path / 'best.pt')
        phases = {}
        for stage in ['train', 'report']:
            phase = read(suite / 'phases' / f'{job["id"]}_{stage}.json')
            assert phase['phase'] == 'completed' and phase['checkpoint_sha256'] == digest
            when = datetime.fromisoformat(phase['finished'])
            assert (when <= barrier) if stage == 'train' else (when >= barrier)
            phases[stage] = phase['finished']
        assert (path / 'evaluation/development_test/metrics.json').stat().st_mtime >= barrier.timestamp()
        histories[job['id']] = [{k: h[k] for k in ['epoch', 'loss', 'lr', 'train_seconds', 'epoch_seconds']} |
                               {'check_aupr': h['validation']['macro_aupr']} for h in history]
        sampling[method].extend([{'seed': seed, 'epoch': h['epoch'], **h['sampling']}
                                 for h in history if 'sampling' in h])
        selection[job['id']] = {'best_epoch': best_epoch, 'best_at_budget_end': best_epoch == 24,
                               'elapsed_seconds': status['elapsed_seconds']}
        thresholds[job['id']] = decision['thresholds']
        for role in ['check', 'development_test']:
            evaluation = read(path / 'evaluation' / role / 'metrics.json')
            assert evaluation['checkpoint_sha256'] == digest and evaluation['decision'] == decision
            assert evaluation['config_sha256'] == fingerprint(config)
            assert evaluation['split_sha256'] == split['sha256']
            with np.load(path / 'evaluation' / role / 'predictions.npz') as z:
                y, logits, indices, ids = [z[k] for k in ['targets', 'logits', 'indices', 'identities']]
            assert set(indices.tolist()) == set(roles[role])
            np.testing.assert_array_equal(ids, dev_ids[indices])
            np.testing.assert_array_equal(y, dev_labels[indices])
            assert np.isfinite(logits).all()
            if role not in references:
                references[role] = (y, indices, ids)
            for observed, reference in zip([y, indices, ids], references[role]):
                np.testing.assert_array_equal(observed, reference)
            codes = y.astype(int) @ (1 << np.arange(6))
            assert np.all(np.bincount(codes, minlength=64)[1:] == (16 if role == 'check' else 64))
            probability = probabilities(logits)
            for condition, metric in evaluation['metrics'].items():
                check = evaluate_metrics(y, logits, LABELS, metric['thresholds'])
                for key in ['samples', 'subset_accuracy', 'macro_f1', 'macro_aupr', 'hamming_loss']:
                    assert abs(check[key] - metric[key]) < 1e-12, (job['id'], role, condition, key)
                row = next(r for r in results['rows'] if r['method'] == method and r['seed'] == seed
                           and r['role'] == role and r['decision'] == condition)
                for reported, recomputed in [('macro_f1', 'macro_f1'), ('macro_aupr', 'macro_aupr'), ('exact', 'subset_accuracy')]:
                    assert abs(row[reported] - check[recomputed]) < 1e-12
                pred = probability >= np.array(metric['thresholds'])
                correct = np.all(pred == y, axis=1)
                assert abs(row['rare_combo_exact'] - correct[np.isin(codes, rare)].mean()) < 1e-12
                assert abs(row['tail_f1'] - np.mean([check['per_label'][LABELS[c]]['f1'] for c in tail])) < 1e-12
                detail[role][method].setdefault(condition, []).append(check)
                if role == 'development_test':
                    fp, fn = pred & ~y.astype(bool), ~pred & y.astype(bool)
                    errors = fp | fn
                    entry = {'seed': seed, 'fp': fp.sum(0).tolist(), 'fn': fn.sum(0).tolist(),
                             'label_error_share': (errors.sum(0) / errors.sum()).tolist(),
                             'error_count_histogram': np.bincount(errors.sum(1), minlength=7).tolist(),
                             'fmb_absent_exact': float(correct[y[:, 0] == 0].mean()),
                             'fmb_present_exact': float(correct[y[:, 0] == 1].mean()),
                             'rare_exact': float(correct[np.isin(codes, rare)].mean()),
                             'nonrare_exact': float(correct[~np.isin(codes, rare)].mean())}
                    strata[method].setdefault(condition, []).append(entry)
        job_audits.append({'job': job['id'], 'best_epoch': best_epoch, 'steps': 6000,
                           'checkpoint_sha256': digest, 'phases_finished': phases})

    sampling_summary = {}
    for method in METHODS:
        records = sampling[method]
        if not records:
            sampling_summary[method] = {'replacement': False, 'mean_unique_per_epoch': 8000,
                'label_fractions': (train_labels.mean(0)).tolist(), 'mean_cardinality': float(train_labels.sum(1).mean())}
            continue
        for record in records:
            assert record['draws'] == 8000 and sum(record['combination_exposures']) == 8000
            combo = np.array(record['combination_exposures'])
            labels = (((np.arange(1, 64)[:, None] & (1 << np.arange(6))) > 0) * combo[:, None]).sum(0)
            np.testing.assert_array_equal(labels, record['label_exposures'])
        by_pair = {(r['seed'], r['epoch']): r for r in records}
        shifted_matches = sum(by_pair[(s, e + 1)]['combination_exposures'] == by_pair[(s + 1, e)]['combination_exposures']
                              for s in range(2050, 2054) for e in range(1, 24))
        sampling_summary[method] = {'replacement': True,
            'audited_epochs': len(records), 'mean_unique_per_epoch': float(np.mean([r['unique_fit_identities'] for r in records])),
            'unique_per_epoch_range': [min(r['unique_fit_identities'] for r in records), max(r['unique_fit_identities'] for r in records)],
            'label_fractions': (np.mean([r['label_exposures'] for r in records], axis=0) / 8000).tolist(),
            'mean_cardinality': float(np.mean([r['mean_cardinality'] for r in records])),
            'distinct_epoch_combination_histograms': len(set(tuple(r['combination_exposures']) for r in records)),
            'adjacent_seed_shifted_matches': shifted_matches, 'adjacent_seed_shifted_pairs': 92}
    per_label = {role: {m: {d: {label: {key: float(np.mean([r['per_label'][label][key] for r in records]))
                           for key in ['f1', 'aupr', 'precision', 'recall', 'negative_fpr']}
                           for label in LABELS} for d, records in ds.items()}
                        for m, ds in methods.items()} for role, methods in detail.items()}
    audit_result = {'status': 'passed', 'jobs': 25, 'epochs': 600, 'total_optimizer_steps': 150000,
                    'source_snapshot_files_hash_verified': len(plan['source_manifest']),
                    'data_file_hashes_rechecked': split['source_sha256'],
                    'disjoint_identity_and_role_checks': True, 'metrics_recomputed_conditions': 100,
                    'report_after_all_training': True, 'training_completion_barrier': barrier.isoformat(),
                    'pipeline_completion': done, 'job_checks': job_audits}
    analysis = {'suite': str(suite), 'report': str(out), 'labels': LABELS, 'tail_labels': [LABELS[c] for c in tail],
                'rare_combo_codes': rare.tolist(), 'rare_combo_fit_count': int(counts.min()),
                'rare_combo_report_samples': int(len(rare) * 64), 'aggregate': results['aggregate'],
                'effects': results['effects'], 'per_label': per_label, 'error_strata': strata,
                'sampling': sampling_summary, 'selection': selection, 'histories': histories,
                'thresholds': thresholds, 'completion_audit': audit_result,
                'weighted_bce_expected_class_weight_mass': (2 * (1 - np.array(plan['train_label_counts']) / 8000)).tolist(),
                'limitations': [
                    'Python PCG64 physical-equation port; not a verified bitwise MATLAB reproduction.',
                    'One fresh fixed simulation pool; no real measured data or new-domain validation.',
                    'Five initializations but resampling uses seed+epoch: adjacent seeds reuse draw streams at shifted epochs.',
                    'Paired seed t intervals are descriptive and conditional on this pool; no multiple-comparison adjustment.',
                    'Report identities have now been consumed; do not tune thresholds or hyperparameters on them.',
                    'Per-label F1 calibration optimizes a different objective from whole-vector Exact.',
                    'Rare combinations use all 36 minimum-count ties, not the legacy metric arbitrary bottom 13.',
                    'Equal optimizer steps and draws do not imply equal unique-identity exposure under replacement.'
                ]}
    write(out / 'completion_audit.json', audit_result)
    write(out / 'analysis.json', analysis)
    print(json.dumps({'audit': 'passed', 'jobs': 25, 'conditions_recomputed': 100,
                       'output': str(out / 'analysis.json'), 'sampling': sampling_summary}, ensure_ascii=False))


def render(suite):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font_manager.fontManager.addfont('/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf')
    plt.rcParams.update({'font.family': ['DejaVu Sans', 'Droid Sans Fallback'], 'axes.unicode_minus': False,
                         'axes.spines.top': False, 'axes.spines.right': False, 'font.size': 10})
    out = Path(read(suite / 'plan.json')['report_output'])
    a = read(out / 'analysis.json')
    agg = a['aggregate']['development_test']
    colors = ['#62768a', '#6e9a7e', '#198491', '#bf734c', '#906ca0']
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.1), layout='constrained')
    x = np.arange(5)
    for ax, metric, title in zip(axes, ['macro_f1', 'exact'], ['Macro-F1 (%)', '完整六标签正确率 Exact (%)']):
        for d, offset, color, label in [('fixed_0.5', -.18, '#64798f', '固定阈值 0.5'), ('calibrated', .18, '#1b9296', '逐标签 F1 校准')]:
            ax.bar(x + offset, [agg[m][d][metric]['mean'] * 100 for m in METHODS], width=.34, color=color, label=label,
                   yerr=[agg[m][d][metric]['std'] * 100 for m in METHODS], capsize=3)
        ax.set_xticks(x, ['BCE', '标签采样', '组合采样', '加权BCE', 'LA'])
        ax.set_title(title)
        ax.set_ylim((88, 94) if metric == 'macro_f1' else (50, 70))
        ax.legend(fontsize=8)
    fig.suptitle('同一新身份报告池，五种子均值 ± 样本标准差；每种方法都报告两种决策')
    fig.savefig(out / 'decision_comparison.png', dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4), layout='constrained')
    for ax, metric, title in zip(axes, ['macro_f1', 'exact', 'rare_combo_exact'], ['Macro-F1差', 'Exact差', '稀缺组合Exact差']):
        for i, method in enumerate(METHODS[1:]):
            e = a['effects'][method][metric]['training_effect_after_equal_calibration']
            low, high = e['paired_seed_95_t_interval']
            ax.errorbar(e['mean'] * 100, i, xerr=np.array([[e['mean'] - low], [high - e['mean']]]) * 100,
                        fmt='o', color=colors[i + 1], capsize=4)
        ax.axvline(0, color='#9aa7b2', ls='--')
        ax.set_yticks(range(4), NAMES[1:]); ax.invert_yaxis()
        ax.set_title(title); ax.set_xlabel('相对 BCE 的百分点差；双方均校准')
    fig.suptitle('五种子配对 t 区间（描述性95%；固定数据池；采样流跨种子存在重叠）')
    fig.savefig(out / 'paired_effects.png', dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout='constrained')
    for ax, metric, title in zip(axes, ['f1', 'aupr'], ['逐类 F1 差（双方均校准）', '逐类 AUPR 差（阈值无关）']):
        values = np.array([[100 * (a['per_label']['development_test'][m]['calibrated'][c][metric] -
                                   a['per_label']['development_test']['bce']['calibrated'][c][metric]) for c in LABELS] for m in METHODS[1:]])
        image = ax.imshow(values, cmap='RdBu', vmin=-9, vmax=9, aspect='auto')
        ax.set_xticks(range(6), LABELS); ax.set_yticks(range(4), NAMES[1:]); ax.set_title(title)
        for i in range(4):
            for j in range(6):
                ax.text(j, i, f'{values[i, j]:+.2f}', ha='center', va='center', color='white' if abs(values[i, j]) > 5 else '#123')
    fig.colorbar(image, ax=axes, label='相对BCE的百分点差', shrink=.85)
    fig.savefig(out / 'per_label_tradeoffs.png', dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4), layout='constrained')
    for m, name, color in zip(METHODS, NAMES, colors):
        curves = np.array([[h['check_aupr'] for h in a['histories'][f'{m}_seed{s}']] for s in range(2050, 2055)]) * 100
        avg, std = curves.mean(0), curves.std(0, ddof=1)
        ax.plot(range(1, 25), avg, label=name, color=color)
        ax.fill_between(range(1, 25), avg - std, avg + std, color=color, alpha=.13)
    ax.set(xlabel='训练轮数', ylabel='check Macro-AUPR (%)', ylim=(90, 99), title='选模集合上的学习曲线；阴影为五种子标准差')
    ax.legend(ncol=3, fontsize=9)
    fig.savefig(out / 'selection_curves.png', dpi=170)
    plt.close(fig)

    def table(headers, rows):
        return '<div class="scroll"><table><thead><tr>' + ''.join('<th>' + escape(str(h)) + '</th>' for h in headers) + '</tr></thead><tbody>' + ''.join('<tr>' + ''.join('<td>' + escape(str(c)) + '</td>' for c in row) + '</tr>' for row in rows) + '</tbody></table></div>'

    def pic(name):
        return '<img alt="实验结果统计图" src="data:image/png;base64,' + base64.b64encode((out / name).read_bytes()).decode() + '">'

    def stat(method, decision, metric):
        v = agg[method][decision][metric]
        return f"{100 * v['mean']:.3f} ± {100 * v['std']:.3f}"

    text = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>新v3长尾对照：完成审计与归因分析</title><style>
body{font:16px/1.8 system-ui,sans-serif;color:#234;background:#f0f4f6;margin:0}main{max-width:1180px;margin:28px auto;padding:32px;background:white}h1,h2{color:#155e75}h2{margin-top:32px}.note{background:#edf6f6;border-left:5px solid #248a94;padding:16px}.scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:14px;margin:18px 0}td,th{border:1px solid #d6e0e6;padding:8px;text-align:left}th{background:#155e75;color:white}img{width:100%;margin:12px 0}a{color:#126f84}code{overflow-wrap:anywhere}</style><main>'''
    text += '<h1>新v3长尾基线：结果核对与性能归因</h1><p>2026-11-03复核；5种训练方法 × 5个初始化种子 × 2种冻结决策；报告数据已消费。</p>'
    text += '<div class="note">25/25作业全部完成，600/600轮、150,000次优化器更新。北京时间11月2日21:39:52启动，11月3日00:55:41完成，耗时3小时15分49秒。8作业并发、每作业4个物理CPU核。检查点、配置、划分、源码哈希和100组测评条件复算全部通过。</div>'
    text += '<h2>首先回答：收益来自哪里</h2><p>BCE逐标签F1校准是本轮最稳定的低成本改进：Macro-F1增加0.925个百分点，Exact增加5.675个百分点，五种子均提高。LA固定0.5时看似有效，但各方法都校准后只剩0.004个百分点F1差、0.124个百分点Exact差，AUPR几乎相同，主要表现为决策偏移。</p>'
    text += '<p>组合采样在固定0.5时取得最高Exact（65.258%），并有小幅平均排序收益；相对BCE+校准的63.373%，增加1.885个百分点，五种子均为正，但尾类F1下降1.643个百分点。其优势在同样F1校准后缩小，整体F1和Exact差的五种子区间跨零。稀缺组合Exact仍提高3.958个百分点，但尾类F1下降1.234个百分点（五种子全部下降），不能宣称全面改善长尾。标签采样未显示稳定额外收益；当前weighted BCE方案整体退化。</p>'
    text += pic('decision_comparison.png')
    text += table(['方法', '阈值', 'Macro-F1 %', 'Macro-AUPR %', 'Exact %', '尾类F1 %', '稀缺组合Exact %'],
                  [[n, '0.5' if d == 'fixed_0.5' else '逐类F1校准'] + [stat(m, d, k) for k in ['macro_f1', 'macro_aupr', 'exact', 'tail_f1', 'rare_combo_exact']]
                   for m, n in zip(METHODS, NAMES) for d in ['fixed_0.5', 'calibrated']])
    text += '<p>数值为五种子均值±样本标准差。Exact表示六个标签全部判断正确，不能用逐标签F1代替。尾类预定义为训练阳性计数最低的AMA和SinFM；稀缺组合为全部36个最小计数组合（每组合训练41条），报告集共2,304条，保留全部并列项。</p>'
    text += '<h2>公平归因：双方使用相同校准规则</h2>' + pic('paired_effects.png')
    rows = []
    for m, n in zip(METHODS[1:], NAMES[1:]):
        for k, label in [('macro_f1', 'Macro-F1'), ('exact', 'Exact'), ('rare_combo_exact', '稀缺组合Exact')]:
            e = a['effects'][m][k]['training_effect_after_equal_calibration']
            rows.append([n, label, f"{100 * e['mean']:+.3f}", ' ～ '.join(f'{100 * v:+.3f}' for v in e['paired_seed_95_t_interval']), f"{sum(v > 0 for v in e['per_seed'])}/5"])
    text += table(['方法', '指标', '相对BCE差 pp', '描述性配对95%区间 pp', '正向种子'], rows)
    text += '<p>固定0.5下的改善同时含表征、先验和决策点变化；统一校准后的差异更适合排查额外训练收益，但仍不是对单一机制的完整因果分离。AUPR不受阈值变化影响。区间只描述本数据池下五个训练初始化的差异，没有多重比较校正，也不能代表新数据池或真实设备的不确定性。</p>'
    text += '<h2>性能阻碍：逐类误报、排序和曝光</h2>' + pic('per_label_tradeoffs.png')
    text += '<p>BCE固定0.5时FMB误报率31.13%，校准后降到7.53%；F1从84.26%升到88.49%，同时召回由94.66%降到85.16%。整体正确率受头类误报影响显著。最稀缺AMA反而容易：BCE校准F1为98.49%，AUPR为99.81%，因此“样本少”并不等于“当前最难”。</p>'
    text += '<p>组合采样把FMB校准F1升至96.96%、AUPR升至99.53%，说明这一类存在真实排序改善；但SinFM校准F1从95.56%降至93.49%，MFT从91.82%降至88.31%，ISRJ从89.70%降至88.36%。该方法改变了共现分布与优化重点，收益存在明显转移。</p>'
    text += '<p>weighted BCE使用正例权重(N−n_pos)/n_pos，FMB仅0.189、AMA为2.567。FMB校准F1降至80.18%，AUPR降至91.47%（BCE为96.18%），因此退化不能只靠换阈值恢复。这一实现同时改变类别间损失质量：FMB平均加权质量约0.318，AMA约1.439；它不是只提高尾类而不影响头类的中性操作。本轮只能否定这组具体权重与预算，不能否定所有加权损失。</p>'
    text += table(['方法', '每轮独立身份', 'FMB曝光 %', 'AMA曝光 %', '平均标签数/条'],
                  [[n, f"{a['sampling'][m]['mean_unique_per_epoch']:.1f}", f"{100*a['sampling'][m]['label_fractions'][0]:.2f}",
                    f"{100*a['sampling'][m]['label_fractions'][1]:.2f}", f"{a['sampling'][m]['mean_cardinality']:.3f}"] for m, n in zip(METHODS, NAMES)])
    text += '<p>标签采样受共现约束，FMB曝光仍为84.17%，AMA只升到38.76%，并使每条标签数从3.112升至3.548；它未实现六类均衡。组合采样六类曝光约50.7%–51.0%，更贴近本报告池的均衡组合分布，但每轮独立身份仅约3,966个。相同8,000次抽样、250步不等于相同独立样本覆盖；这些均为已记录的实际曝光，不是训练失败。</p>'
    text += '<h2>为什么F1校准会降低完整识别正确率</h2><p>组合采样校准后Macro-F1仅增加0.042个百分点，Exact却从65.258%降至63.800%，五种子中四个下降；平均逐类误报从3.239%升到6.384%。校准阈值大多低于0.5，提高正例召回也增加负例误报。逐类F1目标不会直接最大化六标签同时正确，且calibration仅1,008身份，阈值有估计波动。观察支持目标不一致和阈值稳定性问题；单凭本轮不能证明哪一因素占主导。LA和weighted BCE同样出现校准后Exact下降，不能将F1校准当作所有模型必然获益的步骤。</p>'
    text += '<h2>审计发现与可信度边界</h2><p>80份冻结源码和配置文件校验通过，train/development数据哈希重新核验；8,000训练身份与6,048开发身份无交集。calibration/check/report分别为1,008/1,008/4,032，逐组合16/16/64，角色身份互斥。全部训练按check Macro-AUPR选模，阈值来自calibration。25个报告测评均在00:53:32全部训练完成标记后执行，权重与阈值和check测评一致。没有发现未完成作业或非有限输出。</p>'
    text += '<p><strong>新发现：采样随机流跨种子复用。</strong>有放回采样使用seed+epoch，因此相邻种子的相邻轮抽样流重复。标签采样和组合采样各120个训练轮只出现28种组合曝光记录，92/92对相邻偏移轮完全一致。模型初始化和训练噪声仍不同，均值比较可保留；五种子并非五套完全独立抽样流，当前配对区间不宜作为严格独立种子的显著性证据。下一轮应通过(seed, epoch)的独立派生修正后再确认关键对比；本次没有修改冻结源码或重跑挑结果。</p>'
    text += '<p>本机没有可用MATLAB/Octave，数据使用原v3物理公式与逐分量独立10–30 dB JNR的Python PCG64移植；没有完成MATLAB跨语言分布一致性验证，不声称严格复现原随机流。训练标签IF约3、组合最大/最小约7.85，属于有限程度的不均衡；尚不能推断更重长尾、不同功率分布和实测场景效果。本池已完成一次预定报告，之后不能继续用它调阈值或挑超参数。</p>'
    text += '<h2>下一步按核心指标推进</h2><ol><li><strong>先明确整体正确率是否以Exact为主。</strong>若是，下一轮预先设定check选模和calibration阈值目标，比较逐标签F1与小范围、带约束的Exact决策；报告两者及逐类召回/误报，避免只报F1改善。</li><li><strong>保留BCE+校准为低成本参考，组合采样+0.5为整体Exact候选。</strong>候选来自本轮结果，尚需修复采样流派生并用新报告身份确认。优先研究自然采样与组合均衡采样的温和混合，限制重复曝光，同时监测SinFM/MFT/ISRJ退化；不直接叠加损失与采样。</li><li><strong>暂缓推广当前weighted BCE与LA。</strong>前者有排序退化，后者公平校准后基本持平；下一笔训练预算优先用于验证组合共现、阈值目标和覆盖率，而不是增加结构复杂度。</li><li><strong>架构实验后移。</strong>基线和决策目标确认后，再将已有查询头与该基线按同数据、同预算比较，并增加重长尾和功率分档验证。本轮不能证明Query2Label已获益。</li></ol>'
    text += pic('selection_curves.png')
    at_end = sum(v['best_at_budget_end'] for v in a['selection'].values())
    text += f'<p>25次训练有{at_end}次最佳check AUPR出现在第24轮；当前结论限定24轮预算，不能断言所有方法已经充分收敛。不同损失的数值不宜直接互比，学习进展使用相同check AUPR展示。</p>'
    text += '<p><a href="results.csv">逐种子结果CSV</a> · <a href="results.json">原始归因统计</a> · <a href="analysis.json">逐类、错误分层及采样诊断</a> · <a href="completion_audit.json">完成审计</a> · <a href="plan.html">冻结方案</a></p></main></html>'
    (out / 'analysis.html').write_text(text)
    print(out / 'analysis.html')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite', required=True, type=Path)
    p.add_argument('--render', action='store_true')
    args = p.parse_args()
    (render if args.render else audit)(args.suite.resolve())


if __name__ == '__main__':
    main()
