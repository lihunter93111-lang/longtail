"""Post-completion independent audit and error attribution; no model changes."""
import argparse
from datetime import datetime
from html import escape
import json
from pathlib import Path

import numpy as np

STYLE = '<style>body{font:16px/1.8 system-ui;max-width:1280px;margin:24px auto;padding:24px;color:#234}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:7px;border:1px solid #ccd}th{background:#e9f3f4}</style>'


def table(headers, rows):
    return '<table><tr>'+''.join('<th>'+escape(str(v))+'</th>' for v in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+escape(str(v))+'</td>' for v in row)+'</tr>' for row in rows)+'</table>'


def review(suite):
    from sklearn.metrics import average_precision_score, roc_auc_score
    from longtail.io import read_json, write_json, sha256
    from longtail.metrics import probabilities
    from longtail.segment_statistics import report_gate, transform
    done = read_json(suite/'pipeline_completed.json')
    assert done['status'] == 'completed'
    plan, summary = read_json(suite/'plan.json'), read_json(suite/'summary.json')
    report_gate(suite)
    for path, digest in plan['data_hashes'].items():
        assert sha256(path) == digest, path
    for ref in plan['references'].values():
        for source in [ref['new'][k] for k in ['B', 'linear', 'local']]+[ref['FMB'], ref['target_head']]:
            assert sha256(source['path']) == source['sha256']
    for rec in plan['source_manifest']:
        assert sha256(suite/'source_snapshot'/rec['path']) == rec['sha256']
        assert sha256(Path(plan['root'])/rec['path']) == rec['sha256']
    transforms, fit_replays, details, shift_checks = [], [], [], []
    names = list(summary['report_means'])
    for seed in plan['seeds']:
        cache = suite/'cache'/f'seed{seed}'
        with np.load(cache/'fit.npz') as fit:
            old_file = Path(plan['previous_suite'])/'cache/targeted'/f'seed{seed}'/'new_fit.npz'
            with np.load(old_file) as old:
                np.testing.assert_array_equal(fit['identities'], old['identities'])
                np.testing.assert_array_equal(fit['targets'], old['targets'])
                np.testing.assert_allclose(fit['features'], old['features'], rtol=1e-5, atol=2e-6)
                np.testing.assert_allclose(fit['natural_logits'][:, 2:], old['candidate_logits'][:, 2:], rtol=1e-5, atol=2e-6)
                fit_replays.append({'seed': seed, 'same_natural_fit_identities_targets_features': True,
                                    'maximum_feature_abs_difference': float(np.max(np.abs(fit['features']-old['features'])))})
            import hashlib
            fit_hash = hashlib.sha256(np.asarray(fit['identities'], '<i8').tobytes()).hexdigest()
            for path in sorted((suite/'models'/f'seed{seed}').glob('*_transform.npz')):
                method = path.stem.removesuffix('_transform')
                state = dict(np.load(path))
                assert str(state['fit_id_sha256']) == fit_hash
                x = fit[method]
                np.testing.assert_array_equal(state['mean'], x.mean(0, dtype=np.float64).astype(np.float32))
                np.testing.assert_array_equal(state['scale'], np.maximum(x.std(0, dtype=np.float64), 1e-5).astype(np.float32))
                ortho = state['components'] @ state['components'].T
                np.testing.assert_allclose(ortho, np.eye(128), atol=1e-5)
                assert np.isfinite(transform(x[:256], state)).all()
                transforms.append({'seed': seed, 'method': method,
                    'input_dimensions': int(state['input_dimensions']),
                    'retained_variance': float(state['explained_variance_fraction']),
                    'PCA_coefficients': int(state['components'].size),
                    'natural_fit_only_binding_and_moments_replayed': True})
        output = suite/'evaluation'/f'seed{seed}'
        th = np.asarray(plan['references'][str(seed)]['new']['thresholds'])
        for role in ['development_test', 'ood_wide', 'pair_report']:
            with np.load(output/('predictions_'+role+'.npz')) as records:
                y = records['targets'].astype(bool)
                gap = -records['relative_power_db']
                anchor = probabilities(records['logits_anchor_pair1']) >= th
                correct_anchor = (anchor == y).all(1)
                for name in names:
                    z = records['logits_'+name]
                    p = probabilities(z)
                    pred = p >= th
                    correct = (pred == y).all(1)
                    bins = []
                    for lo, hi in [(0, 6), (6, 10), (10, 15), (15, 20), (20, 30)]:
                        mask = y[:, [2, 4]] & (gap[:, [2, 4]] >= lo-1e-6) & (gap[:, [2, 4]] < hi-1e-6)
                        bins.append({'lower_db': lo, 'upper_db': hi, 'support': int(mask.sum()),
                                     'recall': float(pred[:, [2, 4]][mask].mean()) if mask.any() else None})
                    labels = {}
                    for c, label in [(2, 'SawFM'), (4, 'MFT')]:
                        positive = y[:, c] & (gap[:, c] >= 15-1e-6) & (gap[:, c] < 20-1e-6)
                        selected = positive | ~y[:, c]
                        labels[label] = {'15_20_positive_count': int(positive.sum()),
                            '15_20_recall': float(pred[positive, c].mean()),
                            '15_20_aupr': float(average_precision_score(positive[selected], p[selected, c])),
                            '15_20_auroc': float(roc_auc_score(positive[selected], p[selected, c])),
                            'false_positives': int((pred[:, c] & ~y[:, c]).sum()),
                            'false_negatives': int((~pred[:, c] & y[:, c]).sum()),
                            'weak_false_negatives': int((~pred[:, c] & records['weak_mask'][:, c].astype(bool)).sum())}
                    details.append({'seed': seed, 'role': role, 'method': name,
                        'exact_fixes_vs_pair1': int((correct & ~correct_anchor).sum()),
                        'exact_regressions_vs_pair1': int((~correct & correct_anchor).sum()),
                        'focus_gap_bins': bins, 'per_label': labels})
        with np.load(output/'predictions_shift0.npz') as base:
            ids = base['identities'].copy()
            p0 = {m: probabilities(base['logits_'+m]) >= th for m in names}
            y = base['targets'].astype(bool)
        for shift in [1, 3, 7]:
            with np.load(output/(f'predictions_shift{shift}.npz')) as shifted:
                np.testing.assert_array_equal(ids, shifted['identities'])
                for name in names:
                    pp = probabilities(shifted['logits_'+name]) >= th
                    shift_checks.append({'seed': seed, 'shift': shift, 'method': name,
                        'target_prediction_flip_rate': float((pp[:, [2, 4]] != p0[name][:, [2, 4]]).mean()),
                        'exact_delta_vs_unshifted': float((pp == y).all(1).mean()-(p0[name] == y).all(1).mean())})
    means = {}
    for role in ['development_test', 'ood_wide', 'pair_report']:
        means[role] = {}
        for name in names:
            rows = [v for v in details if v['role'] == role and v['method'] == name]
            means[role][name] = {
                'exact_fixes_mean': float(np.mean([v['exact_fixes_vs_pair1'] for v in rows])),
                'exact_regressions_mean': float(np.mean([v['exact_regressions_vs_pair1'] for v in rows])),
                'focus_gap_bins': [{'lower_db': b['lower_db'], 'upper_db': b['upper_db'], 'support': b['support'],
                    'recall': float(np.mean([r['focus_gap_bins'][i]['recall'] for r in rows])) if b['support'] else None}
                    for i, b in enumerate(rows[0]['focus_gap_bins'])],
                'per_label': {label: {key: float(np.mean([r['per_label'][label][key] for r in rows]))
                                      for key in rows[0]['per_label'][label]} for label in ['SawFM', 'MFT']}}
    result = {'completed': done, 'summary': summary, 'PCA_audit': transforms, 'fit_replays': fit_replays,
              'details': details, 'means': means, 'shift_audit': shift_checks,
              'audit': {'all_original_source_snapshot_data_and_reference_hashes_verified': True,
                        'all_PCA_fit_identity_bindings_moments_replayed': True,
                        'all_current_fit_features_match_prior_frozen_features': True,
                        'report_frozen_gate_reverified': True,
                        'model_threshold_selection_changes': False}}
    report = Path(plan['report_output'])
    write_json(report/'review.json', result)
    html = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>分段统计完成复核</title>'+STYLE+'<h1>冻结主干分段统计：完成复核</h1><p>主阈值固定；只改SawFM/MFT。训练、选型和独立报告完成后复核，不修改权重、阈值或选择。</p><p><a href="analysis.html">自动报告</a> · <a href="review.json">完整复核JSON</a> · <a href="plan.html">预先冻结方案</a></p>'
    html += '<p>开始：'+escape(done['started'])+'；完成：'+escape(done['finished'])+'；check预选：'+escape(summary['selection']['selected'])+'。</p>'
    html += table(['方法', 'Exact%', 'Saw弱召回%', 'MFT弱召回%', '两目标弱AP%', '两目标FPR%'],
        [[m]+[f'{100*v[k]:.3f}' for k in ['exact', 'saw_weak_recall', 'mft_weak_recall', 'focus_weak_aupr', 'focus_fpr']] for m, v in summary['report_means'].items()])
    for name, comp in summary['comparisons'].items():
        d = comp['deltas']['exact']
        html += '<p>'+escape(name)+f'：Exact {100*d["mean"]:+.3f}pp，条件95%区间[{100*d["paired95"][0]:+.3f},{100*d["paired95"][1]:+.3f}]；正式门槛'+('通过' if comp['passed'] else '未全部通过')+'。</p>'
    html += '<h2>压缩与控制检查</h2>'+table(['种子', '特征', '输入维数', '保留方差%', '冻结PCA系数'], [[v['seed'], v['method'], v['input_dimensions'], f'{100*v["retained_variance"]:.3f}', v['PCA_coefficients']] for v in transforms])
    html += '<h2>主报告15～20dB边界</h2>'+table(['方法', '标签', '召回%', 'AP%', 'AUROC%', 'FP数均值', 'FN数均值'], [[m, c]+[f'{100*v[k]:.3f}' for k in ['15_20_recall', '15_20_aupr', '15_20_auroc']]+[f'{v[k]:.1f}' for k in ['false_positives', 'false_negatives']] for m, labels in means['development_test'].items() for c, v in labels['per_label'].items()])
    html += '<p>PCA只用自然训练身份拟合，均值/标准差及身份SHA独立重放；新提取的训练特征与历史冻结特征一致。其他四类输出不变、批次抽样相同和报告冻结顺序均有完成审计。PCA丢弃低方差弱证据的可能性尚未排除；阴性结果不能否定全部时序方法。固定池五种子条件区间及同一仿真器仍限制结论。</p></html>'
    (report/'review.html').write_text(html)
    print(json.dumps({'review': str(report/'review.html'), 'selected': summary['selection']['selected'], 'audit': result['audit']}, ensure_ascii=False))


def plot(suite):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plan = json.loads((suite/'plan.json').read_text())
    report = Path(plan['report_output'])
    data = json.loads((report/'review.json').read_text())
    summary = data['summary']
    methods = summary['selection']['methods']
    labels = {'global128': 'Global128 control', 'global256': 'Global256', 'segments124': 'Segments1/2/4',
              'anchor_pair1': 'Frozen6.25% reference', 'frozen_natural': 'Frozen natural'}
    colors = ['#6489aa', '#d08a47', '#239a8b']
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3), constrained_layout=True)
    for i, method in enumerate(methods):
        comp = summary['comparisons'][method+'_vs_pair1']
        d = comp['deltas']['exact']
        low, high = np.asarray(d['paired95'])*100
        center = d['mean']*100
        axes[0].barh(i, center, color=colors[i], alpha=.85)
        axes[0].errorbar(center, i, xerr=np.array([[center-low], [high-center]]), fmt='none', color='#243848', capsize=4)
        dx, dy = comp['deltas']['focus_fpr']['mean']*100, comp['deltas']['focus_weak_aupr']['mean']*100
        axes[1].scatter(dx, dy, s=65, color=colors[i])
        axes[1].annotate(labels[method], (dx, dy), xytext=(4, 7), textcoords='offset points', fontsize=8)
    axes[0].set_yticks(range(len(methods)), [labels[m] for m in methods])
    axes[0].axvline(0, color='#999', lw=1)
    axes[0].axvline(.5, color='#aaa', lw=1, ls='--')
    axes[0].set_xlabel('Exact change vs reference (pp)')
    axes[0].set_title('Fixed thresholds; paired95% CI')
    axes[1].axhline(0, color='#aaa', lw=.8)
    axes[1].axvline(0, color='#aaa', lw=.8)
    axes[1].set_xlabel('Saw/MFT mean FPR change (pp)')
    axes[1].set_ylabel('Saw/MFT mean weak AP change (pp)')
    axes[1].set_title('Ranking and false-positive tradeoff')
    for method, color in [('anchor_pair1', '#555')]+list(zip(methods, colors)):
        bins = data['means']['development_test'][method]['focus_gap_bins'][:4]
        xs = [(v['lower_db']+v['upper_db'])/2 for v in bins]
        ys = [v['recall']*100 for v in bins]
        axes[2].plot(xs, ys, '-o', label=labels[method], color=color, markersize=4)
    axes[2].set_xlabel('Component power gap (dB; bin centers)')
    axes[2].set_ylabel('Saw/MFT positive recall (%)')
    axes[2].set_title('Original10--30dB report distribution')
    axes[2].legend(fontsize=7)
    for axis in axes:
        axis.spines[['top', 'right']].set_visible(False)
        axis.grid(alpha=.15)
    fig.savefig(report/'outcome_overview.png', dpi=180)
    plt.close(fig)
    html = (report/'review.html').read_text()
    if 'outcome_overview.png' not in html:
        html = html.replace('</html>', '<p><img src="outcome_overview.png" alt="Exact区间、弱排序和功率分层" style="max-width:100%"></p></html>')
        (report/'review.html').write_text(html)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite', type=Path, required=True)
    p.add_argument('--plot-only', action='store_true')
    a = p.parse_args()
    if a.plot_only:
        plot(a.suite.resolve())
    else:
        review(a.suite.resolve())
