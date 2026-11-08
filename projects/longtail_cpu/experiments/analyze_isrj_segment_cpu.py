"""Independent post-completion ISRJ audit and descriptive error attribution."""
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
    import hashlib
    import torch
    from sklearn.metrics import average_precision_score, roc_auc_score
    from longtail.io import read_json, write_json, sha256
    from longtail.metrics import probabilities
    from longtail.segment_statistics import report_gate, load_transform
    from longtail.isrj_segment import cache, predictions, metric, mean_metrics, KEYS
    from longtail.replication_temporal import contrast
    from longtail.staged_training import CLASSES
    done = read_json(suite/'pipeline_completed.json')
    assert done['status'] == 'completed' and done['audit']['status'] == 'passed'
    plan, summary = read_json(suite/'plan.json'), read_json(suite/'summary.json')
    report_gate(suite)
    for path, digest in {**plan['data_hashes'], **plan['immutable_sources']}.items():
        assert sha256(path) == digest, path
    for rec in plan['source_manifest']:
        assert sha256(suite/'source_snapshot'/rec['path']) == rec['sha256']
        assert sha256(Path(plan['root'])/rec['path']) == rec['sha256']
    transforms, details, shifts, curves, auxiliary_rows = [], [], [], [], []
    names = list(summary['report_means'])
    replay_count = 0
    for seed in plan['seeds']:
        fit_source = plan['fit_caches'][str(seed)]
        assert sha256(fit_source['path']) == fit_source['sha256']
        with np.load(fit_source['path']) as fit:
            digest = hashlib.sha256(np.asarray(fit['identities'], '<i8').tobytes()).hexdigest()
            for method in ['global128', 'segments124']:
                spec = plan['feature_transforms'][str(seed)][method]
                state = load_transform(spec['path'])
                assert str(state['fit_id_sha256']) == digest
                x = fit[method]
                np.testing.assert_array_equal(state['mean'], x.mean(0, dtype=np.float64).astype(np.float32))
                np.testing.assert_array_equal(state['scale'], np.maximum(x.std(0, dtype=np.float64), 1e-5).astype(np.float32))
                transforms.append({'seed': seed, 'method': method,
                    'retained_variance': float(state['explained_variance_fraction']),
                    'input_dimensions': int(state['input_dimensions']),
                    'frozen_PCA_coefficients': int(state['components'].size),
                    'fit_ids_and_moments_replayed': True})
        output = suite/'evaluation'/f'seed{seed}'
        decisions = read_json(output/'decisions.json')
        threshold = np.asarray(plan['references'][str(seed)]['new']['thresholds'])
        torch.set_num_threads(4)
        raw = cache(suite, seed, 'development_test')
        replay = predictions(suite, seed, summary['selection']['methods'], raw)
        with np.load(output/'predictions_development_test.npz') as stored:
            for name, value in replay.items():
                np.testing.assert_allclose(value, stored['logits_'+name], rtol=1e-5, atol=2e-6)
                np.testing.assert_array_equal(probabilities(value) >= threshold, probabilities(stored['logits_'+name]) >= threshold)
                original = read_json(output/'metrics_development_test.json')['metrics'][name]
                for rule, th in decisions['models'][name]['thresholds'].items():
                    if th is None:
                        continue
                    measured = metric(raw, value, th)
                    assert all(abs(measured[k]-original[rule][k]) < 1e-12 for k in KEYS)
                    replay_count += 1
        del raw, replay
        for role in ['development_test', 'ood_wide']:
            with np.load(output/('predictions_'+role+'.npz')) as rec:
                y, weak = rec['targets'].astype(bool), rec['weak_mask'].astype(bool)
                gap = -rec['relative_power_db']
                anchor = probabilities(rec['logits_reference']) >= threshold
                anchor_correct = (anchor == y).all(1)
                other_correct = (anchor[:, :5] == y[:, :5]).all(1)
                for name in names:
                    p = probabilities(rec['logits_'+name])
                    pred = p >= threshold
                    err = pred != y
                    exact = ~err.any(1)
                    np.testing.assert_array_equal(pred[:, :5], anchor[:, :5])
                    bins = []
                    for lo, hi in [(0, 6), (6, 10), (10, 15), (15, 20), (20, 30)]:
                        pos = y[:, 5] & (gap[:, 5] >= lo-1e-6) & (gap[:, 5] < hi-1e-6)
                        selected = pos | ~y[:, 5]
                        bins.append({'lower_db': lo, 'upper_db': hi, 'support': int(pos.sum()),
                            'recall': float(pred[pos, 5].mean()) if pos.any() else None,
                            'aupr': float(average_precision_score(pos[selected], p[selected, 5])) if pos.any() else None,
                            'auroc': float(roc_auc_score(pos[selected], p[selected, 5])) if pos.any() else None})
                    fixes = exact & ~anchor_correct
                    regressions = ~exact & anchor_correct
                    counts = {}
                    for c, label in enumerate(CLASSES):
                        counts[label] = {'fp': int((pred[:, c] & ~y[:, c]).sum()),
                            'fn': int((~pred[:, c] & y[:, c]).sum()),
                            'weak_fn': int((~pred[:, c] & weak[:, c]).sum()),
                            'only_this_label_wrong': int((err[:, c] & (err.sum(1) == 1)).sum())}
                    details.append({'seed': seed, 'role': role, 'method': name, 'samples': len(y),
                        'exact_fixes': int(fixes.sum()), 'exact_regressions': int(regressions.sum()),
                        'exact_fixes_positive_ISRJ': int((fixes & y[:, 5]).sum()),
                        'exact_fixes_negative_ISRJ': int((fixes & ~y[:, 5]).sum()),
                        'exact_regressions_positive_ISRJ': int((regressions & y[:, 5]).sum()),
                        'exact_regressions_negative_ISRJ': int((regressions & ~y[:, 5]).sum()),
                        'ISRJ_gap_bins': bins, 'counts': counts,
                        'ISRJ_only_oracle_Exact': float(other_correct.mean())})
        with np.load(output/'predictions_shift0.npz') as base:
            ids, y = base['identities'].copy(), base['targets'].astype(bool)
            unshifted = {name: probabilities(base['logits_'+name]) >= threshold for name in names}
        for shift in [1, 3, 7]:
            with np.load(output/(f'predictions_shift{shift}.npz')) as rec:
                np.testing.assert_array_equal(ids, rec['identities'])
                for name in names:
                    pp = probabilities(rec['logits_'+name]) >= threshold
                    shifts.append({'seed': seed, 'method': name, 'shift_samples': shift,
                        'isrj_prediction_flip_rate': float((pp[:, 5] != unshifted[name][:, 5]).mean()),
                        'exact_delta_vs_unshifted': float((pp == y).all(1).mean()-(unshifted[name] == y).all(1).mean())})
        for method in summary['selection']['methods']:
            folder = suite/'models'/f'seed{seed}'
            status = read_json(folder/(method+'_status.json'))
            history = read_json(folder/('history_'+method+'.json'))
            curves.append({'seed': seed, 'method': method, 'best_epoch': status['best_epoch'],
                'check_Exact_epoch18_to24': history[-1]['check_exact']-history[17]['check_exact'],
                'check_weakAP_epoch18_to24': history[-1]['check_isrj_weak_ap']-history[17]['check_isrj_weak_ap']})
        main = read_json(output/'metrics_development_test.json')['metrics']
        auxiliary_rows.append(main)
    means = {}
    for role in ['development_test', 'ood_wide']:
        means[role] = {}
        for name in names:
            rows = [r for r in details if r['role'] == role and r['method'] == name]
            keys = ['exact_fixes', 'exact_regressions', 'exact_fixes_positive_ISRJ', 'exact_fixes_negative_ISRJ',
                    'exact_regressions_positive_ISRJ', 'exact_regressions_negative_ISRJ', 'ISRJ_only_oracle_Exact']
            m = {key: float(np.mean([r[key] for r in rows])) for key in keys}
            m['counts'] = {c: {key: float(np.mean([r['counts'][c][key] for r in rows])) for key in rows[0]['counts'][c]} for c in CLASSES}
            m['ISRJ_gap_bins'] = [{**{key: b[key] for key in ['lower_db', 'upper_db', 'support']},
                **{key: float(np.mean([r['ISRJ_gap_bins'][i][key] for r in rows])) if b['support'] else None for key in ['recall', 'aupr', 'auroc']}}
                for i, b in enumerate(rows[0]['ISRJ_gap_bins'])]
            means[role][name] = m
    auxiliary = {}
    for rule in ['isrj_f1', 'matched_calibration_fpr']:
        auxiliary[rule] = {'means': {name: mean_metrics([r[name][rule] for r in auxiliary_rows]) for name in names},
            'segment_minus_reference': {key: contrast([r['segments124'][rule][key] for r in auxiliary_rows],
                                                      [r['reference'][rule][key] for r in auxiliary_rows]) for key in KEYS}}
    elapsed = (datetime.fromisoformat(done['finished'])-datetime.fromisoformat(done['started'])).total_seconds()
    result = {'completed': done, 'elapsed_seconds': elapsed, 'summary': summary, 'transforms': transforms,
        'details': details, 'means': means, 'shift_audit': shifts, 'training_curves': curves, 'auxiliary': auxiliary,
        'audit': {'original_sources_snapshot_weights_transforms_data_SHA_verified': True,
                  'PCA_natural_fit_identity_and_moments_replayed': True,
                  'primary_predictions_recomputed_and_all_rules_metrics_replayed': replay_count,
                  'other_five_logits_and_predictions_unchanged': True,
                  'weights_thresholds_selection_modified': False},
        'limitations': ['Same physical simulator and PCG64; no measured-data validation.',
            'Fixed-pool conditional five-seed intervals; secondary comparisons uncorrected.',
            'Small circular shifts up to0.7us do not test broad arrival-time changes.',
            'New report is consumed; future tuning requires fresh identities.']}
    report = Path(plan['report_output'])
    write_json(report/'review.json', result)
    candidate = summary['comparisons']['segments124_vs_reference']
    control = summary['comparisons']['segments124_vs_global128']
    assessment = {'selected': summary['selection']['selected'], 'passed_vs_reference': candidate['passed'],
        'passed_vs_same_budget_global128': control['passed'], 'primary': 'locked',
        'main_report_Exact': summary['report_means']['segments124']['exact'],
        'five_separate_models_not_ensemble': True, 'review_audit': result['audit'],
        'next_error_priority': sorted(means['development_test']['segments124']['counts'],
            key=lambda c: means['development_test']['segments124']['counts'][c]['only_this_label_wrong'], reverse=True),
        'limits': result['limitations']}
    write_json(report/'candidate_assessment.json', assessment)
    html = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>ISRJ分段头完成复核</title>'+STYLE+'<h1>ISRJ分段统计头：完成复核与收益归因</h1><p>五种子分别评测，非集成。主阈值固定，仅更新ISRJ；其他五类输出不变。完成后复核不修改权重、阈值和check选型。</p><p><a href="analysis.html">自动报告</a> · <a href="plan.html">冻结方案</a> · <a href="review.json">复核JSON</a></p>'
    html += f'<p>全部完成，耗时{int(elapsed//60)}分{int(elapsed%60)}秒；10个小头、240轮、60000优化器步；270组自动审计及45组主报告独立重放通过。</p>'
    html += '<h2>同一8064新身份报告，原固定阈值</h2>'+table(['方法', 'Exact%', 'MacroF1%', 'ISRJ弱召回%', 'ISRJ弱AP%', 'ISRJ FPR%'], [[name]+[f'{100*m[key]:.3f}' for key in ['exact', 'macro_f1', 'isrj_weak_recall', 'isrj_weak_ap', 'isrj_fpr']] for name, m in summary['report_means'].items()])
    for name, comp in summary['comparisons'].items():
        d = comp['deltas']['exact']
        html += '<p>'+escape(name)+f'：Exact差{100*d["mean"]:+.3f}pp，条件95%区间[{100*d["paired95"][0]:+.3f},{100*d["paired95"][1]:+.3f}]，{d["positive_seeds"]}/5种子提高；正式验收'+('通过' if comp['passed'] else '未全部通过')+'。</p>'
    html += '<p>本轮冻结参考在新报告上为75.045%，上一轮75.724%来自不同身份池，不能跨池直接相减或叠加两轮增益。同预算全局头无可靠Exact收益，支持局部分段统计的额外价值。</p>'
    html += '<h2>错误修正与回退</h2>'+table(['方法', '修正原错误', '引入新错误', 'ISRJ FP', 'ISRJ FN', 'ISRJ弱FN'], [[name]+[f'{m[key]:.1f}' for key in ['exact_fixes', 'exact_regressions']]+[f'{m["counts"]["ISRJ"][key]:.1f}' for key in ['fp', 'fn', 'weak_fn']] for name, m in means['development_test'].items()])
    html += '<h2>ISRJ功率差分层</h2>'+table(['条件', '方法', '功率差dB', '正例数', '召回%', 'AP%', 'AUROC%'], [[role, name, f'{b["lower_db"]}–{b["upper_db"]}', b['support']]+[f'{100*b[key]:.3f}' if b[key] is not None else '无正例' for key in ['recall', 'aupr', 'auroc']] for role, values in means.items() for name in ['reference', 'segments124'] for b in values[name]['ISRJ_gap_bins']])
    html += '<h2>剩余单标签错误</h2>'+table(['标签', '仅该标签错误数均值', 'FP数均值', 'FN数均值'], [[c]+[f'{v[key]:.1f}' for key in ['only_this_label_wrong', 'fp', 'fn']] for c, v in means['development_test']['segments124']['counts'].items()])
    html += '<h2>辅助校准决策，不替代主验收</h2>'+table(['规则', '方法', 'Exact%', 'ISRJ弱召回%', 'ISRJ FPR%'], [[rule, name]+[f'{100*m[key]:.3f}' for key in ['exact', 'isrj_weak_recall', 'isrj_fpr']] for rule, values in auxiliary.items() for name, m in values['means'].items()])
    html += '<p>匹配的仅为calibration FP预算，报告实际FPR独立测量，不能宣称严格等测试误报率。固定身份池五种子条件区间；同一仿真器，缺少实测验证。分段统计为何有效仍是机制解释假设；轻微平移不能排除大范围到达时间和参数变化下的回退。</p></html>'
    (report/'review.html').write_text(html)
    print(json.dumps({'review': str(report/'review.html'), 'Exact': assessment['main_report_Exact'],
                      'next_error_priority': assessment['next_error_priority'], 'audit': result['audit']}, ensure_ascii=False))


def plot(suite):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plan = json.loads((suite/'plan.json').read_text())
    report = Path(plan['report_output'])
    data = json.loads((report/'review.json').read_text())
    summary = data['summary']
    names = ['global128', 'segments124']
    labels = {'reference': 'Frozen current reference', 'global128': 'Global128 control', 'segments124': 'ISRJ segment head'}
    colors = {'reference': '#666', 'global128': '#6589ad', 'segments124': '#229c8b'}
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3), constrained_layout=True)
    for i, name in enumerate(names):
        d = summary['comparisons'][name+'_vs_reference']['deltas']['exact']
        value, low, high = d['mean']*100, d['paired95'][0]*100, d['paired95'][1]*100
        axes[0].barh(i, value, color=colors[name], alpha=.85)
        axes[0].errorbar(value, i, xerr=np.array([[value-low], [high-value]]), fmt='none', color='#243848', capsize=4)
    axes[0].set_yticks(range(2), [labels[name] for name in names])
    axes[0].axvline(0, color='#777', lw=1)
    axes[0].axvline(.5, color='#aaa', ls='--', lw=1)
    axes[0].set_xlabel('Exact change vs reference (pp)')
    axes[0].set_title('Fixed thresholds; paired 95% CI')
    for name, m in summary['report_means'].items():
        axes[1].scatter(m['isrj_fpr']*100, m['isrj_weak_recall']*100, s=70, color=colors[name], label=labels[name])
    axes[1].set_xlabel('ISRJ false-positive rate (%)')
    axes[1].set_ylabel('ISRJ weak-positive recall (%)')
    axes[1].set_title('Recall improves while FPR falls')
    axes[1].margins(x=.25, y=.2)
    axes[1].legend(fontsize=7, loc='upper right')
    for name in ['reference', 'global128', 'segments124']:
        bins = data['means']['development_test'][name]['ISRJ_gap_bins'][:4]
        axes[2].plot([(b['lower_db']+b['upper_db'])/2 for b in bins], [b['recall']*100 for b in bins], '-o', label=labels[name], color=colors[name], markersize=4)
    axes[2].set_xlabel('Power gap (dB; bin centers)')
    axes[2].set_ylabel('ISRJ positive recall (%)')
    axes[2].set_title('Original 10--30 dB report')
    axes[2].legend(fontsize=7)
    for axis in axes:
        axis.spines[['top', 'right']].set_visible(False)
        axis.grid(alpha=.15)
    fig.savefig(report/'outcome_overview.png', dpi=180)
    plt.close(fig)
    html = (report/'review.html').read_text()
    if 'outcome_overview.png' not in html:
        (report/'review.html').write_text(html.replace('</html>', '<p><img src="outcome_overview.png" style="max-width:100%" alt="ISRJ收益与功率差分层"></p></html>'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', type=Path, required=True)
    parser.add_argument('--plot-only', action='store_true')
    args = parser.parse_args()
    if args.plot_only:
        plot(args.suite.resolve())
    else:
        review(args.suite.resolve())
