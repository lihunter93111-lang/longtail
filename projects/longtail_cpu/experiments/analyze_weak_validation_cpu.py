"""Recheck saved CPU validation evidence and explain its limits; no new fitting."""
import argparse
import base64
from datetime import datetime
from html import escape
import importlib.util
import json
from pathlib import Path

import numpy as np

LABELS = ['FMB', 'AMA', 'SawFM', 'SinFM', 'MFT', 'ISRJ']
METHODS = list('ABCD')
NAMES = {'A': 'A 现有融合', 'B': 'B 自然数据32k', 'C': 'C 弱难例32k', 'D': 'D 弱难例＋加权'}


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def equal(actual, expected):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            equal(actual[key], expected[key])
    elif isinstance(expected, list):
        assert len(actual) == len(expected)
        for a, e in zip(actual, expected):
            equal(a, e)
    elif isinstance(expected, (float, np.floating)):
        assert np.isclose(actual, expected, rtol=0, atol=1e-12, equal_nan=True), (actual, expected)
    else:
        assert actual == expected, (actual, expected)


def audit(suite):
    import h5py
    import torch
    from scipy.stats import t
    from longtail.io import sha256
    from longtail.engine.checkpoint import load_checkpoint
    from longtail.data.dataset import read_metadata
    from longtail.data.power_truth import read_power_truth
    from longtail.data.sampling import EpochBudgetSampler
    from longtail.metrics import fit_thresholds, probabilities
    from longtail.decision import fit_fpr_caps

    torch.set_num_threads(2)
    plan = read(suite / 'plan.json'); out = Path(plan['report_output'])
    result = read(out / 'results.json'); frozen = read(suite / 'frozen_ready.json')
    done = read(suite / 'pipeline_completed.json'); split = read(suite / 'context_split.json')
    assert done['status'] == 'completed' and done['new_models_completed'] == 15 and done['evaluated_models'] == 25
    assert sha256(suite / 'plan.json') == frozen['plan_sha256']
    assert sha256(suite / 'selected_candidate.json') == frozen['selection_sha256']
    equal(result['plan'], plan)
    for item in plan['source_manifest']:
        assert sha256(suite / 'source_snapshot' / item['path']) == item['sha256']
    spec = importlib.util.spec_from_file_location('frozen_weak_runner', suite / 'source_snapshot/experiments/run_weak_validation_cpu.py')
    runner = importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)
    barrier = datetime.fromisoformat(frozen['finished'])
    manifest_path = Path(plan['evaluation_manifest']); manifest = read(manifest_path)
    root = manifest_path.parent; dev_path = root / manifest['files']['development']['path']
    y, identities, _ = read_metadata(dev_path); truth = read_power_truth(dev_path, y)
    checked_files = set()
    for item in [manifest_path] + [Path(read(j['config'])['data']['manifest']) for j in plan['training_jobs']]:
        m = read(item)
        for f in m['files'].values():
            path = item.parent / f['path']
            if path.resolve() not in checked_files:
                assert sha256(path) == f['sha256']; checked_files.add(path.resolve())
    generation = read(root / 'generation_plan.json')
    report_ids = identities[split['roles']['development_test']]
    for path in generation['checked_identity_files']:
        with h5py.File(path) as handle:
            assert not np.intersect1d(report_ids, handle['receiver_noise_seed'][:].ravel()).size, path
    with h5py.File(dev_path) as new, h5py.File(Path(generation['source_dataset_manifest']).parent / 'development.h5') as old:
        for role in ['calibration', 'check']:
            for name in old:
                np.testing.assert_array_equal(new[name][:, split['roles'][role]], old[name][:, split['roles'][role]])
    for i, role in enumerate(['calibration', 'check', 'development_test']):
        for other in ['calibration', 'check', 'development_test'][i+1:]:
            assert not set(split['roles'][role]) & set(split['roles'][other])
    train_manifest = Path(read(plan['training_jobs'][0]['config'])['data']['manifest'])
    train_y, train_ids, _ = read_metadata(train_manifest.parent / read(train_manifest)['files']['train']['path'])
    assert not np.intersect1d(train_ids, identities).size
    train_code = train_y.astype(int) @ (1 << np.arange(6))
    np.testing.assert_array_equal(np.bincount(train_code, minlength=64)[1:], np.array(split['fit_combination_counts']) * 4)
    histories, job_audits, coverage, thresholds, cal_by_seed, sampling_by_seed = {}, [], {}, {}, {}, {}
    for job in plan['evaluation_jobs']:
        path = Path(job['output']); held = frozen['jobs'][job['id']]
        assert sha256(job['checkpoint']) == held['checkpoint_sha256']
        assert sha256(path / 'decisions.json') == held['decisions_sha256']
        payload, config = load_checkpoint(job['checkpoint'])
        assert config['model'] == plan['model'] and config['dataloader']['preprocess'] == plan['dataloader']['preprocess']
        if job['reference']:
            assert held['checkpoint_sha256'] == job['checkpoint_sha256']
        else:
            assert config == read(job['config']) == read(path / 'config.json')
            assert config['train']['epochs'] == 24 and config['train']['draws_per_epoch'] == 8000
            assert config['dataloader']['batch_size'] == 32 and config['train']['device'] == 'cpu'
            assert config['train']['loss'] == ('weak_positive_bce' if job['method'] == 'D' else 'bce')
            history = read(path / 'history.json'); status = read(path / 'status.json')
            assert len(history) == 24 and status['status'] == 'completed' and status['parameters'] == 257511
            assert all(h['samples_seen'] == 8000 and h['sampler_padding'] == 0 and h['fit_pool_size'] == 32000 for h in history)
            best_epoch = int(np.argmax([h['validation']['macro_aupr'] for h in history])) + 1
            assert payload['epoch'] == status['best_epoch'] == best_epoch
            last, _ = load_checkpoint(path / 'last.pt')
            assert last['epoch'] == 24 and {int(v['step']) for v in last['optimizer']['state'].values()} == {6000}
            del last
            sampler = EpochBudgetSampler(train_y, job['seed'], 8000); visited = set()
            for h in history:
                sampler.set_epoch(h['epoch']); visited.update(sampler); equal(h['sampling'], sampler.audit())
            prior = sampling_by_seed.setdefault(job['seed'], [h['sampling'] for h in history])
            equal(prior, [h['sampling'] for h in history])
            coverage[job['id']] = len(visited)
            histories[job['id']] = {'best_epoch': best_epoch, 'history': [
                {'epoch': h['epoch'], 'loss': h['loss'], 'check_aupr': h['validation']['macro_aupr'], 'epoch_seconds': h['epoch_seconds']}
                for h in history]}
            phase = read(suite / 'phases' / f'{job["id"]}_train.json')
            assert phase['status'] == 'completed' and datetime.fromisoformat(phase['finished']) <= barrier
            assert phase['checkpoint_sha256'] == held['checkpoint_sha256'] and phase['decisions_sha256'] == held['decisions_sha256']
        del payload
        phase = read(suite / 'phases' / f'{job["id"]}_report.json')
        assert phase['status'] == 'completed' and datetime.fromisoformat(phase['started']) >= barrier
        assert phase['checkpoint_sha256'] == held['checkpoint_sha256'] and phase['decisions_sha256'] == held['decisions_sha256']
        records = {}
        for role in ['calibration', 'check', 'development_test']:
            folder = path / 'evaluation' / role; z = dict(np.load(folder / 'predictions.npz')); records[role] = z
            assert set(z['indices'].tolist()) == set(split['roles'][role]) and np.isfinite(z['logits']).all()
            np.testing.assert_array_equal(z['targets'], y[z['indices']]); np.testing.assert_array_equal(z['identities'], identities[z['indices']])
            for key in ['weak_mask', 'relative_power_db']:
                np.testing.assert_array_equal(z[key], truth[key][z['indices']])
            origin = read(folder / 'source.json')
            assert origin['checkpoint_sha256'] == held['checkpoint_sha256'] and origin['context_split_sha256'] == split['sha256']
            assert origin['context_manifest_sha256'] == sha256(manifest_path) and origin['predictions_sha256'] == sha256(folder / 'predictions.npz')
            if job['reference'] and role != 'development_test':
                assert sha256(folder / 'predictions.npz') == sha256(Path(job['source_output']) / 'evaluation' / role / 'predictions.npz')
        cal = records['calibration']; decisions = read(path / 'decisions.json')
        assert decisions['fit_role'] == 'calibration' and decisions['context_split_sha256'] == split['sha256']
        assert decisions['checkpoint_sha256'] == held['checkpoint_sha256'] and decisions['calibration_predictions_sha256'] == sha256(path / 'evaluation/calibration/predictions.npz')
        equal(decisions['thresholds']['f1'], fit_thresholds(cal['targets'], cal['logits'], None))
        if job['method'] == 'A':
            cal_by_seed[job['seed']] = cal
        baseline = cal_by_seed[job['seed']]
        caps = ((probabilities(baseline['logits']) >= fit_thresholds(baseline['targets'], baseline['logits'], None)) & ~baseline['targets'].astype(bool)).sum(0)
        th, meta = fit_fpr_caps(cal['targets'], cal['logits'], caps)
        equal(th, decisions['thresholds']['matched_fpr']); equal(meta, decisions['matched_fpr_fit'])
        assert all(meta['empirical_caps_satisfied'])
        thresholds[job['id']] = decisions['thresholds']
        job_audits.append({'job': job['id'], 'reference': job['reference'], **held})
    per_label, errors, fpr_deltas, rows_by_role = {}, {}, {}, {}
    recomputed = 0
    for role in ['calibration', 'check', 'development_test']:
        per_label[role] = {}; errors[role] = {}; rows_by_role[role] = {}
        for method in METHODS:
            per_label[role][method] = {d: [] for d in ['f1', 'matched_fpr']}; errors[role][method] = []
            for seed in range(2050, 2055):
                job = next(j for j in plan['evaluation_jobs'] if j['id'] == f'{method}_seed{seed}')
                fjob = next(j for j in plan['references'] if j['id'] == f'FMB_seed{seed}')
                z = dict(np.load(Path(job['output']) / 'evaluation' / role / 'predictions.npz'))
                fmb = dict(np.load(Path(fjob['output']) / 'evaluation' / role / 'predictions.npz'))
                folder = suite / 'views' / job['id'] / 'evaluation' / role; saved = read(folder / 'metrics.json')
                assert saved['checkpoint_sha256'] == frozen['jobs'][job['id']]['checkpoint_sha256']
                assert saved['FMB_checkpoint_sha256'] == frozen['jobs'][fjob['id']]['checkpoint_sha256']
                assert saved['decisions_sha256'] == frozen['jobs'][job['id']]['decisions_sha256']
                fused = dict(np.load(folder / 'fusion_predictions.npz'))
                np.testing.assert_array_equal(fused['logits'][:, 0], fmb['logits'][:, 0])
                np.testing.assert_array_equal(fused['logits'][:, 1:], z['logits'][:, 1:])
                for key in z.keys() - {'logits'}:
                    np.testing.assert_array_equal(fused[key], z[key])
                for decision, th in thresholds[job['id']].items():
                    for view in ['single', 'fusion']:
                        threshold = th.copy(); logits = z['logits'] if view == 'single' else fused['logits']
                        if view == 'fusion': threshold[0] = .5
                        measured = runner.metric(z, logits, threshold, plan, split)
                        equal(saved['metrics'][view][decision], measured)
                        yy = z['targets'].astype(bool); pred = probabilities(logits) >= threshold
                        assert measured['exact'] == (pred == yy).all(1).mean()
                        for c, label in enumerate(LABELS):
                            tp = int((pred[:, c] & yy[:, c]).sum()); fp = int((pred[:, c] & ~yy[:, c]).sum()); fn = int((~pred[:, c] & yy[:, c]).sum())
                            assert measured['per_label'][label]['tp'] == tp and measured['per_label'][label]['fp'] == fp and measured['per_label'][label]['fn'] == fn
                        row = next(v for v in result['rows'] if v['method'] == method and v['seed'] == seed and v['role'] == role and v['view'] == view and v['decision'] == decision)
                        for key in runner.KEYS: equal(row[key], measured[key])
                        recomputed += 1
                    per_label[role][method][decision].append(saved['metrics']['fusion'][decision])
                yy = z['targets'].astype(bool); th = thresholds[job['id']]['f1'].copy(); th[0] = .5
                pred = probabilities(fused['logits']) >= th; fn = yy & ~pred; weak = z['weak_mask']; bad = (pred != yy).any(1)
                errors[role][method].append({'seed': seed, 'weak_positive_share': float(weak.sum() / yy.sum()),
                    'weak_fraction_of_false_negatives': float((fn & weak).sum() / fn.sum()),
                    'only_weak_fn_fraction_of_wrong_samples': float((bad & np.all((pred != yy) == (fn & weak), axis=1)).sum() / bad.sum()),
                    'class_error_counts': (pred != yy).sum(0).tolist(), 'false_negatives': fn.sum(0).tolist(), 'false_positives': (pred & ~yy).sum(0).tolist()})
                rows_by_role[role].setdefault(method, []).append(saved['metrics']['fusion']['f1'])
    means = {role: {m: {d: {c: {k: float(np.mean([v['per_label'][c][k] for v in ms])) for k in ['f1', 'aupr', 'recall', 'negative_fpr']} for c in LABELS} for d, ms in ds.items()} for m, ds in methods.items()} for role, methods in per_label.items()}
    for role in means:
        fpr_deltas[role] = {d: {m: {c: means[role][m][d][c]['negative_fpr'] - means[role]['A'][d][c]['negative_fpr'] for c in LABELS} for m in 'BCD'} for d in ['f1', 'matched_fpr']}
    eligible = [m for m in 'BCD' if np.mean([v['exact'] for v in rows_by_role['check'][m]]) > np.mean([v['exact'] for v in rows_by_role['check']['A']]) and np.mean([v['tail_f1'] for v in rows_by_role['check'][m]]) >= np.mean([v['tail_f1'] for v in rows_by_role['check']['A']]) - .003 and max(fpr_deltas['check']['f1'][m].values()) <= .01]
    chosen = max(eligible, key=lambda m: np.mean([v['exact'] for v in rows_by_role['check'][m]])) if eligible else None
    selection = read(suite / 'selected_candidate.json')
    assert eligible == selection['eligible'] and chosen == selection['selected_candidate']
    for role, views in result['aggregate'].items():
        for view, decisions in views.items():
            for decision, methods in decisions.items():
                for method, metrics in methods.items():
                    rows = [row for row in result['rows'] if row['role'] == role and row['view'] == view and row['decision'] == decision and row['method'] == method]
                    assert len(rows) == 5
                    for key, value in metrics.items():
                        equal(value, {'mean': float(np.mean([row[key] for row in rows])), 'std': float(np.std([row[key] for row in rows], ddof=1))})
    for decision, comparisons in result['effects'].items():
        for name, effect in comparisons.items():
            a, b = name.split('_vs_')
            for key, observed in effect.items():
                delta = np.array([x[key] - y[key] for x, y in zip(per_label['development_test'][a][decision], per_label['development_test'][b][decision])])
                radius = t.ppf(.975, 4) * delta.std(ddof=1) / np.sqrt(5)
                equal(observed, {'mean': float(delta.mean()), 'paired_seed_95_t_interval': [float(delta.mean()-radius), float(delta.mean()+radius)], 'per_seed': delta.tolist()})
    primary = result['effects']['f1'][chosen + '_vs_A']; success = plan['protocol']['success']
    conditions = {'exact_gain_ge_1pp': primary['exact']['mean'] >= success['minimum_exact_gain'],
                  'positive_seeds_ge4': sum(v > 0 for v in primary['exact']['per_seed']) >= success['minimum_positive_seeds'],
                  'paired_interval_lower_positive': primary['exact']['paired_seed_95_t_interval'][0] > 0,
                  'tail_f1_drop_le_0.3pp': primary['tail_f1']['mean'] >= -success['maximum_tail_f1_drop'],
                  'class_fpr_rise_le_1pp': max(fpr_deltas['development_test']['f1'][chosen].values()) <= success['maximum_class_fpr_rise']}
    equal(conditions, result['acceptance']['conditions']); assert all(conditions.values()) == result['acceptance']['passed']
    a_f1 = per_label['development_test']['A']['f1']; b_caps = per_label['development_test']['B']['matched_fpr']
    delta = np.array([b['exact'] - a['exact'] for a, b in zip(a_f1, b_caps)])
    radius = t.ppf(.975, 4) * delta.std(ddof=1) / np.sqrt(5)
    secondary = {'status': 'Exploratory comparison of predeclared decisions; no refitting',
        'B_matched_vs_A_f1_exact': {'mean': float(delta.mean()), 'paired_seed_95_t_interval': [float(delta.mean()-radius), float(delta.mean()+radius)], 'per_seed': delta.tolist()},
        'class_fpr_delta_vs_A_f1': {c: float(np.mean([b['per_label'][c]['negative_fpr'] - a['per_label'][c]['negative_fpr'] for a,b in zip(a_f1,b_caps)])) for c in LABELS},
        'B_SawFM_tradeoff': {d: {key: float(np.mean([v['weak_component']['per_label']['SawFM'][key] for v in per_label['development_test']['B'][d]])) for key in ['weak_recall','negative_fpr']} for d in ['f1','matched_fpr']}}
    completion_audit = {'status': 'passed', 'new_models': 15, 'reused_models': 10, 'training_epochs': 360, 'optimizer_steps': 90000,
        'frozen_source_files_verified': len(plan['source_manifest']), 'unique_data_files_hashed': len(checked_files),
        'calibration_decisions_replayed': 25, 'metric_conditions_recomputed': recomputed, 'report_identity_files_rechecked': len(generation['checked_identity_files']),
        'report_identity_overlap': 0, 'calibration_check_arrays_preserved': True, 'sampling_records_replayed': 360,
        'B_C_D_sampling_identical_within_seed': True, 'independent_seed_epoch_draw_orders': len({a['draw_order_sha256'] for v in sampling_by_seed.values() for a in v}),
        'models_decisions_selection_frozen_before_report': True, 'freeze_time': frozen['finished'], 'completion': done,
        'primary_acceptance_replayed': result['acceptance'], 'jobs': job_audits}
    analysis = {'suite': str(suite), 'aggregate': result['aggregate'], 'effects': result['effects'], 'selection': selection, 'acceptance': result['acceptance'],
        'per_label': means, 'per_seed_metrics': per_label, 'fpr_deltas': fpr_deltas, 'error_diagnosis': errors, 'coverage': coverage,
        'histories': histories, 'data_audit': plan['data_audit'], 'secondary_decision_analysis': secondary, 'audit': completion_audit,
        'interpretation': 'Only C is the preselected primary candidate, and it fails its paired-interval gate. B is the strongest exploratory candidate but breaches the per-class FPR gate on SawFM. No replacement of the primary candidate or post-report threshold fitting.'}
    write(out / 'completion_audit.json', completion_audit); write(out / 'analysis.json', analysis)
    print(json.dumps({'audit': 'passed', 'recomputed_conditions': recomputed, 'selected': chosen, 'primary_passed': result['acceptance']['passed'], 'coverage_range': [min(coverage.values()), max(coverage.values())]}, ensure_ascii=False))


def render(suite):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font_manager.fontManager.addfont('/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf')
    plt.rcParams.update({'font.family': ['DejaVu Sans', 'Droid Sans Fallback'], 'axes.unicode_minus': False, 'axes.spines.top': False, 'axes.spines.right': False})
    out = Path(read(suite / 'plan.json')['report_output']); a = read(out / 'analysis.json'); agg = a['aggregate']['development_test']['fusion']
    colors = ['#64798f', '#198c96', '#c58b50', '#a27297']
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout='constrained')
    for i, method in enumerate(METHODS):
        for j, decision in enumerate(['f1', 'matched_fpr']):
            v = agg[decision][method]['exact']; axes[0].bar(i+(j-.5)*.32, v['mean']*100, width=.3, color=colors[i], alpha=1 if j==0 else .5, yerr=v['std']*100, capsize=3)
        if method != 'A':
            delta = a['effects']['f1'][method+'_vs_A']['exact']; lo, hi = delta['paired_seed_95_t_interval']
            axes[1].errorbar(i-1, delta['mean']*100, yerr=np.array([[delta['mean']-lo], [hi-delta['mean']]])*100, fmt='o', color=colors[i], capsize=5)
            axes[1].scatter(np.full(5, i-1)+np.linspace(-.12,.12,5), np.array(delta['per_seed'])*100, s=17, color=colors[i], alpha=.5)
    axes[0].set(xticks=range(4), xticklabels=['A 基线', 'B 自然32k', 'C 弱难例', 'D 弱加权'], ylabel='Exact (%)', ylim=(64,75), title='深色F1阈值；浅色calibration误报预算；误差条为标准差')
    axes[1].axhline(0, color='#aaa'); axes[1].axhline(1, color='#aaa', linestyle=':'); axes[1].set(xticks=range(3), xticklabels=['B−A（探索）', 'C−A（主候选）', 'D−A（探索）'], ylabel='Exact差 pp', title='五种子配对95% t区间与逐种子差值')
    fig.savefig(out/'accuracy_and_evidence.png', dpi=170); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.6), layout='constrained')
    for ax, key, title in zip(axes, ['negative_fpr', 'recall'], ['逐类误报变化：负值更好', '逐类整体召回变化']):
        matrix = np.array([[100*(a['per_label']['development_test'][m]['f1'][c][key]-a['per_label']['development_test']['A']['f1'][c][key]) for c in LABELS] for m in 'BCD'])
        ax.imshow(matrix, cmap='RdBu_r', vmin=-2.5, vmax=2.5, aspect='auto'); ax.set(xticks=range(6), xticklabels=LABELS, yticks=range(3), yticklabels=list('BCD'), title=title+' pp')
        for i in range(3):
            for j in range(6): ax.text(j,i,f'{matrix[i,j]:+.2f}',ha='center',va='center',color='white' if abs(matrix[i,j])>1.8 else '#123')
    fig.savefig(out/'class_tradeoffs.png', dpi=170); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout='constrained')
    for method, color in zip(METHODS, colors):
        rows = a['per_seed_metrics']['development_test'][method]['f1']
        v = [[r['weak_component']['per_label'][c]['weak_recall']*100 for c in LABELS] for r in rows]
        axes[0].plot(LABELS, np.mean(v, axis=0), 'o-', color=color, label=NAMES[method])
        if method != 'A':
            h = np.array([[v['check_aupr']*100 for v in a['histories'][f'{method}_seed{s}']['history']] for s in range(2050,2055)])
            axes[1].plot(range(1,25), h.mean(0), color=color, label=NAMES[method])
    axes[0].set(ylabel='弱分量召回 (%)', title='自然新报告池：目标难类为SawFM/MFT/ISRJ'); axes[0].legend(fontsize=8)
    axes[1].set(xlabel='固定预算轮数（250步/轮）', ylabel='check Macro-AUPR (%)', ylim=(94,99), title='check训练曲线：保持6000步'); axes[1].legend(fontsize=8)
    fig.savefig(out/'weak_recall_and_training.png', dpi=170); plt.close(fig)
    def table(headers, rows):
        return '<div class="scroll"><table><tr>'+''.join('<th>'+escape(str(v))+'</th>' for v in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+escape(str(v))+'</td>' for v in row)+'</tr>' for row in rows)+'</table></div>'
    def pic(name):
        return '<img alt="实验分析图" src="data:image/png;base64,'+base64.b64encode((out/name).read_bytes()).decode()+'">'
    text='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>弱分量CPU验证：结果检查与分析</title><style>body{font:16px/1.8 system-ui;color:#234;background:#f0f4f6;margin:0}main{max-width:1180px;margin:28px auto;padding:32px;background:white}h1,h2{color:#155e75}table{border-collapse:collapse;width:100%;font-size:14px;margin:18px 0}td,th{border:1px solid #d6e0e6;padding:8px}th{background:#155e75;color:white}.scroll{overflow-x:auto}.note{background:#edf6f6;border-left:5px solid #248a94;padding:16px}img{width:100%}a{color:#126f84}</style><main>'
    text+='<h1>弱分量CPU验证：结果检查与分析</h1><p>2026-11-04复核。训练于11月3日16:52:35启动，18:37:45完成，实际1小时45分10秒。15个新增模型均完成24轮、6000步；复用10个冻结模型，25个模型均完成新报告测评。全部方法固定同种子的FMB组合专家0.5输出，其余五类来自对应分支。</p>'
    text+='<p class="note">扩大自然身份池B的指标最强；C的高比例条件弱难例和D的弱正例加权未超过B。正式主候选为事先在check选出的C，Exact提高1.195个百分点，但配对区间跨零，主验收未通过。B属于探索性结果，且SawFM误报上升超过预设逐类门槛，不能把B改称本轮正式通过的主候选。</p>'
    keys=['exact','macro_f1','macro_aupr','tail_f1','target_weak_recall','mean_fpr']
    text+='<h2>同一4032新身份上的融合结果</h2>'
    text+=table(['组','Exact %','Macro-F1 %','Macro-AUPR %','尾类F1 %','三目标弱召回 %','平均FPR %'],[[NAMES[m]]+[f"{agg['f1'][m][k]['mean']*100:.3f}" for k in keys] for m in METHODS])
    text+='<p>Exact表示一个样本的六个标签全部正确；尾类为AMA/SinFM，三目标为SawFM/MFT/ISRJ。每项为五种子均值。当前A为68.953%，上轮A为68.423%，两次报告身份不同，增益应使用本轮同池差值。</p>'+pic('accuracy_and_evidence.png')
    text+='<h2>收益来源</h2><p>B−A的Exact为+2.252 pp，条件配对95%区间[+0.821,+3.683] pp，五种子均正向；Macro-AUPR提高0.226 pp，目标弱召回提高4.244 pp。按calibration相同误报预算的辅助决策仍提高Exact 1.954 pp，说明收益包含排序改善，并非仅由F1阈值工作点变化解释。辅助决策只匹配calibration经验FP预算，不保证新报告实际FPR相等。</p>'
    text+='<p>C−B的Exact为−1.057 pp，目标弱召回为−2.714 pp，Macro-AUPR为−0.119 pp。B/C的物理参数、噪声及抽样相同，变化在训练功率分布；本轮不支持把当前强度的难例增强作为改进。13718/32000条波形被条件修改，弱正例占比从22.73%上升到33.29%，而报告仍为原自然功率分布。这提示增强强度与训练预算可能不匹配；尚不能单独归因为某种具体训练机制。</p>'
    text+='<p>D−C的Exact为−0.432 pp，区间跨零；Macro-AUPR为−0.087 pp，五种子均下降，非弱分量召回下降0.256 pp。本轮没有证据支持再增加1.5倍弱正例权重。归一化防止整类平均权重无意放大，但仍会改变难例和普通样本之间的梯度分配。</p>'+pic('weak_recall_and_training.png')
    text+='<h2>误报与正式验收</h2><p>B的平均FPR虽由4.913%降至4.587%，SawFM逐类FPR却由6.915%升至8.488%，增加1.573 pp。check也增加1.169 pp，超过预定1 pp门槛，因此B在报告开放之前已被排除。D在check也因SawFM误报被排除。C为唯一合格check候选，新报告中4/5种子Exact正向，均值增益达到1 pp，但95%区间为[−0.565,+2.956] pp，不能宣称正式验收成功。</p>'+pic('class_tradeoffs.png')
    text+=table(['主候选C的预定条件','复核结果'],[[k,str(v)] for k,v in a['acceptance']['conditions'].items()])
    text+='<h2>现有辅助方案：B加经验误报预算</h2><p>本轮预定义的辅助阈值已经提供一个较均衡的候选：B的Exact为70.813%，相对采用同样辅助决策的A（68.859%）提高1.954 pp，五种子均提高；六类平均FPR均未上升，SawFM为6.169%，低于A的6.996%。相对当前A主决策（68.953%）提高1.860 pp，条件95%区间[+0.781,+2.939] pp。它只作为预定义辅助方案的探索证据，不改写主候选C未通过的结论。</p><p>代价同样存在：B从F1改为误报预算阈值，Exact降低0.392 pp，SawFM弱召回从57.143%降至49.814%，也低于A主决策的52.091%。目标三类平均弱召回仍为60.053%，高于A主决策56.818%。不能把整体指标改善解释为每一种弱分量都改善。</p>'
    text+='<h2>下一步验证重点</h2><p>优先把B加当前已实现的经验误报预算阈值作为下一轮预先指定候选，同时把SawFM弱召回设为观察指标。固定这些权重和决策，用新报告身份确认；不使用当前已消费报告重新拟合阈值。若希望兼顾SawFM弱召回，再只用calibration/check设计其召回下限，作为单独的开发比较。</p><p>第二优先级是在B上验证更长预算：当前32000池每次只训练192000次样本曝光，平均每身份6次，A平均24次；先对比6000与12000步，预先设定学习率日程，避免把新日程混入其他对照。若继续验证难例增强，先单独测试10%/20%的全池比例，固定自然背景样本和损失，随后再研究后期逐步加入；本轮没有验证较低比例或课程学习有效。</p>'
    text+='<h2>完整性与适用边界</h2><p>复核93份冻结源文件、15个模型的6000步及360轮抽样、25份校准决策、240组单模型/融合测评条件，并重放主候选选择与验收；数据哈希、calibration/check保留和56份旧文件的报告身份无交集检查通过。模型、阈值和check候选于18:35:43冻结，首个报告作业18:35:44启动。C/D使用完全相同的训练文件，同种子的B/C/D逐轮抽样一致；跨五种子和24轮共有120个不同抽样流。</p><p>统计区间以本次固定模拟报告池为条件，仅反映五个训练种子的差异，探索性比较没有多重检验校正。calibration/check已复用多轮，未来应增加独立开发身份确认。数据为原v3物理方程的Python PCG64移植，未证明MATLAB随机流逐点等价，也不代表实测泛化。没有修改已冻结权重、阈值、主候选或原始报告结果。</p><p><a href="completion_audit.json">完成审计</a> · <a href="analysis.json">完整逐类分析</a> · <a href="results.csv">原始逐种子CSV</a> · <a href="plan.html">冻结方案</a></p></main></html>'
    (out/'analysis.html').write_text(text)
    print(json.dumps({'analysis':str(out/'analysis.html'),'figures':3},ensure_ascii=False))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--suite',type=Path,required=True);parser.add_argument('--render-only',action='store_true')
    args=parser.parse_args();suite=args.suite.resolve()
    if not args.render_only: audit(suite)
    else: render(suite)
