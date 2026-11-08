"""Two-seed screening, conditional five-seed confirmation and frozen reports."""
import argparse
import fcntl
from html import escape
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from longtail.io import read_json, sha256, write_json
from longtail.segment_statistics import METHODS, now

ROOT = Path(__file__).resolve().parents[1]
SEEDS = list(range(2050, 2055))
SLOTS = [list(range(8+4*i, 12+4*i)) for i in range(4)]+[list(range(40, 44))]
STYLE = '<style>body{font:16px/1.8 system-ui;max-width:1280px;margin:24px auto;padding:24px;color:#234}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:7px;border:1px solid #ccd}th{background:#e9f3f4}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style>'


def table(headers, rows):
    return '<table><tr>'+''.join('<th>'+escape(str(v))+'</th>' for v in headers)+'</tr>'+''.join(
        '<tr>'+''.join('<td>'+escape(str(v))+'</td>' for v in row)+'</tr>' for row in rows)+'</table>'


def prepare(suite, data, report):
    import numpy as np
    import torch
    from longtail.data.splits import build_split
    from longtail.data.dataset import IQDataset
    from longtail.replication_temporal import load_backbone
    from longtail.models.temporal_heads import unpooled_features
    from longtail.segment_statistics import segment_statistics, StatisticsResidual, compose
    if suite.exists():
        raise FileExistsError('Use a new suite')
    ready = read_json(data/'generation_complete.json')
    assert ready['status'] == 'completed' and ready['identity_overlap_existing'] == 0
    assert set(sum(SLOTS, [])) <= os.sched_getaffinity(0)
    previous = ROOT/'runs/calibration_targeted_cpu_20261104'
    old = read_json(previous/'plan.json')
    selected = read_json(previous/'selected_models.json')
    refs = {}
    for row in selected['models']:
        seed = row['seed']
        assert row['selected'] == 'pair1' and row['decision_rule'] == 'locked'
        refs[str(seed)] = {'new': row['frozen_new_reference'], 'target_head': row['target_head_copy'],
                           'FMB': old['references'][str(seed)]['FMB']}
        refs[str(seed)]['new']['thresholds'] = row['thresholds']
        for source in [refs[str(seed)]['new'][k] for k in ['B', 'linear', 'local']]+[refs[str(seed)]['FMB'], row['target_head_copy']]:
            assert sha256(source['path']) == source['sha256']
    torch.set_num_threads(4)
    model, cfg = load_backbone(Path(refs['2050']['new']['B']['path']))
    manifest = read_json(ready['manifest'])
    dataset = IQDataset(data/'train.h5', np.arange(8), cfg['dataloader']['preprocess'])
    x = torch.stack([dataset[i]['signal'] for i in range(8)])
    dataset.close()
    with torch.inference_mode():
        features, temporal = unpooled_features(model, x)
        torch.testing.assert_close(features, model.forward_features(x), rtol=0, atol=0)
        pooled = segment_statistics(temporal)
        torch.testing.assert_close(pooled[:, :256], model.global_pool(temporal), rtol=0, atol=0)
    residual = StatisticsResidual()
    assert sum(p.numel() for p in residual.parameters()) == 4194
    zz = np.random.default_rng(2050).normal(size=(8, 6)).astype(np.float32)
    np.testing.assert_array_equal(compose(zz, residual(features.clone()).detach().numpy()), zz)
    optimizer = torch.optim.AdamW(residual.parameters(), lr=.0003)
    optimizer.zero_grad()
    residual(features.clone()).sub(1).square().mean().backward()
    torch.nn.utils.clip_grad_norm_(residual.parameters(), 5, error_if_nonfinite=True)
    optimizer.step()
    suite.mkdir(parents=True)
    report.mkdir(parents=True, exist_ok=True)
    split_cfg = json.loads(json.dumps(cfg['data']))
    split_cfg['per_combination'] = {'calibration': 128, 'check': 32, 'development_test': 128}
    split = build_split(manifest, data, split_cfg)
    write_json(suite/'split.json', split)
    hashes = {ready['manifest']: sha256(ready['manifest']), str(data/'train.h5'): manifest['files']['train']['sha256']}
    hashes.update({v['path']: v['sha256'] for v in ready['files'].values()})
    hashes[ready['pair_fit']['path']] = ready['pair_fit']['sha256']
    source = []
    files = sorted((ROOT/'src').rglob('*.py'))+sorted((ROOT/'experiments').glob('*.py'))+sorted((ROOT/'tests').glob('*.py'))+sorted((ROOT/'configs').glob('*.json'))+[ROOT/'pyproject.toml', ROOT/'uv.lock']
    for file in files:
        dest = suite/'source_snapshot'/file.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(file, dest)
        source.append({'path': str(file.relative_to(ROOT)), 'sha256': sha256(dest)})
    protocol = {
        'order': 'two-seed screen -> check-only promotion -> remaining three seeds -> freeze -> report',
        'CPU_only': True, 'parallel_jobs_max': 5, 'physical_cores_per_job': 4,
        'pilot_seeds': SEEDS[:2], 'confirmation_seeds': SEEDS, 'methods': METHODS,
        'features': {'global128': 'current learned128-dimensional projection',
                     'global256': 'pre-projection full global mean/std256',
                     'segments124': 'temporal128x375; concatenate mean/std in1/2/4 disjoint bins:1792'},
        'capacity_control': 'Each feature standardized and PCA128 fit on same natural32k only, no whitening; identical128->32 GELU->2 residual head4194 trainable parameters.',
        'frozen_transform_sizes': '128x128 /128x256 /128x1792 PCA matrices; not total-memory-matched; retained variance reported',
        'reference': 'previous check-selected6.25% pair1 candidate; original B/linear/local/FMB/target-head weights and thresholds immutable',
        'initialization': 'all added residual outputs zero: exactly same initial candidate logits; original global path retained',
        'changed_labels': ['SawFM', 'MFT'], 'other_four': 'exact copied reference logits and thresholds',
        'training': {'epochs': 24, 'steps_per_epoch': 250, 'steps_per_model': 6000,
                     'batch_size': 32, 'natural_views_per_step': 30, 'paired_views_per_step': 2,
                     'paired_fraction': .0625, 'sampling': 'same seed/epoch draw streams for all feature arms; six target-gap cells balanced',
                     'optimizer': 'AdamW', 'learning_rate': .0003, 'weight_decay': .0001,
                     'schedule': 'cosine24', 'loss': 'ordinary BCE only two target logits; gradient norm cap5'},
        'checkpoint': 'highest check locked Exact under mean weak-recall drop<=1pp each, per-classFPR rise<=0.5pp, weakAP drop<=0.2pp, tailF1 drop<=0.3pp; epoch0 fallback; same for every arm',
        'pilot_promotion': 'global128 always confirmed. global256/segments124 promoted only if two-seed mean locked check Exact>=anchor+0.1pp, >=global128, and all same safeguards. No report access.',
        'final_selection': 'highest five-seed mean check locked Exact under same safeguards; include unchanged reference; tie retains reference',
        'primary_decision': 'original locked thresholds; no threshold search for primary',
        'auxiliary_decisions': ['target F1 on fresh8064 calibration', 'same calibration FP counts as frozen pair1; test FPR measured independently'],
        'evaluation': {'calibration': 8064, 'check': 2016, 'report': 8064,
                       'ood_10_40dB': 4032, 'new_edge_heldout_pairs': 1152,
                       'shift_audit': '1008 predeclared balanced report identities; cyclic shifts0/1/3/7 samples; no augmentation or selection'},
        'formal_success': 'Exact gain>=0.5pp,>=4/5 positive seeds, paired95 lower>0; each Saw/MFT weak drop<=1pp; each FPR rise<=0.5pp; weakAP drop<=0.2pp; tailF1 drop<=0.3pp',
        'attribution': 'feature candidates compared against BOTH frozen pair1 and same-budget global128; full feature improvement requires both formal comparisons pass',
        'stop': 'No feature passes pilot -> only global128 five-seed confirmation, no automatic backbone changes',
        'limitations': ['Same Python physical-equation simulator, PCG64; no measured data or MATLAB bitwise parity.',
                       'PCA can discard low-variance weak evidence; a negative result rejects this implementation, not all temporal information.',
                       'Frozen transform coefficients differ despite equal trainable head size; cache/compute/memory recorded.',
                       'Five reused model seeds on fixed pools; conditional t intervals, secondary comparisons uncorrected.',
                       'Pilot selection and epoch selection share check; independent report is the final confirmation.',
                       'Power gaps>=20dB absent from fit; OOD is diagnostic, not training coverage.']}
    plan = {'created': now(), 'root': str(ROOT), 'data': ready, 'data_root': str(data),
            'previous_suite': str(previous), 'references': refs, 'seeds': SEEDS, 'slots': SLOTS,
            'data_hashes': hashes, 'source_manifest': source, 'report_output': str(report), 'protocol': protocol,
            'preflight': {'exact_original_forward': True, 'exact_global_statistics': True,
                          'real_IQ_optimizer_step': True, 'initial_logits_unchanged': True,
                          'head_parameters': 4194, 'temporal_shape': list(temporal.shape[1:]),
                          'new_identity_overlap': 0}}
    write_json(suite/'plan.json', plan)
    write_json(suite/'preflight.json', plan['preflight'])
    (report/'plan.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>分段统计验证方案</title>'+STYLE+'<h1>冻结主干的分段统计验证</h1><pre>'+escape(json.dumps(protocol, ensure_ascii=False, indent=2))+'</pre></html>')
    render(suite)


def launch(suite, phase, seeds, methods, report=False):
    jobs = []
    plan = read_json(suite/'plan.json')
    (suite/'logs').mkdir(exist_ok=True)
    for seed in seeds:
        status = suite/'phases'/f'{phase}_seed{seed}.json'
        if status.exists() and read_json(status)['status'] == 'completed':
            continue
        log = (suite/'logs'/f'{phase}_seed{seed}.log').open('a')
        cmd = [sys.executable, '-u', str(Path(__file__).resolve()), '--suite', str(suite),
               '--worker', phase, '--seed', str(seed), '--methods', ','.join(methods)]
        if report:
            cmd.append('--report')
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        jobs.append((seed, proc, log))
    while jobs:
        rows = []
        for seed, proc, _ in jobs:
            code = proc.poll()
            row = {'seed': seed, 'pid': proc.pid, 'state': 'running' if code is None else 'completed' if code == 0 else 'failed'}
            row['training'] = []
            for method in methods:
                path = suite/'models'/f'seed{seed}'/('history_'+method+'.json')
                if path.exists():
                    history = read_json(path)
                    if history:
                        row['training'].append({'method': method, **history[-1]})
            rows.append(row)
        progress = {'updated': now(), 'phase': phase, 'methods': methods, 'jobs': rows,
                    'running': sum(r['state'] == 'running' for r in rows),
                    'completed': sum(r['state'] == 'completed' for r in rows),
                    'failed': sum(r['state'] == 'failed' for r in rows)}
        write_json(suite/'progress.json', progress)
        if not progress['running']:
            break
        time.sleep(4)
    for _, _, log in jobs:
        log.close()
    if jobs and progress['failed']:
        raise RuntimeError('Worker failure; inspect '+phase+' logs')


def select_pilot(suite):
    from longtail.segment_statistics import collect, guard
    seeds = read_json(suite/'plan.json')['seeds'][:2]
    _, means = collect(suite, seeds, 'check', ['anchor_pair1']+METHODS)
    promoted, checks = [], {}
    for method in METHODS[1:]:
        checks[method] = guard(means[method], means['anchor_pair1']) | {
            'exact_gain_at_least0_1pp': means[method]['exact'] >= means['anchor_pair1']['exact']+.001-1e-12,
            'not_below_same_budget_control': means[method]['exact'] >= means['global128']['exact']-1e-12}
        if all(checks[method].values()):
            promoted.append(method)
    result = {'selected_methods': ['global128']+promoted, 'promoted_features': promoted,
              'checks': checks, 'check_means': means, 'seeds': seeds, 'selected': now(),
              'report_access': False}
    write_json(suite/'pilot_selection.json', result)
    print({'event': 'pilot_selection', **result}, flush=True)
    return result


def select_final(suite, methods):
    from longtail.segment_statistics import collect, guard
    seeds = read_json(suite/'plan.json')['seeds']
    _, means = collect(suite, seeds, 'check', ['anchor_pair1']+methods)
    eligibility = {m: guard(means[m], means['anchor_pair1']) for m in ['anchor_pair1']+methods}
    scores = {m: means[m]['exact'] for m in eligibility if all(eligibility[m].values())}
    winner = max(scores, key=lambda m: (scores[m], m == 'anchor_pair1'))
    result = {'selected': winner, 'primary_rule': 'locked', 'methods': methods,
              'eligibility': eligibility, 'check_scores': scores, 'check_means': means,
              'selected_at': now(), 'report_access': False}
    write_json(suite/'selection.json', result)
    files = {str(suite/'selection.json'): sha256(suite/'selection.json'),
             str(suite/'pilot_selection.json'): sha256(suite/'pilot_selection.json'),
             str(suite/'split.json'): sha256(suite/'split.json')}
    for seed in seeds:
        out = suite/'models'/f'seed{seed}'
        files[str(suite/'evaluation'/f'seed{seed}'/'decisions.json')] = sha256(suite/'evaluation'/f'seed{seed}'/'decisions.json')
        for method in methods:
            for suffix in ['.pt', '_transform.npz']:
                path = out/(method+suffix)
                files[str(path)] = sha256(path)
    write_json(suite/'frozen.json', {'frozen_at': now(), 'plan_sha256': sha256(suite/'plan.json'), 'files': files})


def summarize(suite, methods):
    from longtail.segment_statistics import collect, comparisons
    plan = read_json(suite/'plan.json')
    names = ['anchor_pair1', 'frozen_natural']+methods
    rows, means = collect(suite, plan['seeds'], 'development_test', names)
    comp = {m+'_vs_pair1': comparisons(rows, m, 'anchor_pair1') for m in methods}
    comp.update({m+'_vs_global128': comparisons(rows, m, 'global128') for m in methods if m != 'global128'})
    supplemental = {}
    for role in ['ood_wide', 'pair_report', 'shift0', 'shift1', 'shift3', 'shift7']:
        _, supplemental[role] = collect(suite, plan['seeds'], role, names)
    # Auxiliary calibration rules are descriptive; never replace locked primary.
    auxiliary = {}
    for rule in ['target_f1', 'matched_calibration_fpr']:
        auxiliary[rule] = {}
        for m in names:
            selected_rows = [r[m][rule] for r in rows if rule in r[m]]
            if len(selected_rows) == 5:
                from longtail.targeted_validation import mean_metrics
                auxiliary[rule][m] = mean_metrics(selected_rows)
    result = {'seeds': plan['seeds'], 'selection': read_json(suite/'selection.json'),
              'pilot': read_json(suite/'pilot_selection.json'), 'report_means': means,
              'comparisons': comp, 'supplemental': supplemental, 'auxiliary': auxiliary,
              'conditional_intervals': 'paired Student t over5 reused model seeds, fixed identity pools'}
    write_json(suite/'summary.json', result)
    write_json(Path(plan['report_output'])/'results.json', result)
    models = []
    winner = result['selection']['selected']
    for seed in plan['seeds']:
        out = suite/'models'/f'seed{seed}'
        models.append({'seed': seed, 'selected': winner, 'references': plan['references'][str(seed)],
                       'residual_head': {'path': str(out/(winner+'.pt')), 'sha256': sha256(out/(winner+'.pt'))} if winner != 'anchor_pair1' else None,
                       'transform': {'path': str(out/(winner+'_transform.npz')), 'sha256': sha256(out/(winner+'_transform.npz'))} if winner != 'anchor_pair1' else None,
                       'thresholds': plan['references'][str(seed)]['new']['thresholds']})
    write_json(suite/'selected_models.json', {'models': models, 'five_separate_models_not_ensemble': True,
                                            'check_selected_only': True, 'formal_report_acceptance': comp.get(winner+'_vs_pair1', {}).get('passed', False)})


def audit(suite, methods):
    import numpy as np
    from longtail.segment_statistics import cache, predictions, metric, report_gate, NON_TARGETS
    plan = read_json(suite/'plan.json')
    report_gate(suite)
    for path, digest in plan['data_hashes'].items():
        assert sha256(path) == digest, path
    for rec in plan['source_manifest']:
        assert sha256(suite/'source_snapshot'/rec['path']) == rec['sha256']
    conditions, models, steps = 0, 0, 0
    for seed in plan['seeds']:
        ref = plan['references'][str(seed)]
        for source in [ref['new'][k] for k in ['B', 'linear', 'local']]+[ref['FMB'], ref['target_head']]:
            assert sha256(source['path']) == source['sha256']
        output = suite/'evaluation'/f'seed{seed}'
        decision = read_json(output/'decisions.json')
        for role in ['development_test', 'ood_wide', 'pair_report', 'shift0', 'shift1', 'shift3', 'shift7']:
            saved = read_json(output/('metrics_'+role+'.json'))['metrics']
            with np.load(output/('predictions_'+role+'.npz')) as pred:
                records = {k: pred[k] for k in ['targets', 'weak_mask', 'relative_power_db']}
                for name, rules in saved.items():
                    z = pred['logits_'+name]
                    np.testing.assert_array_equal(z[:, NON_TARGETS], pred['logits_anchor_pair1'][:, NON_TARGETS])
                    for rule, old in rules.items():
                        measured = metric(records, z, decision['models'][name]['thresholds'][rule])
                        for key in ['exact', 'macro_f1', 'focus_weak_aupr', 'focus_fpr']:
                            assert abs(measured[key]-old[key]) < 1e-12
                        conditions += 1
        train_files = list((suite/'models'/f'seed{seed}').glob('*_status.json'))
        histories = []
        for path in train_files:
            status = read_json(path)
            assert status['status'] == 'completed' and status['optimizer_steps'] == 6000
            method = path.name.removesuffix('_status.json')
            history = read_json(path.parent/('history_'+method+'.json'))
            assert len(history) == 24 and all(h['natural_views'] == 7500 and h['paired_views'] == 500 for h in history)
            assert np.all(np.sum([h['pair_stratum_counts'] for h in history], axis=0) == 1000)
            histories.append(history)
            models += 1
            steps += 6000
        for history in histories[1:]:
            assert all(a['natural_order_sha256'] == b['natural_order_sha256'] and a['pair_order_sha256'] == b['pair_order_sha256'] for a, b in zip(histories[0], history))
    write_json(suite/'completion_audit.json', {'status': 'passed', 'completed': now(),
        'models_trained': models, 'optimizer_steps': steps, 'metric_conditions_replayed': conditions,
        'data_and_reference_hashes': True, 'snapshot_sources_verified': len(plan['source_manifest']),
        'other_four_logits_exactly_unchanged': True, 'same_batch_streams': True,
        'PCA_fit_role': 'natural fit only', 'report_frozen_barrier': True})


def render(suite):
    plan = read_json(suite/'plan.json')
    html = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>冻结主干的分段统计验证</title>'+STYLE+'<h1>冻结主干：全局压缩与分段统计验证</h1><p><a href="plan.html">冻结方案</a> · <a href="results.json">详细结果JSON</a></p>'
    if (suite/'pipeline_status.json').exists():
        html += '<p>状态：'+escape(read_json(suite/'pipeline_status.json')['status'])+'</p>'
    if (suite/'pilot_selection.json').exists():
        p = read_json(suite/'pilot_selection.json')
        html += '<h2>两种子检查池筛选</h2><p>进入五种子确认：'+escape(', '.join(p['selected_methods']))+'。未使用报告池。</p>'
        html += table(['方法', 'check Exact%', '两目标弱AP%', '两目标FPR%'], [[m]+[f'{100*v[k]:.3f}' for k in ['exact', 'focus_weak_aupr', 'focus_fpr']] for m, v in p['check_means'].items()])
    if (suite/'summary.json').exists():
        r = read_json(suite/'summary.json')
        html += '<h2>独立报告：五种子、固定原阈值</h2><p>check预选：'+escape(r['selection']['selected'])+'；五种子分别评测，非集成。</p>'
        html += table(['方法', 'Exact%', 'MacroF1%', 'Saw弱召回%', 'MFT弱召回%', '两目标弱AP%', '两目标FPR%'],
                      [[m]+[f'{100*v[k]:.3f}' for k in ['exact', 'macro_f1', 'saw_weak_recall', 'mft_weak_recall', 'focus_weak_aupr', 'focus_fpr']] for m, v in r['report_means'].items()])
        for name, comp in r['comparisons'].items():
            d = comp['deltas']['exact']
            html += '<p>'+escape(name)+f'：Exact差{100*d["mean"]:+.3f}pp，条件95%区间[{100*d["paired95"][0]:+.3f},{100*d["paired95"][1]:+.3f}]，{d["positive_seeds"]}/5种子正向；正式门槛'+('通过' if comp['passed'] else '未全部通过')+'。</p>'
        html += '<h2>辅助诊断</h2>'+table(['条件', '方法', 'Exact%', 'Saw弱召回%', 'MFT弱召回%'], [[role, m]+[f'{100*v[k]:.3f}' for k in ['exact', 'saw_weak_recall', 'mft_weak_recall']] for role, values in r['supplemental'].items() for m, v in values.items()])
    html += '<p>仅改变SawFM/MFT两个输出，其他四类保持参考输出与阈值。PCA只用自然训练身份拟合，但可能丢失低方差弱证据。匹配误报只匹配校准池预算，报告误报独立测量。固定池五种子条件区间；筛选/辅助比较未作多重检验校正；同一仿真器，缺少实测验证。</p></html>'
    (Path(plan['report_output'])/'analysis.html').write_text(html)


def run(suite):
    lock = (suite/'coordinator.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (suite/'pipeline_completed.json').exists():
        return
    status_path = suite/'pipeline_status.json'
    previous = read_json(status_path) if status_path.exists() else {}
    started = previous.get('started', now())
    try:
        write_json(status_path, {'status': 'running', 'started': started, 'pid': os.getpid(), 'phase': 'pilot'})
        if not (suite/'pilot_selection.json').exists():
            launch(suite, 'pilot', SEEDS[:2], METHODS)
            pilot = select_pilot(suite)
        else:
            pilot = read_json(suite/'pilot_selection.json')
        render(suite)
        methods = pilot['selected_methods']
        if not (suite/'frozen.json').exists():
            write_json(status_path, {'status': 'running', 'started': started, 'pid': os.getpid(), 'phase': 'confirmation', 'methods': methods})
            launch(suite, 'confirmation', SEEDS, methods)
            select_final(suite, methods)
        render(suite)
        write_json(status_path, {'status': 'running', 'started': started, 'pid': os.getpid(), 'phase': 'independent_report', 'methods': methods})
        launch(suite, 'report', SEEDS, methods, report=True)
        summarize(suite, methods)
        audit(suite, methods)
        complete = {'status': 'completed', 'started': started, 'finished': now(), 'methods': methods,
                    'audit': read_json(suite/'completion_audit.json')}
        write_json(status_path, complete)
        write_json(suite/'pipeline_completed.json', complete)
        render(suite)
    except BaseException as error:
        write_json(status_path, {'status': 'failed', 'started': started, 'failed': now(), 'error': repr(error)})
        render(suite)
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite', required=True, type=Path)
    p.add_argument('--prepare', action='store_true')
    p.add_argument('--data', type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--worker')
    p.add_argument('--seed', type=int)
    p.add_argument('--methods')
    p.add_argument('--report', action='store_true')
    a = p.parse_args()
    suite = a.suite.resolve()
    if a.prepare:
        prepare(suite, a.data.resolve(), a.output.resolve())
    elif a.worker:
        import torch
        from longtail.segment_statistics import worker
        cpus = read_json(suite/'plan.json')['slots'][SEEDS.index(a.seed)]
        os.sched_setaffinity(0, set(cpus))
        os.nice(5)
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        phase = suite/'phases'/f'{a.worker}_seed{a.seed}.json'
        began = now()
        write_json(phase, {'status': 'running', 'started': began, 'pid': os.getpid(), 'cpus': cpus})
        try:
            worker(suite, a.seed, a.methods.split(','), a.report)
            write_json(phase, {'status': 'completed', 'started': began, 'finished': now(), 'pid': os.getpid()})
        except BaseException as error:
            write_json(phase, {'status': 'failed', 'started': began, 'failed': now(), 'error': repr(error)})
            raise
    else:
        run(suite)


if __name__ == '__main__':
    main()
