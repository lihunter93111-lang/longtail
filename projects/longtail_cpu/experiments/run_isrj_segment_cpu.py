"""CPU ISRJ readout screen -> confirmation -> frozen independent evaluation."""
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

from longtail.io import read_json, write_json, sha256
from longtail.segment_statistics import now
from run_segment_statistics_cpu import STYLE, table

ROOT = Path(__file__).resolve().parents[1]
SEEDS = list(range(2050, 2055))
SLOTS = [list(range(8+4*i, 12+4*i)) for i in range(4)]+[list(range(40, 44))]
METHODS = ['global128', 'segments124']


def prepare(suite, data, report):
    import numpy as np
    import torch
    from longtail import segment_statistics as segment, isrj_segment as experiment
    from longtail.data.splits import build_split
    from longtail.data.dataset import IQDataset
    from longtail.replication_temporal import load_backbone
    from longtail.models.temporal_heads import unpooled_features
    if suite.exists():
        raise FileExistsError('Use a new suite')
    ready = read_json(data/'generation_complete.json')
    assert ready['status'] == 'completed' and ready['identity_overlap_existing'] == 0
    previous = ROOT/'runs/segment_statistics_cpu_20261105'
    old = read_json(previous/'plan.json')
    selected = read_json(previous/'selected_models.json')
    assert selected['formal_report_acceptance']
    refs, segment_refs, transforms, fit_caches, immutable = {}, {}, {}, {}, {}
    for row in selected['models']:
        seed, key = row['seed'], str(row['seed'])
        assert row['selected'] == 'segments124'
        refs[key], segment_refs[key] = row['references'], row
        transforms[key] = {}
        for method in METHODS:
            path = previous/'models'/f'seed{seed}'/(method+'_transform.npz')
            transforms[key][method] = {'path': str(path), 'sha256': sha256(path)}
        cache = previous/'cache'/f'seed{seed}'/'fit.npz'
        binding = cache.with_name('fit_source.json')
        rec = read_json(binding)
        assert sha256(cache) == rec['cache_sha256']
        assert rec['backbone_sha256'] == refs[key]['new']['B']['sha256']
        fit_caches[key] = {'path': str(cache), 'sha256': rec['cache_sha256'], 'binding': str(binding)}
        sources = [refs[key]['new'][k] for k in ['B', 'linear', 'local']]+[refs[key]['FMB'], refs[key]['target_head'], row['residual_head'], row['transform']]+list(transforms[key].values())
        for source in sources:
            assert sha256(source['path']) == source['sha256']
            immutable[source['path']] = source['sha256']
    torch.set_num_threads(4)
    backbone, cfg = load_backbone(Path(refs['2050']['new']['B']['path']))
    with np.load(fit_caches['2050']['path']) as f:
        codes = f['targets'].astype(int) @ (1 << np.arange(6))
        ids = np.array([np.flatnonzero(codes == c)[0] for c in range(1, 64)])
        smoke_rows = np.unique(np.concatenate([ids]+[np.flatnonzero(f['weak_mask'][:, c])[:4] for c in range(6)]))
        smoke = {k: f[k][smoke_rows].copy() for k in ['targets', 'weak_mask', 'relative_power_db', 'indices', 'identities', 'features', 'global128', 'segments124', 'anchor_logits']}
    dataset = IQDataset(data/'train.h5', smoke['indices'][:8], cfg['dataloader']['preprocess'])
    x = torch.stack([dataset[i]['signal'] for i in range(8)])
    dataset.close()
    with torch.inference_mode():
        g, t = unpooled_features(backbone, x)
        torch.testing.assert_close(g, backbone.forward_features(x), rtol=0, atol=0)
        np.testing.assert_allclose(g.numpy(), smoke['features'][:8], rtol=1e-5, atol=2e-6)
        np.testing.assert_allclose(segment.segment_statistics(t).numpy(), smoke['segments124'][:8], rtol=1e-5, atol=2e-6)
    suite.mkdir(parents=True)
    report.mkdir(parents=True, exist_ok=True)
    config = json.loads(json.dumps(cfg['data']))
    config['per_combination'] = {'calibration': 128, 'check': 32, 'development_test': 128}
    manifest = read_json(ready['manifest'])
    split = build_split(manifest, data, config)
    write_json(suite/'split.json', split)
    hashes = {ready['manifest']: sha256(ready['manifest']), str(data/'train.h5'): manifest['files']['train']['sha256'],
              **{v['path']: v['sha256'] for v in ready['files'].values()}}
    training = {'epochs': 24, 'steps_per_epoch': 250, 'batch_size': 32,
        'sampling': 'natural32k, same without-replacement epoch draw streams in both arms; no new pairs',
        'optimizer': 'AdamW', 'learning_rate': .0003, 'weight_decay': .0001, 'scheduler': 'cosine24',
        'loss': 'ordinary BCE on ISRJ only', 'gradient_norm_cap': 5}
    protocol = {'CPU_only': True, 'seeds': SEEDS, 'pilot_seeds': SEEDS[:2], 'max_parallel_jobs': 5, 'physical_cores_per_job': 4,
        'reference': 'frozen check-selected segments124 candidate from previous suite, original thresholds locked',
        'arms': {'reference': 'unchanged, no training', 'global128': 'frozen global128 -> same frozen training-only PCA ->128/32/1 residual',
                 'segments124': 'frozen segment1/2/4 stats1792 -> same frozen training-only PCA ->128/32/1 residual'},
        'parameters_per_new_head': 4161, 'initialization': 'zero output: identical initial logits',
        'only_changed_label': 'ISRJ', 'other_five_logits_and_thresholds': 'exact copy of current candidate including SawFM/MFT',
        'transform_fit': 'reuse previous natural32k-only transforms; verify original fit identity SHA; no refitting on evaluation data',
        'training': training, 'new_models_min_max': [7, 10], 'optimizer_steps_min_max': [42000, 60000],
        'checkpoint': 'highest check locked Exact under ISRJ weak-recall drop<=1pp, FPR rise<=0.5pp, weakAP drop<=0.2pp, tailF1 drop<=0.3pp; epoch0 fallback',
        'pilot_promotion': 'global128 always five-seed. segments124 only if two-seed check Exact >=reference+0.1pp, >=global128, and all safeguards; no report access',
        'final_selection': 'five-seed check locked Exact with safeguards; tie retains reference',
        'primary_decision': 'original locked thresholds', 'auxiliary_decisions': 'ISRJ-only calibration F1 and same calibration FP count; actual report FPR measured',
        'data': {'calibration': 8064, 'check': 2016, 'report': 8064, 'ood10_40dB': 4032,
                 'shift_audit': 'predeclared1008 report identities, cyclic shifts0/1/3/7, no selection'},
        'formal_success': 'Exact>=+0.5pp;>=4/5 positive; paired95 lower>0; ISRJ weak drop<=1pp; classFPR rise<=0.5pp; ISRJ weakAP drop<=0.2pp; tailF1 drop<=0.3pp',
        'attribution': 'require superiority over both frozen reference and same-budget global128 for a feature claim',
        'scope_stop': 'No automatic new pair generation, trainable projection, adapter or backbone fine-tuning',
        'limits': ['Same Python simulator and PCG64, no measured data or MATLAB bitwise parity.',
                   'Five reused model seeds, fixed-pool conditional intervals; secondary comparisons uncorrected.',
                   'Frozen PCA may discard useful low-variance evidence; total frozen transform size differs.',
                   'No fit power gaps above20dB; OOD is diagnostic.']}
    plan = {'created': now(), 'root': str(ROOT), 'previous_suite': str(previous), 'data': ready,
        'data_root': str(data), 'report_output': str(report), 'references': refs, 'segment_references': segment_refs,
        'feature_transforms': transforms, 'fit_caches': fit_caches, 'immutable_sources': immutable,
        'data_hashes': hashes, 'training': training, 'seeds': SEEDS, 'slots': SLOTS, 'protocol': protocol}
    write_json(suite/'plan.json', plan)
    smoke = experiment.attach_reference(suite, 2050, smoke)
    torch.set_num_threads(1)
    for method in METHODS:
        torch.manual_seed(2050)
        head = experiment.ISRJResidual()
        assert sum(p.numel() for p in head.parameters()) == 4161
        features = torch.from_numpy(experiment.features(suite, 2050, method, smoke))
        with torch.inference_mode():
            np.testing.assert_array_equal(experiment.compose(smoke['reference_logits'], head(features).numpy()), smoke['reference_logits'])
        optimizer = torch.optim.AdamW(head.parameters(), lr=.0003)
        optimizer.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(head(features)+torch.from_numpy(smoke['reference_logits'][:, [5]]), torch.from_numpy(smoke['targets'][:, [5]].astype(np.float32)))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 5, error_if_nonfinite=True)
        optimizer.step()
        z = experiment.compose(smoke['reference_logits'], head(features).detach().numpy())
        np.testing.assert_array_equal(z[:, :5], smoke['reference_logits'][:, :5])
        m = experiment.metric(smoke, z, refs['2050']['new']['thresholds'])
        assert np.isfinite([m['exact'], m['isrj_weak_ap'], m['isrj_weak_recall'], loss.item()]).all()
    snapshot = []
    files = sorted((ROOT/'src').rglob('*.py'))+sorted((ROOT/'experiments').glob('*.py'))+sorted((ROOT/'tests').glob('*.py'))+sorted((ROOT/'configs').glob('*.json'))+[ROOT/'pyproject.toml', ROOT/'uv.lock']
    for file in files:
        dest = suite/'source_snapshot'/file.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(file, dest)
        snapshot.append({'path': str(file.relative_to(ROOT)), 'sha256': sha256(dest)})
    plan['source_manifest'] = snapshot
    plan['preflight'] = {'real_IQ_forward_and_cache_replay': True, 'both_heads_real_feature_optimizer_and_metric_steps': True,
        'initial_logits_identical': True, 'other_five_logits_exactly_unchanged': True,
        'parameters_each': 4161, 'new_identity_overlap': 0, 'smoke_metric_role': 'train-only smoke, not performance'}
    write_json(suite/'plan.json', plan)
    write_json(suite/'preflight.json', plan['preflight'])
    (report/'plan.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>ISRJ分段头方案</title>'+STYLE+'<h1>ISRJ全局与分段统计头验证</h1><pre>'+escape(json.dumps(protocol, ensure_ascii=False, indent=2))+'</pre></html>')
    render(suite)


def launch(suite, phase, seeds, methods, report=False):
    jobs = []
    (suite/'logs').mkdir(exist_ok=True)
    for seed in seeds:
        status = suite/'phases'/f'{phase}_seed{seed}.json'
        if status.exists() and read_json(status)['status'] == 'completed':
            continue
        out = (suite/'logs'/f'{phase}_seed{seed}.log').open('a')
        cmd = [sys.executable, '-u', str(Path(__file__).resolve()), '--suite', str(suite), '--worker', phase,
               '--seed', str(seed), '--methods', ','.join(methods)]+(['--report'] if report else [])
        jobs.append((seed, subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT), out))
    while jobs:
        rows = []
        for seed, proc, _ in jobs:
            code = proc.poll()
            row = {'seed': seed, 'pid': proc.pid, 'state': 'running' if code is None else 'completed' if code == 0 else 'failed', 'training': []}
            for method in methods:
                path = suite/'models'/f'seed{seed}'/('history_'+method+'.json')
                if path.exists():
                    h = read_json(path)
                    if h:
                        row['training'].append({'method': method, **h[-1]})
            rows.append(row)
        progress = {'updated': now(), 'phase': phase, 'methods': methods, 'jobs': rows,
            **{state: sum(r['state'] == state for r in rows) for state in ['running', 'completed', 'failed']}}
        write_json(suite/'progress.json', progress)
        if not progress['running']:
            break
        time.sleep(4)
    for _, _, out in jobs:
        out.close()
    if jobs and progress['failed']:
        raise RuntimeError('Worker failure; inspect '+phase+' logs')


def select(suite, pilot=False):
    from longtail.isrj_segment import collect, guard
    seeds = SEEDS[:2] if pilot else SEEDS
    methods = METHODS if pilot else read_json(suite/'pilot_selection.json')['selected_methods']
    _, means = collect(suite, seeds, 'check', ['reference']+methods)
    if pilot:
        checks = guard(means['segments124'], means['reference']) | {
            'gain_at_least0_1pp': means['segments124']['exact'] >= means['reference']['exact']+.001-1e-12,
            'not_below_global128': means['segments124']['exact'] >= means['global128']['exact']-1e-12}
        result = {'selected_methods': ['global128']+(['segments124'] if all(checks.values()) else []),
                  'checks': checks, 'check_means': means, 'selected_at': now(), 'report_access': False}
        write_json(suite/'pilot_selection.json', result)
    else:
        checks = {m: guard(means[m], means['reference']) for m in means}
        scores = {m: means[m]['exact'] for m in means if all(checks[m].values())}
        winner = max(scores, key=lambda m: (scores[m], m == 'reference'))
        result = {'selected': winner, 'methods': methods, 'primary_rule': 'locked', 'check_means': means,
                  'eligibility': checks, 'selected_at': now(), 'report_access': False}
        write_json(suite/'selection.json', result)
        files = dict(read_json(suite/'plan.json')['immutable_sources'])
        for path in [suite/'selection.json', suite/'pilot_selection.json', suite/'split.json']:
            files[str(path)] = sha256(path)
        for seed in seeds:
            paths = [suite/'evaluation'/f'seed{seed}'/'decisions.json']+[suite/'models'/f'seed{seed}'/(m+'.pt') for m in methods]
            files.update({str(p): sha256(p) for p in paths})
        write_json(suite/'frozen.json', {'frozen_at': now(), 'plan_sha256': sha256(suite/'plan.json'), 'files': files})
    print({'event': 'pilot_selection' if pilot else 'final_selection', **result}, flush=True)


def summarize_and_audit(suite):
    import numpy as np
    from longtail.isrj_segment import collect, comparisons, metric
    from longtail.segment_statistics import report_gate
    plan = read_json(suite/'plan.json')
    selection = read_json(suite/'selection.json')
    methods = selection['methods']
    rows, means = collect(suite, SEEDS, 'development_test', ['reference']+methods)
    comp = {m+'_vs_reference': comparisons(rows, m, 'reference') for m in methods}
    if 'segments124' in methods:
        comp['segments124_vs_global128'] = comparisons(rows, 'segments124', 'global128')
    supplemental = {role: collect(suite, SEEDS, role, ['reference']+methods)[1] for role in ['ood_wide', 'shift0', 'shift1', 'shift3', 'shift7']}
    summary = {'selection': selection, 'pilot': read_json(suite/'pilot_selection.json'),
               'report_means': means, 'comparisons': comp, 'supplemental': supplemental,
               'intervals': 'five reused seeds, fixed-pool conditional Student t; secondary comparisons uncorrected'}
    write_json(suite/'summary.json', summary)
    write_json(Path(plan['report_output'])/'results.json', summary)
    report_gate(suite)
    for path, digest in {**plan['data_hashes'], **plan['immutable_sources']}.items():
        assert sha256(path) == digest, path
    for binding in plan['fit_caches'].values():
        assert sha256(binding['path']) == binding['sha256']
    for rec in plan['source_manifest']:
        assert sha256(suite/'source_snapshot'/rec['path']) == rec['sha256']
    conditions = models = steps = 0
    for seed in SEEDS:
        output = suite/'evaluation'/f'seed{seed}'
        decisions = read_json(output/'decisions.json')
        for role in ['development_test', 'ood_wide', 'shift0', 'shift1', 'shift3', 'shift7']:
            metrics = read_json(output/('metrics_'+role+'.json'))['metrics']
            with np.load(output/('predictions_'+role+'.npz')) as rec:
                data = {k: rec[k] for k in ['targets', 'weak_mask', 'relative_power_db']}
                for name, rules in metrics.items():
                    z = rec['logits_'+name]
                    np.testing.assert_array_equal(z[:, :5], rec['logits_reference'][:, :5])
                    for rule, old in rules.items():
                        m = metric(data, z, decisions['models'][name]['thresholds'][rule])
                        assert all(abs(m[k]-old[k]) < 1e-12 for k in ['exact', 'macro_f1', 'isrj_weak_ap', 'isrj_fpr'])
                        conditions += 1
        histories = []
        for path in (suite/'models'/f'seed{seed}').glob('*_status.json'):
            st = read_json(path)
            assert st['status'] == 'completed' and st['optimizer_steps'] == 6000
            histories.append(read_json(path.parent/('history_'+path.name.removesuffix('_status.json')+'.json')))
            assert len(histories[-1]) == 24
            models += 1
            steps += st['optimizer_steps']
        if len(histories) == 2:
            assert all(a['natural_order_sha256'] == b['natural_order_sha256'] for a, b in zip(*histories))
    write_json(suite/'completion_audit.json', {'status': 'passed', 'finished': now(), 'models': models,
        'optimizer_steps': steps, 'metric_conditions_replayed': conditions,
        'original_weights_transforms_data_hashes_verified': True, 'other_five_logits_exactly_unchanged': True,
        'same_draw_streams': True, 'report_frozen_gate': True})
    winner = selection['selected']
    result = []
    for seed in SEEDS:
        head = suite/'models'/f'seed{seed}'/(winner+'.pt')
        result.append({'seed': seed, 'selected': winner, 'current_reference': plan['segment_references'][str(seed)],
            'isrj_head': {'path': str(head), 'sha256': sha256(head)} if winner != 'reference' else None,
            'feature_transform': plan['feature_transforms'][str(seed)].get(winner),
            'thresholds': plan['references'][str(seed)]['new']['thresholds']})
    write_json(suite/'selected_models.json', {'models': result, 'five_separate_models_not_ensemble': True,
        'check_selected_only': True, 'formal_report_acceptance': comp.get(winner+'_vs_reference', {}).get('passed', False)})


def render(suite):
    plan = read_json(suite/'plan.json')
    html = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>ISRJ分段统计头验证</title>'+STYLE+'<h1>固定当前候选：ISRJ全局与分段统计头</h1><p><a href="plan.html">冻结方案</a>；原阈值为主结果，只有ISRJ允许改变，其他五类固定。五种子分别测评，非集成。</p>'
    if (suite/'pipeline_status.json').exists():
        html += '<p>状态：'+escape(read_json(suite/'pipeline_status.json')['status'])+'</p>'
    if (suite/'pilot_selection.json').exists():
        p = read_json(suite/'pilot_selection.json')
        html += '<h2>两种子check筛选</h2><p>进入确认：'+escape(', '.join(p['selected_methods']))+'</p>'
        html += table(['方法', 'Exact%', 'ISRJ弱召回%', 'ISRJ弱AP%', 'ISRJ误报%'], [[m]+[f'{100*v[k]:.3f}' for k in ['exact', 'isrj_weak_recall', 'isrj_weak_ap', 'isrj_fpr']] for m, v in p['check_means'].items()])
    if (suite/'summary.json').exists():
        s = read_json(suite/'summary.json')
        html += '<h2>五种子独立报告，原固定阈值</h2><p>check预选：'+escape(s['selection']['selected'])+'</p>'
        html += table(['方法', 'Exact%', 'MacroF1%', 'ISRJ弱召回%', 'ISRJ弱AP%', 'ISRJ误报%'], [[m]+[f'{100*v[k]:.3f}' for k in ['exact', 'macro_f1', 'isrj_weak_recall', 'isrj_weak_ap', 'isrj_fpr']] for m, v in s['report_means'].items()])
        for name, c in s['comparisons'].items():
            d = c['deltas']['exact']
            html += '<p>'+escape(name)+f'：Exact差{100*d["mean"]:+.3f}pp，条件95%区间[{100*d["paired95"][0]:+.3f},{100*d["paired95"][1]:+.3f}]；正式门槛'+('通过' if c['passed'] else '未全部通过')+'。</p>'
    html += '<p>拟合采用自然32k，仅新增ISRJ头。校准只用新calibration，report在选型、权重和阈值SHA冻结后读取。匹配校准FP数不保证报告误报率相等。同一仿真器及固定池条件区间限制结论；报告消费后不继续调参。</p></html>'
    (Path(plan['report_output'])/'analysis.html').write_text(html)


def run(suite):
    lock = (suite/'coordinator.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (suite/'pipeline_completed.json').exists():
        return
    previous = read_json(suite/'pipeline_status.json') if (suite/'pipeline_status.json').exists() else {}
    started = previous.get('started', now())
    try:
        write_json(suite/'pipeline_status.json', {'status': 'running', 'started': started, 'phase': 'pilot', 'pid': os.getpid()})
        if not (suite/'pilot_selection.json').exists():
            launch(suite, 'pilot', SEEDS[:2], METHODS)
            select(suite, pilot=True)
        methods = read_json(suite/'pilot_selection.json')['selected_methods']
        render(suite)
        if not (suite/'frozen.json').exists():
            write_json(suite/'pipeline_status.json', {'status': 'running', 'started': started, 'phase': 'confirmation', 'methods': methods, 'pid': os.getpid()})
            launch(suite, 'confirmation', SEEDS, methods)
            select(suite)
        write_json(suite/'pipeline_status.json', {'status': 'running', 'started': started, 'phase': 'independent_report', 'methods': methods, 'pid': os.getpid()})
        render(suite)
        launch(suite, 'report', SEEDS, methods, report=True)
        summarize_and_audit(suite)
        result = {'status': 'completed', 'started': started, 'finished': now(), 'methods': methods,
                  'audit': read_json(suite/'completion_audit.json')}
        write_json(suite/'pipeline_status.json', result)
        write_json(suite/'pipeline_completed.json', result)
        render(suite)
    except BaseException as error:
        write_json(suite/'pipeline_status.json', {'status': 'failed', 'started': started, 'failed': now(), 'error': repr(error)})
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
        from longtail.isrj_segment import worker
        cpus = SLOTS[SEEDS.index(a.seed)]
        os.sched_setaffinity(0, set(cpus))
        os.nice(5)
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        phase = suite/'phases'/f'{a.worker}_seed{a.seed}.json'
        started = now()
        write_json(phase, {'status': 'running', 'started': started, 'pid': os.getpid(), 'cpus': cpus})
        try:
            worker(suite, a.seed, a.methods.split(','), a.report)
            write_json(phase, {'status': 'completed', 'started': started, 'finished': now(), 'pid': os.getpid()})
        except BaseException as error:
            write_json(phase, {'status': 'failed', 'started': started, 'failed': now(), 'error': repr(error)})
            raise
    else:
        run(suite)


if __name__ == '__main__':
    main()
