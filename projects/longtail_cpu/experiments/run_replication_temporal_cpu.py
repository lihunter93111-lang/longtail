"""Run pair replication -> temporal readout -> new-identity backbone retraining."""
import argparse
from datetime import datetime
from html import escape
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
SEEDS = list(range(2050, 2055))
SLOTS = [list(range(8+4*i, 12+4*i)) for i in range(4)] + [list(range(40, 44))]
STAGES = ['replication', 'temporal', 'retrain']
STYLE = '<style>body{font:16px/1.8 system-ui;max-width:1250px;margin:24px auto;padding:24px;color:#234}table{border-collapse:collapse;width:100%;font-size:14px;margin:16px 0}th,td{border:1px solid #ccd;padding:7px}th{background:#e9f2f3}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style>'


def now():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')


def table(headers, rows):
    return '<table><tr>'+''.join('<th>'+escape(str(x))+'</th>' for x in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows)+'</table>'


def prepare(suite, data, report):
    import torch
    import numpy as np
    from longtail.io import read_json, write_json, sha256
    from longtail.data.splits import build_split
    from longtail.data.dataset import IQDataset
    from longtail.engine.checkpoint import load_checkpoint
    from longtail.models.factory import build_model
    from longtail.models.temporal_heads import build_residual, unpooled_features
    from longtail.models.decision_heads import residual_logits
    ready = read_json(data/'generation_complete.json')
    assert ready['status'] == 'completed'
    if suite.exists():
        raise FileExistsError('Use a new suite directory')
    assert set(sum(SLOTS, [])) <= os.sched_getaffinity(0)
    torch.set_num_threads(4)
    previous = ROOT/'runs/staged_boundary_cpu_20261104'
    old = read_json(previous/'plan.json')
    refs = {str(seed): {k: old['references'][str(seed)][k] for k in ['B', 'FMB']} for seed in SEEDS}
    for seed in SEEDS:
        for kind, path in [('linear', previous/'stage2'/f'seed{seed}/head_natural.pt'), ('local', previous/'stage3'/f'seed{seed}/local_pair_bce.pt')]:
            refs[str(seed)][kind] = {'checkpoint': str(path), 'sha256': sha256(path)}
    for sources in refs.values():
        for source in sources.values():
            assert sha256(source['checkpoint']) == source['sha256']
    payload, cfg = load_checkpoint(refs['2050']['B']['checkpoint'])
    manifest = read_json(ready['manifests']['repA'])
    dataset = IQDataset(Path(ready['manifests']['repA']).parent/'train.h5', np.arange(8), cfg['dataloader']['preprocess'])
    x = torch.stack([dataset[i]['signal'] for i in range(8)])
    y = torch.stack([dataset[i]['target'] for i in range(8)])
    dataset.close()
    model = build_model(cfg['model'], num_classes=6)
    model.load_state_dict(payload['model'])
    model.eval().requires_grad_(False)
    with torch.inference_mode():
        feature, temporal = unpooled_features(model, x)
        torch.testing.assert_close(feature, model.forward_features(x), rtol=0, atol=0)
        base = model.classifier(feature).clone()
    # Clone outside inference mode before autograd uses frozen activations.
    feature, temporal, base = feature.clone(), temporal.clone(), base.clone()
    smoke = []
    for kind in ['global32', 'global33', 'temporal']:
        torch.manual_seed(2050)
        head = build_residual(kind)
        source = temporal if kind == 'temporal' else feature
        assert torch.equal(residual_logits(base, head(source)), base)
        opt = torch.optim.AdamW(head.parameters(), lr=.001)
        for _ in range(2):
            opt.zero_grad()
            z = residual_logits(base, head(source))
            loss = torch.nn.functional.binary_cross_entropy_with_logits(z[:, 1:], y[:, 1:])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 5, error_if_nonfinite=True)
            opt.step()
        z = residual_logits(base, head(source))
        assert torch.equal(z[:, [0, 1, 3]], base[:, [0, 1, 3]])
        assert not torch.equal(z[:, [2, 4, 5]], base[:, [2, 4, 5]])
        smoke.append({'architecture': kind, 'parameters': sum(p.numel() for p in head.parameters()), 'loss': loss.item(), 'real_iq_two_optimizer_steps': True})
    suite.mkdir(parents=True)
    report.mkdir(parents=True, exist_ok=True)
    data_hashes = {}
    for tag, path in ready['manifests'].items():
        manifest = read_json(path)
        split = build_split(manifest, Path(path).parent, cfg['data'])
        write_json(suite/'splits'/(tag+'.json'), split)
        data_hashes[path] = sha256(path)
        for info in manifest['files'].values():
            data_hashes[str(Path(path).parent/info['path'])] = info['sha256']
    for path in ready['pair_manifests'].values():
        manifest = read_json(path)
        data_hashes[path] = sha256(path)
        groups = {r: {v['edge_id'] for v in manifest['edges'] if v['role'] == r} for r in ['fit', 'check', 'development_test']}
        assert all(not groups[a] & groups[b] for a, b in [('fit', 'check'), ('fit', 'development_test'), ('check', 'development_test')])
        for info in manifest['files'].values():
            data_hashes[info['path']] = info['sha256']
    for info in ready['shifts'].values():
        data_hashes[info['path']] = info['sha256']
    source_manifest = []
    files = sorted((ROOT/'src').rglob('*.py')) + sorted((ROOT/'experiments').glob('*.py')) + sorted((ROOT/'tests').glob('*.py')) + sorted((ROOT/'configs').glob('*.json')) + [ROOT/'pyproject.toml', ROOT/'uv.lock']
    for file in files:
        dest = suite/'source_snapshot'/file.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(file, dest)
        source_manifest.append({'path': str(file.relative_to(ROOT)), 'sha256': sha256(dest)})
    for seed in SEEDS:
        _, config = load_checkpoint(refs[str(seed)]['B']['checkpoint'])
        config['data']['manifest'] = ready['manifests']['retrain']
        config['run']['output_dir'] = str(suite/'backbones'/f'seed{seed}')
        config['train'].update(seed=seed, device='cpu', cpu_threads=4, sampler='uniform', loss='bce', epochs=24, draws_per_epoch=8000, initialization=None)
        write_json(suite/'configs'/f'backbone_seed{seed}.json', config)
    protocol = {
        'stage_order': STAGES, 'seeds': SEEDS, 'cpu_only': True, 'parallel_jobs': 5, 'physical_cores_per_job': 4,
        'conditional_new_model_count': [50, 55], 'conditional_actual_training_steps': [300000, 330000],
        'replication': {'independent_pair_training_pools': 2, 'new_local_models': 20, 'methods': ['natural global32', '25% paired views global32'],
                        'background_edge_partition': 'same19/6/6 edge split, new identities', 'natural_trajectories': 'same per seed across replications; best checkpoint selected separately on fresh check',
                        'selection': 'pair policy only if both check pools improve Exact vs natural and satisfy safeguards; else natural',
                        'inference': '5 reused backbone seeds; average data replications within seed for conditional interval, never10 independent seeds'},
        'temporal': {'new_local_models': 15, 'heads': {'global32': 4227, 'global33': 4359, 'temporal': 4323},
                     'temporal_input': 'exact float32 pre-pooling temporal map; label-specific attention over contextual positions; no extra positional encoding',
                     'selection': 'best check Exact with safeguards vs global32; global32 fallback',
                     'pair_policy': 'replication check only; if paired, use predeclared repA fit identities'},
        'retrain': {'new_backbones': 5, 'new_linear_heads': 5, 'new_local_heads': '5 or10 conditional on selected architecture',
                    'backbone': 'new32k train identities; same long-tail combination counts and natural BCE; random initialization',
                    'FMB': 'same old fixed expert at0.5 in all compared branches; full retraining applies to the B branch',
                    'comparisons': ['new backbone+linear', 'new backbone+linear+global32', 'new backbone+linear+selected local architecture', 'old candidate on identical reports'],
                    'OOD': 'independent0--20dB and10--40dB component JNR; same frozen main calibration decisions; diagnostic only'},
        'budget': {'epochs': 24, 'steps_per_epoch': 250, 'batch': 32, 'steps_per_model': 6000, 'views_per_model': 192000,
                   'backbone_lr': .0001, 'head_lr': .001, 'optimizer': 'AdamW', 'weight_decay': .0001, 'scheduler': 'cosine24epochs'},
        'decisions': 'per-label F1 fit on calibration only; FMB fixed0.5; matched calibration-FPR secondary diagnostics',
        'local_update_columns': ['SawFM', 'MFT', 'ISRJ'],
        'report_gate': 'freeze model weights, calibration thresholds and check selection before each stage reports; stage reports never drive later choices',
        'success': {'minimum_exact_gain_pp': .5, 'minimum_positive_seeds': 4, 'paired95_lower_positive': True,
                    'max_tail_F1_drop_pp': .3, 'max_class_FPR_rise_pp': 1., 'max_Saw_and_ISRJ_weak_recall_drop_pp': 3.},
        'limits': ['Five model seeds on fixed simulated pools; conditional intervals do not cover all data-generation variation.',
                   'Pair-edge holdout is relative to paired training; natural backbone has seen all63 combinations.',
                   'Same physical-equation Python PCG64 simulator; no MATLAB bitwise parity or real measurement validation.',
                   'Secondary exploratory comparisons have no multiplicity correction.']}
    plan = {'created': now(), 'root': str(ROOT), 'previous_suite': str(previous), 'references': refs, 'seeds': SEEDS, 'slots': SLOTS,
            'data_root': str(data), 'data': ready, 'data_hashes': data_hashes, 'source_manifest': source_manifest,
            'report_output': str(report), 'protocol': protocol,
            'preflight': {'real_iq_shape': list(x.shape), 'prepool_shape': list(temporal.shape), 'heads': smoke, 'forward_features_exactly_preserved': True}}
    write_json(suite/'plan.json', plan)
    write_json(suite/'preflight.json', plan['preflight'])
    html = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>配对复现与时序局部头CPU验证</title>'+STYLE+'<h1>配对复现 → 时序局部头 → 新身份完整重训</h1>'
    html += '<p>五种子、CPU、同预算。先做两份独立配对身份复现；再比较4227参数全局头、4359参数全局头和4323参数时序头；最后在新32k身份池重训B分支，FMB专家固定。各阶段仅用check选方案，冻结后打开report。</p><pre>'+escape(json.dumps(protocol, ensure_ascii=False, indent=2))+'</pre></html>'
    (report/'plan.html').write_text(html)
    render(suite)
    print(json.dumps({'prepared': str(suite), 'preflight': plan['preflight']}, ensure_ascii=False), flush=True)


def worker(suite, stage, seed, report):
    from longtail.io import read_json, write_json
    from longtail.replication_temporal import work
    import torch
    plan = read_json(suite/'plan.json')
    slot = plan['seeds'].index(seed)
    os.sched_setaffinity(0, set(plan['slots'][slot]))
    os.nice(5)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    phase = suite/'phases'/(stage+('_report' if report else '_dev')+f'_seed{seed}.json')
    if phase.exists() and read_json(phase)['status'] == 'completed':
        return
    started = now()
    write_json(phase, {'status': 'running', 'started': started, 'pid': os.getpid(), 'cpus': plan['slots'][slot]})
    try:
        work(suite, stage, seed, report)
        write_json(phase, {'status': 'completed', 'started': started, 'finished': now(), 'pid': os.getpid()})
    except BaseException as error:
        write_json(phase, {'status': 'failed', 'started': started, 'failed': now(), 'error': repr(error)})
        raise


def launch(suite, stage, report=False):
    from longtail.io import read_json, write_json
    plan = read_json(suite/'plan.json')
    tag = stage+('_report' if report else '_dev')
    (suite/'logs').mkdir(exist_ok=True)
    jobs = []
    for seed in plan['seeds']:
        log = (suite/'logs'/f'{tag}_seed{seed}.log').open('a')
        cmd = [sys.executable, '-u', str(Path(__file__).resolve()), '--suite', str(suite), '--worker', stage, '--seed', str(seed)]
        if report:
            cmd.append('--report')
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        jobs.append((seed, proc, log))
    while True:
        rows = []
        for seed, proc, log in jobs:
            code = proc.poll()
            phase = suite/'phases'/f'{tag}_seed{seed}.json'
            item = {'seed': seed, 'pid': proc.pid, 'state': 'running' if code is None else 'completed' if code == 0 else 'failed'}
            if phase.exists():
                item['phase'] = read_json(phase)
            history = []
            for file in (suite/stage/f'seed{seed}').glob('history_*.json'):
                h = read_json(file)
                if h:
                    history.append({'method': file.stem.removeprefix('history_'), **{k: h[-1][k] for k in ['epoch', 'optimizer_steps', 'epoch_seconds', 'check_aupr']}})
            item['training'] = history
            backbone = suite/'backbones'/f'seed{seed}/history.json'
            if stage == 'retrain' and backbone.exists():
                h = read_json(backbone)[-1]
                item['backbone'] = {'epoch': h['epoch'], 'epoch_seconds': h['epoch_seconds'], 'check_aupr': h['validation']['macro_aupr']}
            rows.append(item)
        write_json(suite/'progress.json', {'updated': now(), 'stage': stage, 'role': 'report' if report else 'development',
                   'running_jobs': sum(r['state'] == 'running' for r in rows), 'completed_jobs': sum(r['state'] == 'completed' for r in rows),
                   'failed_jobs': sum(r['state'] == 'failed' for r in rows), 'jobs': rows})
        if not any(r['state'] == 'running' for r in rows):
            break
        time.sleep(4)
    for _, _, log in jobs:
        log.close()
    failures = [r for r in rows if r['state'] == 'failed']
    if failures:
        raise RuntimeError(str(failures))


def render(suite):
    from longtail.io import read_json
    plan = read_json(suite/'plan.json')
    html = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>配对复现与时序局部头结果</title>'+STYLE+'<h1>配对复现与时序局部头CPU验证</h1>'
    html += '<p>各表只比较同一身份池。Exact为整组标签完全正确率；数值为五种子均值。后续方案由check选择，报告用于独立检验。</p><p><a href="plan.html">冻结方案</a></p>'
    if (suite/'progress.json').exists():
        p = read_json(suite/'progress.json')
        html += '<p>更新：'+escape(p['updated'])+'；当前阶段：'+escape(p['stage'])+' / '+escape(p['role'])+'。</p>'
    for stage in STAGES:
        file = suite/stage/'summary.json'
        if not file.exists():
            html += '<h2>'+stage+'</h2><p>等待完成。</p>'
            continue
        summary = read_json(file)
        html += '<h2>'+stage+'</h2><p>check选择：'+escape(summary['selection']['selected'])+'</p>'
        for tag, result in summary['reports'].items():
            rows = []
            for method, m in result['means'].items():
                rows.append([method]+[f'{100*m[k]:.3f}' for k in ['exact', 'macro_f1', 'macro_aupr', 'mean_fpr', 'saw_weak_recall', 'target_weak_recall']])
            html += '<h3>'+tag+'</h3>'+table(['方法', 'Exact%', 'Macro F1%', 'Macro AP%', '平均FPR%', 'Saw弱召回%', '三目标弱召回%'], rows)
            for name, comp in result.get('contrasts', {}).items():
                d = comp['exact']
                html += '<p>'+escape(name)+f'：Exact差 {100*d["mean"]:+.3f}pp，五种子条件95%区间 [{100*d["paired95"][0]:+.3f}, {100*d["paired95"][1]:+.3f}]；'+('通过预设门槛' if comp['passed'] else '未通过全部预设门槛')+'。</p>'
        if stage == 'replication':
            html += '<p>两份数据在同一种子内平均后再统计，避免把相同backbone当成10个独立种子：</p><pre>'+escape(json.dumps(summary['data_replication_averaged_within_seed'], indent=2))+'</pre>'
    html += '<p>限制：相同Python仿真器；FMB旧专家固定；五种子区间只反映条件初始化变化。功率偏移测试不等于真实数据验证；组合边界留出仅相对配对训练。</p></html>'
    (Path(plan['report_output'])/'analysis.html').write_text(html)


def complete_audit(suite):
    from longtail.io import read_json, write_json, sha256
    from longtail.replication_temporal import assert_report_gate, METRICS
    from longtail.staged_training import record_metric
    from longtail.metrics import fit_thresholds, probabilities
    from longtail.decision import fit_fpr_caps
    from longtail.data.dataset import read_metadata
    from longtail.data.power_truth import read_power_truth
    import h5py
    import numpy as np
    plan = read_json(suite/'plan.json')
    for source in plan['source_manifest']:
        assert sha256(suite/'source_snapshot'/source['path']) == source['sha256']
    for file, digest in plan['data_hashes'].items():
        assert sha256(file) == digest
    models, measured, replayed_calibrations = [], 0, 0
    for stage in STAGES:
        assert_report_gate(suite, stage)
        frozen = read_json(suite/stage/'frozen.json')
        for seed in plan['seeds']:
            dev = read_json(suite/'phases'/f'{stage}_dev_seed{seed}.json')
            report = read_json(suite/'phases'/f'{stage}_report_seed{seed}.json')
            assert dev['status'] == report['status'] == 'completed'
            assert dev['finished'] <= frozen['frozen_at'] <= report['started']
            output = suite/stage/f'seed{seed}'
            for decisions_file in output.glob('decisions_*.json'):
                tag = decisions_file.stem.removeprefix('decisions_')
                decisions = read_json(decisions_file)
                cal = dict(np.load(output/('predictions_'+tag+'_calibration.npz')))
                expected = {m: fit_thresholds(cal['targets'], cal['logits_'+m], None) for m in decisions['thresholds']}
                for th in expected.values():
                    th[0] = .5
                assert expected == decisions['thresholds']
                anchor = decisions['matched_fpr_reference']
                caps = ((probabilities(cal['logits_'+anchor]) >= expected[anchor]) & ~cal['targets'].astype(bool)).sum(0)
                for method in expected:
                    th, _ = fit_fpr_caps(cal['targets'], cal['logits_'+method], caps)
                    th[0] = .5
                    assert th == decisions['matched_fpr_thresholds'][method]
                replayed_calibrations += 1
            for status in output.glob('*_status.json'):
                state = read_json(status)
                assert state['status'] == 'completed' and state['optimizer_steps'] == 6000
                name = status.name.removesuffix('_status.json')
                history = read_json(output/('history_'+name+'.json'))
                assert len(history) == 24 and all(h['optimizer_steps'] == (i+1)*250 and h['samples'] == 8000 for i, h in enumerate(history))
                assert sha256(output/(name+'.pt')) == state['checkpoint_sha256']
                models.append({'stage': stage, 'seed': seed, 'method': name, 'steps': 6000})
            for file in output.glob('metrics_*.json'):
                metrics = read_json(file)
                tag, role = metrics['tag'], metrics['role']
                pred = dict(np.load(output/('predictions_'+tag+'_'+role+'.npz')))
                decisions = read_json(output/('decisions_'+tag+'.json'))
                if role == 'pair_report':
                    manifest = read_json(plan['data']['pair_manifests'][tag])
                    data_file = Path(manifest['files']['development_test']['path'])
                    indices = np.arange(manifest['files']['development_test']['samples'])
                elif role.startswith('ood_'):
                    data_file = Path(plan['data']['shifts'][role]['path'])
                    indices = np.arange(plan['data']['shifts'][role]['samples'])
                else:
                    manifest = Path(plan['data']['manifests'][tag])
                    data_file = manifest.parent/'development.h5'
                    indices = np.asarray(read_json(suite/'splits'/(tag+'.json'))['roles'][role])
                yy, ids, _ = read_metadata(data_file)
                truth = read_power_truth(data_file, yy)
                np.testing.assert_array_equal(pred['targets'], yy[indices])
                np.testing.assert_array_equal(pred['identities'], ids[indices])
                np.testing.assert_array_equal(pred['weak_mask'], truth['weak_mask'][indices])
                np.testing.assert_allclose(pred['relative_power_db'], truth['relative_power_db'][indices], rtol=0, atol=0, equal_nan=True)
                for method, stored in metrics['metrics'].items():
                    result = record_metric(pred, pred['logits_'+method], decisions['thresholds'][method])
                    assert all(np.isclose(result[k], stored[k], atol=1e-12, rtol=0) for k in METRICS)
                    assert result['per_label'] == stored['per_label']
                    measured += 1
                    secondary = record_metric(pred, pred['logits_'+method], decisions['matched_fpr_thresholds'][method])
                    assert all(np.isclose(secondary[k], metrics['matched_calibration_fpr_metrics'][method][k], atol=1e-12, rtol=0) for k in METRICS)
                    measured += 1
    for seed in plan['seeds']:
        output = suite/'backbones'/f'seed{seed}'
        assert read_json(output/'status.json')['status'] == 'completed'
        history = read_json(output/'history.json')
        assert len(history) == 24 and all(h['samples_seen'] == 8000 for h in history)
        models.append({'stage': 'retrain', 'seed': seed, 'method': 'backbone', 'steps': 6000})
    result = {'status': 'passed', 'new_models': len(models), 'actual_optimizer_steps': sum(m['steps'] for m in models),
              'replayed_metric_conditions': measured, 'replayed_calibration_bundles': replayed_calibrations,
              'prediction_labels_identities_and_power_truth_verified': True, 'all_stage_report_barriers_passed': True,
              'source_and_data_hashes_verified': True, 'models': models}
    write_json(suite/'completion_audit.json', result)
    write_json(Path(plan['report_output'])/'completion_audit.json', result)
    return result


def export_selection(suite):
    from longtail.io import read_json, write_json, sha256
    from longtail.replication_temporal import source_checkpoint, linear_checkpoint
    plan = read_json(suite/'plan.json')
    selection = read_json(suite/'retrain/selection.json')
    selected = selection['selected']
    models = []
    for seed in plan['seeds']:
        parts = {'B': source_checkpoint(suite, 'retrain', seed), 'FMB': Path(plan['references'][str(seed)]['FMB']['checkpoint']),
                 'linear_head': linear_checkpoint(suite, 'retrain', seed), 'local_head': suite/'retrain'/f'seed{seed}'/(selected+'.pt')}
        models.append({'seed': seed, 'components': {k: {'path': str(p), 'sha256': sha256(p)} for k, p in parts.items()},
                       'thresholds': read_json(suite/'retrain'/f'seed{seed}/decisions_retrain.json')['thresholds'][selected],
                       'local_architecture': selected})
    result = {'status': 'check-selected candidate; consult independent report gates', 'selection': selection,
              'models': models, 'five_seeds_are_separate_models_not_ensemble': True}
    write_json(suite/'selected_models.json', result)
    write_json(Path(plan['report_output'])/'selected_models.json', result)


def run(suite):
    from longtail.io import read_json, write_json
    from longtail.replication_temporal import select, freeze, summarize
    import fcntl
    lock = (suite/'coordinator.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    started = now()
    write_json(suite/'pipeline_status.json', {'status': 'running', 'started': started, 'pid': os.getpid()})
    try:
        for stage in STAGES:
            if (suite/stage/'summary.json').exists():
                continue
            launch(suite, stage)
            select(suite, stage)
            freeze(suite, stage)
            launch(suite, stage, report=True)
            summary = summarize(suite, stage)
            write_json(Path(read_json(suite/'plan.json')['report_output'])/(stage+'_results.json'), summary)
            render(suite)
        export_selection(suite)
        audit = complete_audit(suite)
        write_json(suite/'pipeline_completed.json', {'status': 'completed', 'started': started, 'finished': now(), 'audit': audit})
        write_json(suite/'pipeline_status.json', {'status': 'completed', 'started': started, 'finished': now(), 'pid': os.getpid()})
        render(suite)
    except BaseException as error:
        write_json(suite/'pipeline_status.json', {'status': 'failed', 'started': started, 'failed': now(), 'error': repr(error)})
        render(suite)
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite', type=Path, required=True)
    p.add_argument('--data', type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--prepare', action='store_true')
    p.add_argument('--worker', choices=STAGES)
    p.add_argument('--seed', type=int)
    p.add_argument('--report', action='store_true')
    a = p.parse_args()
    if a.prepare:
        prepare(a.suite.resolve(), a.data.resolve(), a.output.resolve())
    elif a.worker:
        worker(a.suite.resolve(), a.worker, a.seed, a.report)
    else:
        run(a.suite.resolve())
