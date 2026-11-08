"""Frozen 3x5 CPU training, three decisions, and a predeclared hybrid.

Prepare from the current workspace, run the snapshot with its src on PYTHONPATH.
All 15 model/decision hashes are frozen before fresh reporting inference.
"""
import argparse
import csv
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
METHODS = {'bce': 'uniform', 'combo_sampler': 'combo_aware', 'mixed_sampler': 'mixed_combo'}
NAMES = {'bce': '自然采样BCE', 'combo_sampler': '纯组合采样', 'mixed_sampler': '50/50混合采样', 'hybrid': '预定义按类融合'}
CORES = list(range(8, 24)) + list(range(40, 56))
KEYS = ['macro_f1', 'macro_aupr', 'exact', 'tail_f1', 'tail_aupr', 'rare_combo_exact', 'mean_fpr']


def now():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')


def table(headers, rows):
    return '<div style="overflow-x:auto"><table><tr>' + ''.join('<th>' + escape(str(v)) + '</th>' for v in headers) + '</tr>' + ''.join('<tr>' + ''.join('<td>' + escape(str(v)) + '</td>' for v in row) + '</tr>' for row in rows) + '</table></div>'


STYLE = '<style>body{font:16px/1.8 system-ui;max-width:1200px;margin:30px auto;padding:20px;color:#234}table{border-collapse:collapse;width:100%;font-size:14px;margin:18px 0}td,th{border:1px solid #ccd;padding:8px}th{background:#e7f2f3}code{overflow-wrap:anywhere}a{color:#16758a}.note{background:#edf6f6;padding:18px}</style>'


def prepare(suite, report, data_root):
    import numpy as np
    import torch
    from longtail.config import resolve_config
    from longtail.data.dataset import read_metadata
    from longtail.data.loaders import make_dataset
    from longtail.data.manage import verify
    from longtail.data.sampling import EpochWeightedSampler, EpochMixedComboSampler, sampler_epoch_seed
    from longtail.data.splits import build_split
    from longtail.io import read_json, write_json, sha256
    from longtail.models.factory import build_model
    if suite.exists() or (report / 'plan.html').exists():
        raise FileExistsError('Use a new suite and report directory')
    generation = read_json(data_root / 'generation_complete.json')
    assert generation['status'] == 'completed' and generation['new_report_identity_overlap_existing'] == 0
    verification = verify(data_root)
    base = resolve_config(ROOT / 'configs/longtail_v3_fresh_cpu.json')
    base['data']['manifest'] = str(data_root / 'dataset.json')
    base['train']['cpu_threads'] = 4
    manifest = read_json(base['data']['manifest'])
    split = build_split(manifest, data_root, base['data'])
    labels, _, _ = read_metadata(data_root / 'train.h5')
    slots = [CORES[i:i + 4] for i in range(0, len(CORES), 4)]
    assert set(CORES) <= os.sched_getaffinity(0)
    # Same real I/Q contract and real optimizer step, never reporting identities.
    torch.set_num_threads(4)
    fit = make_dataset(manifest, data_root, split, 'fit', base['dataloader'])
    smoke = []
    for method, mode in METHODS.items():
        torch.manual_seed(2050)
        sampler = (EpochMixedComboSampler(labels, 2050) if mode == 'mixed_combo' else
                   EpochWeightedSampler(labels, mode, 2050) if mode != 'uniform' else None)
        if sampler is not None:
            sampler.set_epoch(1); indices = list(sampler)[:32]
        else:
            indices = torch.randperm(len(labels), generator=torch.Generator().manual_seed(2050))[:32].tolist()
        batch = [fit[i] for i in indices]
        x, y = torch.stack([r['signal'] for r in batch]), torch.stack([r['target'] for r in batch])
        model = build_model(base['model'], num_classes=6)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        z = model(x); loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)
        assert z.shape == (32, 6) and torch.isfinite(loss)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=True)
        old_weight = model.classifier[-1].weight.detach().clone()
        optimizer.step()
        assert not torch.equal(old_weight, model.classifier[-1].weight)
        smoke.append({'method': method, 'input_shape': list(x.shape), 'loss': loss.item(), 'gradient_norm': norm.item()})
    fit.close()
    derived = [sampler_epoch_seed(s, e, t) for s in range(2050, 2055) for e in range(1, 25) for t in range(3)]
    assert len(set(derived)) == 360
    suite.mkdir(parents=True); report.mkdir(parents=True, exist_ok=True)
    jobs = []
    # Interleave methods so all three train in the first wave.
    for seed in range(2050, 2055):
        for method, sampler in METHODS.items():
            c = json.loads(json.dumps(base))
            c['train'].update(seed=seed, sampler=sampler)
            job_id = f'{method}_seed{seed}'
            c['run']['output_dir'] = str(suite / job_id)
            cfg = suite / 'configs' / (job_id + '.json')
            write_json(cfg, c)
            slot = len(jobs) % len(slots)
            jobs.append({'id': job_id, 'method': method, 'seed': seed, 'config': str(cfg),
                         'output': c['run']['output_dir'], 'slot': slot, 'cpus': slots[slot]})
    files = sorted((ROOT / 'src').rglob('*.py')) + sorted((ROOT / 'configs').glob('*.json')) + \
            sorted((ROOT / 'experiments').glob('*.py')) + sorted((ROOT / 'tests').glob('*.py')) + [ROOT / 'pyproject.toml', ROOT / 'uv.lock']
    source = []
    for path in files:
        rel = path.relative_to(ROOT); dest = suite / 'source_snapshot' / rel
        dest.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(path, dest)
        source.append({'path': str(rel), 'sha256': sha256(dest)})
    plan = {'created': now(), 'jobs': jobs, 'source_manifest': source, 'report_output': str(report),
        'dataset': str(data_root / 'dataset.json'), 'split_sha256': split['sha256'],
        'train_label_counts': split['fit_label_counts'], 'train_combo_counts': split['fit_combination_counts'],
        'protocol': {'seeds': list(range(2050, 2055)), 'epochs': 24, 'draws_per_epoch': 8000,
            'batch_size': 32, 'optimizer_steps_per_job': 6000, 'total_jobs': 15, 'total_epochs': 360,
            'device': 'cpu', 'amp': 'none', 'world_size': 1, 'threads_per_job': 4, 'maximum_parallel_jobs': 8,
            'slots': slots, 'model': 'DFMS-IQ, 257511 parameters, random initialization',
            'optimizer': 'AdamW', 'lr': 1e-4, 'weight_decay': 1e-4, 'scheduler': '24-epoch cosine',
            'loss': 'ordinary BCE for all methods', 'checkpoint_selection': 'check Macro-AUPR, same for all methods',
            'decision_conditions': ['fixed_0.5', 'f1', 'exact_constrained'],
            'exact_constraints': {'reference': 'calibration F1 decisions', 'maximum_recall_drop': .03,
                'maximum_fpr_rise': .02, 'grid': np.linspace(.2, .8, 13).tolist(), 'include': ['0.5', 'F1 threshold'],
                'starts': ['F1', 'feasible nearest0.5'], 'coordinate_passes': 4, 'ties': 'nearest0.5 in logit distance'},
            'hybrid': {'FMB': 'combo_sampler fixed0.5', 'other_five': 'bce per-label F1', 'paired_by_seed': True},
            'primary_metric': 'fresh reporting Exact; also disclose Macro-F1/AUPR, tail F1 and class recall/FPR',
            'report_gate': 'all 15 trainings and calibration/check evaluations complete; hash every checkpoint and decision before report inference',
            'data_roles': {'fit': 8000, 'calibration': 1008, 'check': 1008, 'new_reporting': 4032},
            'data_changes': 'preserve fit/calibration/check exactly, replace reporting identities only',
            'generator': 'v3 physical-equation Python PCG64 port; independent10--30dB, not bitwise MATLAB parity',
            'tail_labels': ['AMA', 'SinFM'], 'rare_combinations': 'all36 minimum-count ties (41 fit identities each)',
            'rng': 'blake2b(seed,epoch,stream), independent streams; mixed natural/combo/shuffle domains',
            'mixed_sampling': '4000 natural without replacement +4000 combo with replacement, shuffled',
            'no_report_tuning': True, 'no_sampler_loss_stacking': True},
        'preflight': {'pytest_passed': 67, 'real_iq_optimizer_steps': smoke, 'derived_seed_streams_unique': 360,
                      'dataset_verification': verification, 'generation_audit': generation},
        'estimate': {'historical_job_minutes_mean': 49.503, 'historical_job_minutes_max': 50.387,
                     'two_training_waves_minutes': [99.0, 100.8], 'reserved_wall_hours': [2, 3],
                     'assumption': 'previous measured 8x4 CPU throughput; machine competition can change ETA'}}
    write_json(suite / 'plan.json', plan)
    write_json(suite / 'split.json', split)
    write_json(suite / 'preflight.json', plan['preflight'])
    text = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>CPU新一轮验证冻结方案</title>' + STYLE
    text += '<h1>CPU新一轮验证：采样、决策与模型互补</h1><p>自然采样、纯组合采样、50/50混合采样，各5种子，共15模型、360轮、90,000步。统一DFMS-IQ、BCE、24轮cosine。8作业各4个物理核，全部CPU，预计预留2–3小时。</p>'
    text += '<p>训练/calibration/check身份和波形保持不变；4032报告身份全部重新生成并审计无交集。独立10–30dB分量JNR，Python PCG64物理移植，非MATLAB逐点复现。采样采用分域哈希派生随机流，修复相邻种子偏移轮复用。</p>'
    text += '<p>每个模型比较固定0.5、逐类F1、受约束Exact校准。Exact只拟合calibration，逐类召回相对F1最多下降3个百分点、误报最多上升2个百分点；固定网格与双起点坐标搜索，平局偏好0.5。check AUPR选模规则保持一致。</p>'
    text += '<p>按类融合规则提前固定：FMB采用同种子纯组合模型0.5决策，其他五类采用BCE的F1决策；无需额外训练，但部署需要两模型。所有模型和决策哈希冻结后，才启动报告测评；不会根据新报告结果挑阈值或融合类别。</p>'
    text += table(['作业', 'CPU核', '轮数', '每轮抽样'], [[j['id'], str(j['cpus']), 24, 8000] for j in jobs])
    text += '<p>67项测试及三方法真实I/Q优化器步检查通过。主要比较整体Exact，同时披露尾类和逐类退化；五种子区间条件于本固定数据池，不代表真实设备泛化。</p><p><a href="report.html">完成后结果</a></p></html>'
    (report / 'plan.html').write_text(text)
    print(json.dumps({'suite': str(suite), 'jobs': 15, 'report': str(report / 'plan.html'), 'preflight': smoke}, ensure_ascii=False), flush=True)


def decision_metrics(output, role, plan):
    import numpy as np
    from longtail.io import read_json, write_json, sha256
    from longtail.metrics import evaluate_metrics, probabilities
    decisions = read_json(output / 'decisions_v2.json')
    source = output / 'evaluation' / role
    z = dict(np.load(source / 'predictions.npz'))
    thresholds = decisions['thresholds']
    counts = np.array(plan['train_combo_counts'])
    rare = np.flatnonzero(counts == counts.min()) + 1
    combo = z['targets'].astype(int) @ (1 << np.arange(6))
    metrics = {}
    classes = read_json(output / 'split.json')['classes']
    for name, threshold in thresholds.items():
        metric = evaluate_metrics(z['targets'], z['logits'], classes, threshold)
        metric['tail_f1'] = float(np.mean([metric['per_label'][c]['f1'] for c in ['AMA', 'SinFM']]))
        metric['tail_aupr'] = float(np.mean([metric['per_label'][c]['aupr'] for c in ['AMA', 'SinFM']]))
        correct = np.all((probabilities(z['logits']) >= threshold) == z['targets'], axis=1)
        metric['rare_combo_exact'] = float(correct[np.isin(combo, rare)].mean())
        metric['rare_combo_codes'] = rare.tolist()
        metric['mean_fpr'] = float(np.mean([v['negative_fpr'] for v in metric['per_label'].values()]))
        metrics[name] = metric
    write_json(source / 'decision_metrics.json', {'role': role, 'checkpoint_sha256': decisions['checkpoint_sha256'],
        'decisions_sha256': sha256(output / 'decisions_v2.json'), 'metrics': metrics})


def worker(suite, job_id, stage):
    from longtail.io import read_json, write_json, sha256
    plan = read_json(suite / 'plan.json'); job = next(j for j in plan['jobs'] if j['id'] == job_id)
    os.sched_setaffinity(0, set(job['cpus'])); os.nice(5)
    import torch
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    from longtail.train import run
    from longtail.evaluate import run as evaluate
    from longtail.decision import constrained_exact_thresholds
    import numpy as np
    output = Path(job['output']); phase = suite / 'phases' / f'{job_id}_{stage}.json'
    started = now()
    write_json(phase, {'status': 'running', 'started': started, 'pid': os.getpid(), 'cpus': job['cpus']})
    try:
        if stage == 'train':
            status = read_json(output / 'status.json') if (output / 'status.json').exists() else {}
            if status.get('status') != 'completed':
                run(read_json(job['config']), resume=output / 'last.pt' if (output / 'last.pt').exists() else None)
            checkpoint = output / 'best.pt'; digest = sha256(checkpoint)
            for role in ['calibration', 'check']:
                dest = output / 'evaluation' / role
                if not (dest / 'metrics.json').exists():
                    evaluate(checkpoint, split_name=role, output=dest, device_name='cpu')
            if not (output / 'decisions_v2.json').exists():
                cal = dict(np.load(output / 'evaluation/calibration/predictions.npz'))
                fitted, details = constrained_exact_thresholds(cal['targets'], cal['logits'])
                f1 = read_json(output / 'decision.json')['thresholds']
                write_json(output / 'decisions_v2.json', {'checkpoint_sha256': digest, 'fit_role': 'calibration',
                    'split_sha256': plan['split_sha256'], 'calibration_predictions_sha256': sha256(output / 'evaluation/calibration/predictions.npz'),
                    'thresholds': {'fixed_0.5': [.5] * 6, 'f1': f1, 'exact_constrained': fitted}, 'exact_fit': details})
            for role in ['calibration', 'check']:
                decision_metrics(output, role, plan)
        else:
            barrier = read_json(suite / 'frozen_decisions_ready.json')
            assert sha256(suite / 'plan.json') == barrier['plan_sha256']
            frozen = barrier['jobs'][job_id]
            checkpoint = output / 'best.pt'; digest = sha256(checkpoint)
            assert digest == frozen['checkpoint_sha256']
            assert sha256(output / 'decisions_v2.json') == frozen['decisions_sha256']
            dest = output / 'evaluation/development_test'
            if not (dest / 'metrics.json').exists():
                evaluate(checkpoint, split_name='development_test', output=dest, device_name='cpu')
            decision_metrics(output, 'development_test', plan)
        assert digest == sha256(output / 'best.pt')
        write_json(phase, {'status': 'completed', 'started': started, 'finished': now(), 'pid': os.getpid(),
            'checkpoint_sha256': digest, 'decisions_sha256': sha256(output / 'decisions_v2.json')})
    except BaseException as error:
        write_json(phase, {'status': 'failed', 'started': started, 'failed': now(), 'error': repr(error)})
        raise


def launch_stage(suite, stage):
    from longtail.io import read_json, write_json
    plan = read_json(suite / 'plan.json')
    queues = {s: [j for j in plan['jobs'] if j['slot'] == s] for s in range(8)}
    active, finished, failures = {}, set(), []
    logs = suite / 'logs'; logs.mkdir(exist_ok=True)
    def start(slot):
        if not queues[slot]:
            return
        job = queues[slot].pop(0)
        log = (logs / f'{job["id"]}_{stage}.log').open('a')
        process = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()), '--suite', str(suite),
                                    '--worker', job['id'], '--stage', stage], stdout=log, stderr=subprocess.STDOUT)
        active[slot] = (job, process, log)
    for slot in queues:
        start(slot)
    while active:
        for slot, (job, process, log) in list(active.items()):
            code = process.poll()
            if code is not None:
                log.close(); del active[slot]
                if code:
                    failures.append({'job': job['id'], 'stage': stage, 'exit_code': code})
                else:
                    finished.add(job['id'])
                start(slot)
        rows = []
        for job in plan['jobs']:
            row = {'job': job['id'], 'method': job['method'], 'seed': job['seed'], 'cpus': job['cpus']}
            history = Path(job['output']) / 'history.json'
            if history.exists():
                h = read_json(history)[-1]
                row.update(epoch=h['epoch'], last_epoch_seconds=h['epoch_seconds'], check_aupr=h['validation']['macro_aupr'])
            current = next((p for j, p, _ in active.values() if j['id'] == job['id']), None)
            row['state'] = 'completed' if job['id'] in finished else 'failed' if any(f['job'] == job['id'] for f in failures) else 'running' if current else 'queued'
            if current:
                row['pid'] = current.pid
            rows.append(row)
        write_json(suite / 'progress.json', {'stage': stage, 'updated': now(), 'completed_jobs': len(finished),
            'total_jobs': 15, 'completed_epochs': sum(j.get('epoch', 0) for j in rows), 'total_epochs': 360,
            'failures': failures, 'jobs': rows})
        if active:
            time.sleep(5)
    if failures:
        raise RuntimeError(str(failures))


def hybrid(suite, role, seed, plan):
    import numpy as np
    from longtail.io import read_json, write_json, sha256
    from longtail.metrics import evaluate_metrics, probabilities
    base = suite / f'bce_seed{seed}'; combo = suite / f'combo_sampler_seed{seed}'
    b = dict(np.load(base / 'evaluation' / role / 'predictions.npz'))
    c = dict(np.load(combo / 'evaluation' / role / 'predictions.npz'))
    for key in ['targets', 'indices', 'identities']:
        np.testing.assert_array_equal(b[key], c[key])
    logits = b['logits'].copy(); logits[:, 0] = c['logits'][:, 0]
    thresholds = read_json(base / 'decisions_v2.json')['thresholds']['f1'].copy(); thresholds[0] = .5
    split = read_json(suite / 'split.json')
    metric = evaluate_metrics(b['targets'], logits, split['classes'], thresholds)
    metric['tail_f1'] = float(np.mean([metric['per_label'][k]['f1'] for k in ['AMA', 'SinFM']]))
    metric['tail_aupr'] = float(np.mean([metric['per_label'][k]['aupr'] for k in ['AMA', 'SinFM']]))
    counts = np.array(plan['train_combo_counts']); rare = np.flatnonzero(counts == counts.min()) + 1
    code = b['targets'].astype(int) @ (1 << np.arange(6))
    correct = np.all((probabilities(logits) >= thresholds) == b['targets'], axis=1)
    metric['rare_combo_exact'] = float(correct[np.isin(code, rare)].mean())
    metric['mean_fpr'] = float(np.mean([v['negative_fpr'] for v in metric['per_label'].values()]))
    dest = suite / f'hybrid_seed{seed}' / 'evaluation' / role; dest.mkdir(parents=True, exist_ok=True)
    b['logits'] = logits
    np.savez_compressed(dest / 'predictions.npz', **b)
    write_json(dest / 'decision_metrics.json', {'role': role, 'rule': plan['protocol']['hybrid'],
        'source_checkpoints': {'bce': sha256(base / 'best.pt'), 'combo': sha256(combo / 'best.pt')},
        'metrics': {'hybrid_predeclared': metric}})


def summarize(suite):
    import numpy as np
    from scipy.stats import t
    from longtail.io import read_json, write_json, sha256
    plan = read_json(suite / 'plan.json'); report = Path(plan['report_output'])
    frozen = read_json(suite / 'frozen_decisions_ready.json')
    rows, sampling, references = [], [], {}
    for job in plan['jobs']:
        output = Path(job['output']); hold = frozen['jobs'][job['id']]
        assert sha256(output / 'best.pt') == hold['checkpoint_sha256']
        assert sha256(output / 'decisions_v2.json') == hold['decisions_sha256']
        h = read_json(output / 'history.json')
        assert len(h) == 24 and all(r['samples_seen'] == 8000 for r in h)
        for row in h:
            if 'sampling' in row:
                sampling.append({'method': job['method'], 'seed': job['seed'], 'epoch': row['epoch'], **row['sampling']})
    conditions = {'bce': ['fixed_0.5', 'f1', 'exact_constrained'], 'combo_sampler': ['fixed_0.5', 'f1', 'exact_constrained'],
                  'mixed_sampler': ['fixed_0.5', 'f1', 'exact_constrained'], 'hybrid': ['hybrid_predeclared']}
    for role in ['check', 'development_test']:
        for method in conditions:
            for seed in range(2050, 2055):
                output = suite / f'{method}_seed{seed}' / 'evaluation' / role
                result = read_json(output / 'decision_metrics.json')
                z = dict(np.load(output / 'predictions.npz'))
                if role not in references:
                    references[role] = z
                for k in ['targets', 'indices', 'identities']:
                    np.testing.assert_array_equal(z[k], references[role][k])
                for d in conditions[method]:
                    metric = result['metrics'][d]
                    rows.append({'method': method, 'seed': seed, 'role': role, 'decision': d,
                                 **{k: metric['subset_accuracy'] if k == 'exact' else metric[k] for k in KEYS}})
    aggregate = {role: {m: {d: {k: {'mean': float(np.mean([r[k] for r in rows if r['role'] == role and r['method'] == m and r['decision'] == d])),
                                            'std': float(np.std([r[k] for r in rows if r['role'] == role and r['method'] == m and r['decision'] == d], ddof=1))}
                                    for k in KEYS} for d in ds} for m, ds in conditions.items()} for role in ['check', 'development_test']}
    comparisons = [('combo_fixed_vs_bce_f1', ('combo_sampler', 'fixed_0.5'), ('bce', 'f1')),
        ('mixed_fixed_vs_combo_fixed', ('mixed_sampler', 'fixed_0.5'), ('combo_sampler', 'fixed_0.5')),
        ('mixed_f1_vs_combo_f1', ('mixed_sampler', 'f1'), ('combo_sampler', 'f1')),
        ('mixed_exact_vs_combo_exact', ('mixed_sampler', 'exact_constrained'), ('combo_sampler', 'exact_constrained')),
        ('hybrid_vs_bce_f1', ('hybrid', 'hybrid_predeclared'), ('bce', 'f1')),
        ('hybrid_vs_combo_fixed', ('hybrid', 'hybrid_predeclared'), ('combo_sampler', 'fixed_0.5'))]
    comparisons += [(m + '_exact_vs_f1', (m, 'exact_constrained'), (m, 'f1')) for m in METHODS]
    effects = {}
    def values(option, key):
        return np.array([next(r[key] for r in rows if r['role'] == 'development_test' and r['method'] == option[0]
                              and r['decision'] == option[1] and r['seed'] == s) for s in range(2050, 2055)])
    for name, a, b in comparisons:
        effects[name] = {}
        for key in KEYS:
            delta = values(a, key) - values(b, key); radius = t.ppf(.975, 4) * delta.std(ddof=1) / np.sqrt(5)
            effects[name][key] = {'mean': float(delta.mean()), 'paired_seed_95_t_interval': [float(delta.mean() - radius), float(delta.mean() + radius)],
                                  'per_seed': delta.tolist()}
    for method in ['combo_sampler', 'mixed_sampler']:
        observed = [r for r in sampling if r['method'] == method]
        assert len(observed) == 120 and len({r['draw_order_sha256'] for r in observed}) == 120
    result = {'created': now(), 'plan': plan, 'rows': rows, 'aggregate': aggregate, 'effects': effects,
        'sampling_audits': sampling, 'validation': {'all15_jobs24epochs': True, 'distinct_sampling_orders_per_method': 120,
            'frozen_checkpoints_and_decisions': True, 'paired_identities': True, 'report_deferred_until_frozen': True},
        'limitations': 'Fixed PCG64 simulation pool; no real-data or MATLAB parity claim; intervals condition on this pool; no multiplicity correction.'}
    write_json(report / 'results.json', result)
    with (report / 'results.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    text = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>CPU新一轮验证结果</title>' + STYLE
    text += '<h1>CPU新一轮验证结果：15模型、三种决策与冻结融合</h1><p>报告身份4032条，训练/calibration/check保持原样；五种子，同24轮6000步预算。所有权重与阈值在新报告推理前冻结。数值为百分数、五种子均值±样本标准差。</p>'
    for role in ['development_test', 'check']:
        text += '<h2>' + ('新身份报告集' if role == 'development_test' else 'check开发集') + '</h2>'
        text += table(['方法', '决策'] + KEYS, [[NAMES[m], d] + [f"{v[k]['mean']*100:.3f} ± {v[k]['std']*100:.3f}" for k in KEYS]
                      for m, ds in aggregate[role].items() for d, v in ds.items()])
    text += '<h2>预定义配对比较：新身份Exact</h2>'
    text += table(['比较', 'Exact差 pp', '条件95%配对区间 pp', '五种子差 pp'],
        [[name, f"{v['exact']['mean']*100:+.3f}", ' ～ '.join(f'{x*100:+.3f}' for x in v['exact']['paired_seed_95_t_interval']),
          ' / '.join(f'{x*100:+.3f}' for x in v['exact']['per_seed'])] for name, v in effects.items()])
    text += '<p>Exact阈值只优化calibration，召回/FPR约束也只在该角色施加；新报告集上的约束可能不成立，应查看逐类结果。逐类F1、整体Exact和尾类表现同时披露，不据新报告结果再选阈值。</p><p>随机流修复已验证，每种有放回方法120轮抽样顺序均不同。模拟器使用原v3物理移植与PCG64，非MATLAB随机流复现。本轮报告已消费；条件区间不能代替新域/实测泛化。</p>'
    text += '<p><a href="results.csv">逐种子CSV</a> · <a href="results.json">完整统计与逐轮采样审计</a> · <a href="plan.html">冻结方案</a></p></html>'
    (report / 'report.html').write_text(text)


def pipeline(suite):
    from longtail.io import read_json, write_json, sha256
    import longtail
    assert Path(longtail.__file__).resolve().is_relative_to(suite / 'source_snapshot/src')
    plan = read_json(suite / 'plan.json'); start = time.time()
    if (suite / 'pipeline_completed.json').exists():
        raise FileExistsError('This suite already completed; use a new suite for another experiment')
    for entry in plan['source_manifest']:
        assert sha256(suite / 'source_snapshot' / entry['path']) == entry['sha256']
    write_json(suite / 'pipeline.json', {'status': 'running', 'started': now(), 'stage': 'training', 'pid': os.getpid(),
                                      'source': str(Path(__file__).resolve())})
    try:
        launch_stage(suite, 'train')
        held = {}
        for job in plan['jobs']:
            path = Path(job['output'])
            held[job['id']] = {'checkpoint_sha256': sha256(path / 'best.pt'), 'decisions_sha256': sha256(path / 'decisions_v2.json')}
        for seed in range(2050, 2055):
            hybrid(suite, 'check', seed, plan)
        barrier = {'finished': now(), 'plan_sha256': sha256(suite / 'plan.json'), 'jobs': held,
                   'hybrid_rule': plan['protocol']['hybrid']}
        if (suite / 'frozen_decisions_ready.json').exists():
            original = read_json(suite / 'frozen_decisions_ready.json')
            assert original['jobs'] == held and original['plan_sha256'] == barrier['plan_sha256']
            barrier = original
        else:
            write_json(suite / 'frozen_decisions_ready.json', barrier)
            write_json(suite / 'all_training_completed.json', {'finished': barrier['finished'], 'jobs': 15})
        write_json(suite / 'pipeline.json', {'status': 'running', 'stage': 'fresh_report', 'updated': now(), 'pid': os.getpid()})
        launch_stage(suite, 'report')
        for seed in range(2050, 2055):
            hybrid(suite, 'development_test', seed, plan)
        summarize(suite)
        completion = {'status': 'completed', 'finished': now(), 'wall_seconds': time.time() - start,
                      'jobs_completed': 15, 'report': str(Path(plan['report_output']) / 'report.html')}
        write_json(suite / 'pipeline.json', completion)
        write_json(suite / 'pipeline_completed.json', completion)
    except BaseException as error:
        write_json(suite / 'pipeline.json', {'status': 'failed', 'failed': now(), 'error': repr(error), 'pid': os.getpid()})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', required=True, type=Path)
    parser.add_argument('--report', type=Path); parser.add_argument('--data-root', type=Path)
    parser.add_argument('--prepare', action='store_true'); parser.add_argument('--run', action='store_true')
    parser.add_argument('--worker'); parser.add_argument('--stage', choices=['train', 'report'])
    args = parser.parse_args(); suite = args.suite.resolve()
    if args.prepare:
        prepare(suite, args.report.resolve(), args.data_root.resolve())
    elif args.worker:
        worker(suite, args.worker, args.stage)
    elif args.run:
        pipeline(suite)
    else:
        parser.error('Choose --prepare or --run')
