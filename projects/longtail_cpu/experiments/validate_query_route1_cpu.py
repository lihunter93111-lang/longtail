"""CPU route-1 ablation: pinned parallel jobs, preflight, timing, frozen eval.

Usage: python experiments/validate_query_route1_cpu.py --suite RUN_ROOT --prepare
       python experiments/validate_query_route1_cpu.py --suite RUN_ROOT --benchmark
       python experiments/validate_query_route1_cpu.py --suite RUN_ROOT --run
"""
import argparse
from copy import deepcopy
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
ARMS = {'baseline': 'cpu', 'single': 'single', 'capacity': 'capacity', 'multiscale': 'multiscale'}
SLOTS = [list(range(a, a + 4)) for a in (8, 12, 16, 20, 40, 44, 48, 52)]


def now():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path);path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def prepare(suite):
    from longtail.config import resolve_config
    from longtail.models.factory import build_model
    from longtail.data.manage import verify
    from longtail.data.splits import build_split
    if (suite / 'plan.json').exists():
        raise FileExistsError('The suite plan is already frozen')
    suite.mkdir(parents=True, exist_ok=True)
    base = resolve_config(ROOT / 'configs/query_route1_cpu.json')
    manifest = read(base['data']['manifest'])
    verified = verify(Path(base['data']['manifest']).parent)
    split = build_split(manifest, Path(base['data']['manifest']).parent, base['data'])
    jobs = []
    for arm, suffix in ARMS.items():
        for seed in (2050, 2051):
            config = resolve_config(ROOT / f'configs/query_route1_{suffix}.json')
            config['train']['seed'] = seed
            job_id = f'{arm}_seed{seed}'
            config['run']['output_dir'] = str(suite / job_id)
            config_path = suite / 'configs' / (job_id + '.json')
            write(config_path, config)
            model = build_model(config['model'], num_classes=6)
            jobs.append({'id': job_id, 'arm': arm, 'seed': seed, 'config': str(config_path),
                         'output': config['run']['output_dir'], 'cpus': SLOTS[len(jobs)],
                         'parameters': sum(p.numel() for p in model.parameters())})
    source_manifest = []
    paths = sorted((ROOT / 'src').rglob('*.py')) + sorted((ROOT / 'configs').glob('*.json')) + [Path(__file__), ROOT / 'pyproject.toml', ROOT / 'uv.lock']
    for path in paths:
        rel = path.relative_to(ROOT)
        target = suite / 'source_snapshot' / rel
        target.parent.mkdir(parents=True, exist_ok=True);shutil.copyfile(path, target)
        source_manifest.append({'path': str(rel), 'sha256': digest(path)})
    plan = {'created': now(), 'purpose': 'Test multi-level complex feature evidence against deep-only queries and matched capacity.',
            'jobs': jobs, 'source_manifest': source_manifest,
            'protocol': {'epochs': 8, 'global_batch': 32, 'cpu_threads_per_job': 4,
                         'maximum_parallel_jobs': 8, 'physical_cpu_slots': SLOTS,
                         'device': 'cpu', 'world_size': 1, 'amp': 'none', 'loss': 'bce',
                         'checkpoint_selection': 'check weak_macro_aupr',
                         'threshold_fit': 'calibration per-label F1',
                         'evaluation': ['check', 'development_test', 'gap_0', 'gap_6', 'gap_12', 'gap_18', 'gap_24'],
                         'seeds': [2050, 2051], 'data_split_sha256': split['sha256'],
                         'locked_test_used': False, 'development_test_consumed_previously': True,
                         'hyperparameters_selected_using_development_test': False,
                         'old_gpu_scores_are_not_the_cpu_control': True},
            'dataset_verification': verified,
            'validation': {'pytest': '44 passed in 25.73s',
                           'same_initialized_global_path': True,
                           'all_query_parameters_receive_finite_gradients': True,
                           'odd_length_positions_and_shapes': True,
                           'capacity_difference_parameters': jobs[6]['parameters'] - jobs[4]['parameters']}}
    write(suite / 'plan.json', plan)
    print(json.dumps({'suite': str(suite), 'jobs': len(jobs), 'split': split['sha256'],
                      'parameters': {j['arm']: j['parameters'] for j in jobs}}, ensure_ascii=False), flush=True)


def worker(suite, job_id, action):
    plan = read(suite / 'plan.json')
    job = next(j for j in plan['jobs'] if j['id'] == job_id)
    os.sched_setaffinity(0, set(job['cpus']))
    os.nice(5)
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import numpy as np
    import torch
    from longtail.config import resolve_config
    from longtail.models.factory import build_model
    from longtail.engine.distributed import seed_all
    config = resolve_config(job['config'])
    torch.set_num_threads(config['train']['cpu_threads'])
    torch.set_num_interop_threads(1)
    torch.backends.mkldnn.enabled = True
    seed_all(config['train']['seed'])
    if action == 'benchmark':
        model = build_model(config['model'], num_classes=6)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['train']['lr'], weight_decay=config['train']['weight_decay'])
        x = torch.randn(32, 2, 3000);y = torch.randint(0, 2, (32, 6)).float()
        steps = []
        for index in range(7):
            started = time.perf_counter();optimizer.zero_grad(set_to_none=True)
            logits = model(x);torch.nn.functional.binary_cross_entropy_with_logits(logits, y).backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
            optimizer.step()
            if index >= 2:steps.append(time.perf_counter() - started)
        model.eval();eval_times = []
        with torch.inference_mode():
            for index in range(4):
                xx = torch.randn(64, 2, 3000);started = time.perf_counter();logits = model(xx)
                assert logits.shape == (64, 6) and torch.isfinite(logits).all()
                if index > 0:eval_times.append(time.perf_counter() - started)
        record = {'job': job_id, 'cpus': job['cpus'], 'threads': torch.get_num_threads(),
                  'train_batch_seconds': statistics.median(steps), 'train_steps': steps,
                  'eval_batch64_seconds': statistics.median(eval_times),
                  'training_8epoch_seconds_estimate': statistics.median(steps) * 250 * 8,
                  'parameters': sum(p.numel() for p in model.parameters()),
                  'synthetic_compute_benchmark_only': True, 'recorded': now()}
        write(suite / 'benchmark' / (job_id + '.json'), record)
        print(json.dumps(record), flush=True)
        return
    from longtail.train import run
    from longtail.evaluate import run as evaluate
    from longtail.engine.checkpoint import load_checkpoint
    from longtail.data.loaders import make_dataset, make_loader
    output = Path(job['output']);started = time.time()
    phase_path = suite / 'phases' / (job_id + '.json')
    write(phase_path, {'job': job_id, 'phase': 'training', 'started': now(), 'pid': os.getpid(), 'cpus': job['cpus']})
    status = read(output / 'status.json') if (output / 'status.json').exists() else {}
    if status.get('status') != 'completed':
        resume = output / 'last.pt' if (output / 'last.pt').exists() else None
        run(config, resume=resume)
    checkpoint = output / 'best.pt'
    checkpoint_hash = digest(checkpoint)
    for split, view in [('check', None), ('development_test', None)] + [('development_test', f'gap_{g}') for g in (0, 6, 12, 18, 24)]:
        target = output / 'evaluation' / split
        if view:target = target / view
        write(phase_path, {'job': job_id, 'phase': 'evaluation', 'split': split, 'view': view, 'updated': now(), 'pid': os.getpid(), 'cpus': job['cpus']})
        if (target / 'metrics.json').exists():
            saved = read(target / 'metrics.json')
            if saved['checkpoint_sha256'] != checkpoint_hash:
                raise ValueError('Existing evaluation belongs to another checkpoint')
        else:
            evaluate(checkpoint, split_name=split, output=target, device_name='cpu', view=view)
    if job['arm'] != 'baseline':
        payload, saved = load_checkpoint(checkpoint)
        model = build_model(saved['model'], num_classes=6).eval()
        model.load_state_dict(payload['model'], strict=True)
        manifest = read(output / 'dataset.json')
        dataset = make_dataset(manifest, Path(saved['data']['manifest']).parent,
                               payload['split'], 'check', saved['dataloader'])
        totals = np.zeros((6, len(model.memory_levels)), dtype=np.float64)
        positives = np.zeros(6, dtype=np.int64)
        weak_totals = totals.copy();weak_counts = positives.copy()
        entropy = np.zeros(6, dtype=np.float64);rows = 0
        with torch.inference_mode():
            for batch in make_loader(dataset, saved['dataloader'], batch_size=32):
                _, details = model.forward_diagnostics(batch['signal'])
                weights = details['attention'].numpy()
                offsets = np.cumsum([0] + details['memory_lengths'])
                masses = np.stack([weights[:, :, offsets[i]:offsets[i+1]].sum(-1)
                                   for i in range(len(model.memory_levels))], axis=-1)
                truth = batch['target'].numpy().astype(bool);weak = batch['weak_mask'].numpy()
                totals += (masses * truth[:, :, None]).sum(0);positives += truth.sum(0)
                weak_totals += (masses * weak[:, :, None]).sum(0);weak_counts += weak.sum(0)
                entropy += (-(weights * np.log(np.maximum(weights, 1e-12))).sum(-1) / np.log(weights.shape[-1])).sum(0)
                rows += len(truth)
        dataset.close()
        prior = np.asarray(details['memory_lengths']) / sum(details['memory_lengths'])
        record = {'role': 'check', 'memory_levels': list(model.memory_levels),
                  'memory_lengths': details['memory_lengths'], 'uniform_mass_reference': prior.tolist(),
                  'gate_tanh': details['query_scale'].tolist(),
                  'positive_attention_mass': (totals / np.maximum(positives[:, None], 1)).tolist(),
                  'weak_attention_mass': (weak_totals / np.maximum(weak_counts[:, None], 1)).tolist(),
                  'weak_enrichment_over_uniform': (weak_totals / np.maximum(weak_counts[:, None], 1) / prior).tolist(),
                  'normalized_entropy': (entropy / rows).tolist(),
                  'classes': manifest['classes'], 'checkpoint_sha256': checkpoint_hash,
                  'interpretation': 'Attention mass is diagnostic; token-count-normalized enrichment is not causal proof.'}
        write(output / 'query_diagnostics.json', record)
    if digest(checkpoint) != checkpoint_hash:
        raise ValueError('Evaluation unexpectedly changed the frozen checkpoint')
    write(phase_path, {'job': job_id, 'phase': 'completed', 'finished': now(),
                      'worker_elapsed_seconds': time.time() - started, 'pid': os.getpid(),
                      'cpus': job['cpus'], 'checkpoint_sha256': checkpoint_hash})


def launch(suite, action):
    plan = read(suite / 'plan.json')
    if action == 'train':
        for record in plan['source_manifest']:
            if digest(ROOT / record['path']) != record['sha256']:
                raise ValueError('Source changed after the experiment plan was frozen: ' + record['path'])
    running = []
    logs = suite / ('benchmark_logs' if action == 'benchmark' else 'logs');logs.mkdir(exist_ok=True)
    started = time.time()
    for job in plan['jobs']:
        env = dict(os.environ)
        env.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
                   OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', PYTHONUNBUFFERED='1')
        handle = (logs / (job['id'] + '.log')).open('a')
        process = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()),
            '--suite', str(suite), '--worker', job['id'], '--action', action],
            cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
        running.append((job, process, handle))
    write(suite / ('benchmark_processes.json' if action == 'benchmark' else 'processes.json'),
          {'controller_pid': os.getpid(), 'started': now(), 'jobs': [{'id': j['id'], 'pid': proc.pid, 'cpus': j['cpus']} for j, proc, _ in running]})
    previous = None;last_print = 0
    while any(proc.poll() is None for _, proc, _ in running):
        progress = []
        for job, proc, _ in running:
            row = {'id': job['id'], 'exit_code': proc.poll()}
            if action == 'train':
                history = Path(job['output']) / 'history.json'
                if history.exists():
                    h = read(history);row['epoch'] = h[-1]['epoch'];row['last_epoch_seconds'] = h[-1].get('epoch_seconds')
                phase = suite / 'phases' / (job['id'] + '.json')
                if phase.exists():row['phase'] = read(phase)['phase']
            progress.append(row)
        snapshot = {'action': action, 'elapsed_seconds': round(time.time() - started, 1), 'updated': now(), 'jobs': progress}
        write(suite / ('benchmark_progress.json' if action == 'benchmark' else 'progress.json'), snapshot)
        content = json.dumps(progress, sort_keys=True)
        if content != previous or time.time() - last_print > 45:
            print(json.dumps(snapshot, ensure_ascii=False), flush=True);previous = content;last_print = time.time()
        time.sleep(3)
    failed = []
    for job, proc, handle in running:
        handle.close()
        if proc.returncode != 0:failed.append({'job': job['id'], 'code': proc.returncode, 'log': str(logs / (job['id'] + '.log'))})
    if failed:
        write(suite / (action + '_failures.json'), failed)
        raise RuntimeError(str(failed))
    if action == 'benchmark':
        records = [read(suite / 'benchmark' / (j['id'] + '.json')) for j in plan['jobs']]
        projected = max(r['training_8epoch_seconds_estimate'] + r['eval_batch64_seconds'] *
                        (8 * 16 + 16 + 6 * 63) for r in records)
        summary = {'finished': now(), 'parallel_jobs': len(records), 'physical_cores_requested': 32,
                   'benchmark_wall_seconds': time.time() - started,
                   'estimated_train_and_eval_wall_seconds': projected,
                   'estimated_wall_range_minutes': [round(projected / 60, 1), round(projected * 1.5 / 60 + 2, 1)],
                   'records': records}
        write(suite / 'benchmark_summary.json', summary)
        print(json.dumps(summary, ensure_ascii=False), flush=True)
    else:
        write(suite / 'completion.json', {'finished': now(), 'wall_seconds': time.time() - started,
                                         'jobs_completed': len(running), 'failures': []})
        print(json.dumps(read(suite / 'completion.json')), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', type=Path, required=True)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--worker')
    parser.add_argument('--action', choices=['benchmark', 'train'])
    args = parser.parse_args();suite = args.suite.resolve()
    if args.worker:worker(suite, args.worker, args.action)
    elif args.prepare:prepare(suite)
    elif args.benchmark:launch(suite, 'benchmark')
    elif args.run:launch(suite, 'train')
    else:parser.error('Choose --prepare, --benchmark, --run, or --worker')


if __name__ == '__main__':
    main()
