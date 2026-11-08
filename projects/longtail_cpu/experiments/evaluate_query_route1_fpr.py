"""Fit alternative FPR decisions on calibration, report frozen predictions.

This never replaces best.pt/decision.json or selects a model/architecture.
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys


def worker(suite, job_id):
    import numpy as np
    import torch
    from longtail.io import read_json, write_json, sha256
    from longtail.evaluate import run as evaluate
    from longtail.metrics import probabilities, evaluate_metrics
    from longtail.weak_metrics import weak_metrics
    plan = read_json(suite / 'plan.json')
    job = next(j for j in plan['jobs'] if j['id'] == job_id)
    os.sched_setaffinity(0, set(job['cpus']));torch.set_num_threads(4);torch.set_num_interop_threads(1)
    output = Path(job['output']);checkpoint = output / 'best.pt';before = sha256(checkpoint)
    cal_dir = output / 'evaluation/calibration'
    if not (cal_dir / 'metrics.json').exists():
        evaluate(checkpoint, split_name='calibration', output=cal_dir, device_name='cpu')
    calibration = dict(np.load(cal_dir / 'predictions.npz'))
    cal_p = probabilities(calibration['logits']);cal_y = calibration['targets'].astype(bool)
    split = read_json(output / 'split.json');decisions = {};metrics = {}
    for alpha in (.01, .05):
        thresholds = [];fpr = []
        for column in range(6):
            negative = np.sort(cal_p[~cal_y[:, column], column])[::-1]
            allowed = int(np.floor(alpha * len(negative)))
            threshold = min(1., float(np.nextafter(negative[min(allowed, len(negative)-1)], np.inf)))
            thresholds.append(threshold);fpr.append(float((negative >= threshold).mean()))
        name = f'fpr_{alpha:.2f}'
        decisions[name] = {'target_negative_fpr': alpha, 'thresholds': thresholds,
                           'fit_role': 'calibration', 'achieved_calibration_fpr': fpr,
                           'split_sha256': split['sha256'], 'checkpoint_sha256': before,
                           'calibration_identities': calibration['identities'].tolist(),
                           'rule': 'Descending negative quantile; nextafter enforces the >= decision convention; ties may be conservative.'}
        views = {}
        for view, path in [('check', output / 'evaluation/check'),
                           ('development_test', output / 'evaluation/development_test')] + [
                           (f'gap_{g}', output / f'evaluation/development_test/gap_{g}') for g in (0,6,12,18,24)]:
            z = dict(np.load(path / 'predictions.npz'))
            m = evaluate_metrics(z['targets'], z['logits'], split['classes'], thresholds,
                                  combination_counts=split['fit_combination_counts'])
            w = weak_metrics(z['targets'], z['logits'], split['classes'], thresholds,
                             z['weak_mask'], z['relative_power_db'], split['fit_label_counts'])
            views[view] = {'samples': m['samples'], 'exact': m['subset_accuracy'], 'macro_f1': m['macro_f1'],
                           'weak_macro_recall': w['weak_macro_recall'], 'weak_micro_recall': w['weak_micro_recall'],
                           'weak_macro_aupr': w['weak_macro_aupr'], 'tail_weak_recall': w.get('tail_weak_recall'),
                           'per_label': w['per_label'], 'power_gap_bins': w.get('power_gap_bins')}
        metrics[name] = views
    if sha256(checkpoint) != before:raise ValueError('FPR evaluation changed checkpoint')
    write_json(output / 'evaluation/fpr_controlled/decisions.json', decisions)
    write_json(output / 'evaluation/fpr_controlled/metrics.json',
               {'checkpoint_sha256': before, 'threshold_fit_role': 'calibration',
                'locked_test_used': False, 'checkpoint_selection_unchanged': True,
                'warning': 'A calibration FPR target does not guarantee the same FPR on another distribution.',
                'metrics': metrics})
    print(job_id, 'FPR decisions and metrics completed', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', required=True, type=Path);parser.add_argument('--job')
    args = parser.parse_args();suite = args.suite.resolve()
    if args.job:worker(suite, args.job)
    else:
        import json
        plan = json.loads((suite / 'plan.json').read_text());processes = []
        for job in plan['jobs']:
            env = dict(os.environ);env.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1')
            processes.append(subprocess.Popen([sys.executable, str(Path(__file__).resolve()),
                '--suite', str(suite), '--job', job['id']], env=env))
        codes = [p.wait() for p in processes]
        if any(codes):raise RuntimeError(str(codes))


if __name__=='__main__':main()
