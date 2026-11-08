"""Frozen check-only memory removal diagnostics; never refits thresholds."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import MethodType


def probe(suite, seed):
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    cpus = list(range(48, 52)) if seed == 2050 else list(range(52, 56))
    os.sched_setaffinity(0, set(cpus))
    import numpy as np
    import torch
    from longtail.engine.checkpoint import load_checkpoint
    from longtail.engine.inference import predict
    from longtail.data.loaders import make_dataset, make_loader
    from longtail.models.factory import build_model
    from longtail.io import read_json, write_json, sha256
    from longtail.metrics import evaluate_metrics
    from longtail.weak_metrics import weak_metrics
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    directory = suite / f'multiscale_seed{seed}'
    phase = read_json(suite / 'phases' / f'multiscale_seed{seed}.json')
    if phase['phase'] != 'completed':raise RuntimeError('Wait for the frozen multiscale run')
    checkpoint = directory / 'best.pt';before = sha256(checkpoint)
    payload, config = load_checkpoint(checkpoint)
    model = build_model(config['model'], num_classes=6).eval()
    model.load_state_dict(payload['model'], strict=True)
    manifest = read_json(directory / 'dataset.json')
    dataset = make_dataset(manifest, Path(config['data']['manifest']).parent,
                           payload['split'], 'check', config['dataloader'])
    original = model._query_memory
    outcomes = {};started = time.time()
    for name, keep in [('all', (1, 2, 3)), ('deep_only', (3,)),
                       ('without_stage2', (2, 3)), ('without_stage3', (1, 3))]:
        def selected(self, pyramid, keep=keep):
            memory, positions, lengths = original(pyramid)
            offsets = np.cumsum([0] + lengths)
            slices = [slice(offsets[i], offsets[i+1]) for i, level in enumerate(self.memory_levels) if level in keep]
            return (torch.cat([memory[:, sl] for sl in slices], dim=1),
                    torch.cat([positions[:, sl] for sl in slices], dim=1),
                    [lengths[i] for i, level in enumerate(self.memory_levels) if level in keep])
        model._query_memory = MethodType(selected, model)
        records = predict(model, make_loader(dataset, config['dataloader'], batch_size=64), torch.device('cpu'))
        metric = evaluate_metrics(records['targets'], records['logits'], manifest['classes'],
            thresholds=payload['decision']['thresholds'], combination_counts=payload['split']['fit_combination_counts'])
        w = weak_metrics(records['targets'], records['logits'], manifest['classes'],
            payload['decision']['thresholds'], records['weak_mask'], records['relative_power_db'],
            payload['split']['fit_label_counts'])
        outcomes[name] = {'kept_levels': list(keep), 'exact': metric['subset_accuracy'],
                          'macro_f1': metric['macro_f1'], 'weak_aupr': w['weak_macro_aupr'],
                          'weak_recall': w['weak_macro_recall'], 'per_label': w['per_label']}
        dest = suite / 'diagnostics' / f'memory_seed{seed}_{name}.npz'
        dest.parent.mkdir(exist_ok=True)
        np.savez_compressed(dest, **records)
    dataset.close()
    reference = read_json(directory / 'evaluation/check/metrics.json')['metrics']['calibrated']
    for key, other in [('exact', 'subset_accuracy'), ('macro_f1', 'macro_f1')]:
        if abs(outcomes['all'][key] - reference[other]) > 1e-12:
            raise ValueError('Full-memory probe differs from the frozen check evaluation')
    if sha256(checkpoint) != before:raise ValueError('Probe modified the checkpoint')
    write_json(suite / 'diagnostics' / f'memory_ablation_seed{seed}.json',
        {'seed': seed, 'role': 'check', 'checkpoint_sha256': before,
         'global_path_unchanged': True, 'thresholds_refitted': False,
         'outcomes': outcomes, 'seconds': time.time() - started,
         'interpretation': 'Removing tokens changes the query context; this measures dependence, not a standalone retrained architecture or causal proof.'})
    print(json.dumps({'seed': seed, 'seconds': time.time() - started, 'outcomes': outcomes}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', type=Path, required=True)
    parser.add_argument('--seed', type=int)
    args = parser.parse_args()
    if args.seed is not None:probe(args.suite.resolve(), args.seed)
    else:
        processes = []
        for seed in (2050, 2051):
            env = dict(os.environ);env.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1')
            processes.append(subprocess.Popen([sys.executable, str(Path(__file__).resolve()),
                '--suite', str(args.suite.resolve()), '--seed', str(seed)], env=env))
        codes = [p.wait() for p in processes]
        if any(codes):raise RuntimeError(str(codes))


if __name__=='__main__':main()
