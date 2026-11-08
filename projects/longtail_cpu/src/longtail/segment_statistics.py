"""Frozen, dimension-controlled probes of global and segment statistics."""
from datetime import datetime
import hashlib
from pathlib import Path
import time
from zoneinfo import ZoneInfo

import h5py
import numpy as np
import torch
from torch import nn
from sklearn.utils.extmath import randomized_svd

from .io import read_json, write_json, sha256
from .engine.checkpoint import save_checkpoint
from .data.dataset import IQDataset
from .data.loaders import make_loader
from .models.temporal_heads import unpooled_features
from .replication_temporal import load_backbone, contrast
from .targeted_validation import candidate_logits, SawMFTHead, pair_schedule, metric, mean_metrics, KEYS
from .staged_training import draw_indices, CLASSES
from .calibration_tradeoff import target_f1, TARGETS
from .decision import fit_fpr_caps
from .metrics import probabilities

METHODS = ['global128', 'global256', 'segments124']
NON_TARGETS = [0, 1, 3, 5]


def now():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')


def segment_statistics(x, levels=(1, 2, 4)):
    """Every time position contributes once per level; same std as DFMS."""
    if x.ndim != 3 or min(levels) < 1 or max(levels) > x.shape[-1]:
        raise ValueError('Expected [batch,channels,time] with nonempty bins')
    parts = []
    for count in levels:
        for value in torch.tensor_split(x, count, dim=-1):
            mean = value.mean(-1)
            std = (value.square().mean(-1)-mean.square()).clamp_min(1e-5).sqrt()
            parts.extend([mean, std])
    return torch.cat(parts, dim=1)


def fit_transform(values, fit_ids, seed, dimensions=128):
    """Standardize and fit PCA on natural fit identities only, never check/cal."""
    x = np.asarray(values, np.float32)
    if x.ndim != 2 or min(x.shape) < dimensions or not np.isfinite(x).all():
        raise ValueError('Invalid fit feature matrix')
    mean = x.mean(0, dtype=np.float64).astype(np.float32)
    scale = np.maximum(x.std(0, dtype=np.float64), 1e-5).astype(np.float32)
    centered = np.ascontiguousarray((x-mean)/scale)
    # Center again to remove float32 rounding. Full128 preserves all dimensions.
    center = centered.mean(0, dtype=np.float64).astype(np.float32)
    centered -= center
    _, singular, components = randomized_svd(centered, n_components=dimensions,
                                            n_iter=4, random_state=seed, flip_sign=True)
    fraction = float(np.square(singular.astype(np.float64)).sum()/np.square(centered.astype(np.float64)).sum())
    return {'mean': mean, 'scale': scale, 'center': center, 'components': components.astype(np.float32),
            'fit_id_sha256': hashlib.sha256(np.asarray(fit_ids, '<i8').tobytes()).hexdigest(),
            'fit_samples': len(x), 'input_dimensions': x.shape[1], 'output_dimensions': dimensions,
            'explained_variance_fraction': fraction, 'seed': seed, 'fit_role': 'natural fit only'}


def transform(values, state):
    x = np.asarray(values, np.float32)
    if x.ndim != 2 or x.shape[1] != len(state['mean']):
        raise ValueError('Feature/transform dimension mismatch')
    return np.ascontiguousarray(((x-state['mean'])/state['scale']-state['center']) @ state['components'].T, dtype=np.float32)


class StatisticsResidual(nn.Module):
    def __init__(self, dimensions=128):
        super().__init__()
        self.hidden = nn.Linear(dimensions, 32)
        self.output = nn.Linear(32, 2)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, x):
        return self.output(nn.functional.gelu(self.hidden(x)))


def compose(anchor, residual):
    out = np.asarray(anchor).copy()
    out[:, TARGETS] += residual
    return out


def guard(row, anchor):
    return {'exact_noninferior': row['exact'] >= anchor['exact']-.002-1e-12,
            'saw_weak': row['saw_weak_recall'] >= anchor['saw_weak_recall']-.01-1e-12,
            'mft_weak': row['mft_weak_recall'] >= anchor['mft_weak_recall']-.01-1e-12,
            'class_fpr': all(row['per_label'][c]['negative_fpr'] <= anchor['per_label'][c]['negative_fpr']+.005+1e-12 for c in CLASSES),
            'weak_ap_noninferior': row['focus_weak_aupr'] >= anchor['focus_weak_aupr']-.002-1e-12,
            'tail_f1': row['tail_f1'] >= anchor['tail_f1']-.003-1e-12}


def report_gate(suite):
    frozen = read_json(suite/'frozen.json')
    assert frozen['plan_sha256'] == sha256(suite/'plan.json')
    for path, digest in frozen['files'].items():
        assert sha256(path) == digest, path


def data_source(suite, role):
    plan = read_json(suite/'plan.json')
    data = plan['data']
    if role == 'pair_fit':
        info = data['pair_fit']
        return Path(info['path']), np.arange(info['samples'])
    if role in ['pair_report', 'ood_wide']:
        info = data['files'][role]
        return Path(info['path']), np.arange(info['samples'])
    manifest = Path(data['manifest'])
    split = read_json(suite/'split.json')
    m = read_json(manifest)
    file = manifest.parent/m['files']['train' if role == 'fit' else 'development']['path']
    if role.startswith('shift'):
        with h5py.File(file, 'r') as handle:
            y = np.asarray(handle['label_selected']).T
        codes = y.astype(int) @ (1 << np.arange(6))
        report_rows = np.asarray(split['roles']['development_test'])
        indices = np.sort(np.concatenate([report_rows[codes[report_rows] == c][:16] for c in range(1, 64)]))
    else:
        indices = np.asarray(split['roles'][role])
    return file, indices


def cache(suite, seed, role):
    if role in ['development_test', 'ood_wide', 'pair_report'] or role.startswith('shift'):
        report_gate(suite)
    plan = read_json(suite/'plan.json')
    ref = plan['references'][str(seed)]
    folder = suite/'cache'/f'seed{seed}'
    folder.mkdir(parents=True, exist_ok=True)
    dest, origin = folder/(role+'.npz'), folder/(role+'_source.json')
    if dest.exists():
        binding = read_json(origin)
        assert sha256(dest) == binding['cache_sha256']
        assert binding['backbone_sha256'] == ref['new']['B']['sha256']
        return dict(np.load(dest))
    file, indices = data_source(suite, role)
    assert sha256(file) == plan['data_hashes'][str(file)]
    for kind in ['B', 'linear', 'local']:
        assert sha256(ref['new'][kind]['path']) == ref['new'][kind]['sha256']
    torch.set_num_threads(4)
    backbone, cfg = load_backbone(Path(ref['new']['B']['path']))
    dataset = IQDataset(file, indices, cfg['dataloader']['preprocess'], cache='memory')
    fields, features = {}, {k: [] for k in METHODS}
    shift = int(role.removeprefix('shift')) if role.startswith('shift') else 0
    try:
        with torch.inference_mode():
            for batch in make_loader(dataset, cfg['dataloader'], batch_size=64):
                signal = batch['signal'].roll(shift, dims=-1) if shift else batch['signal']
                g, temporal = unpooled_features(backbone, signal)
                pooled = segment_statistics(temporal)
                features['global128'].append(g.numpy())
                features['global256'].append(pooled[:, :256].numpy())
                features['segments124'].append(pooled.numpy())
                fields.setdefault('base_logits', []).append(backbone.classifier(g).numpy())
                for key in ['target', 'index', 'identity', 'weak_mask', 'relative_power_db']:
                    fields.setdefault(key, []).append(batch[key].numpy())
        records = {k: np.concatenate(v) for k, v in fields.items()}
        records.update({k: np.concatenate(v) for k, v in features.items()})
        records['features'] = records['global128']
        records['targets'] = records.pop('target').astype(np.uint8)
        records['indices'], records['identities'] = records.pop('index'), records.pop('identity')
        if role not in ['fit', 'pair_fit']:
            assert sha256(ref['FMB']['path']) == ref['FMB']['sha256']
            expert, _ = load_backbone(Path(ref['FMB']['path']))
            outputs = []
            with torch.inference_mode():
                for batch in make_loader(dataset, cfg['dataloader'], batch_size=64):
                    signal = batch['signal'].roll(shift, dims=-1) if shift else batch['signal']
                    outputs.append(expert(signal).numpy())
            records['FMB_logits'] = np.concatenate(outputs)
            del expert
    finally:
        dataset.close()
    records['linear_logits'], records['natural_logits'] = candidate_logits(suite, seed, 'new', records)
    assert sha256(ref['target_head']['path']) == ref['target_head']['sha256']
    payload = torch.load(ref['target_head']['path'], map_location='cpu', weights_only=True)
    target_head = SawMFTHead(payload['initial_state'])
    target_head.load_state_dict(payload['state_dict'])
    target_head.eval()
    with torch.inference_mode():
        delta = target_head(torch.from_numpy(records['features'])).numpy()
    records['anchor_logits'] = records['natural_logits'].copy()
    records['anchor_logits'][:, TARGETS] = records['linear_logits'][:, TARGETS]+delta
    if role in ['pair_fit', 'pair_report']:
        with h5py.File(file) as f:
            for key in ['pair_target', 'target_gap_bin', 'pair_id', 'pair_side', 'edge_id']:
                records[key] = f[key][:].ravel()
    assert all(np.isfinite(records[k]).all() for k in METHODS+['anchor_logits'])
    np.savez(dest, **records)
    write_json(origin, {'cache_sha256': sha256(dest), 'backbone_sha256': ref['new']['B']['sha256'],
                        'data_file': str(file), 'data_sha256': sha256(file), 'role': role,
                        'model_mode': 'eval, all original weights frozen', 'roll_samples': shift,
                        'input': 'single received normalized I/Q; no labels, powers, components or pair difference'})
    print({'event': 'cache_completed', 'seed': seed, 'role': role, 'samples': len(indices)}, flush=True)
    return records


def load_transform(path):
    return dict(np.load(path))


def fit_features(suite, seed, method, fit):
    path = suite/'models'/f'seed{seed}'/(method+'_transform.npz')
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return load_transform(path)
    tick = time.time()
    state = fit_transform(fit[method], fit['identities'], seed)
    np.savez(path, **state)
    write_json(path.with_suffix('.json'), {k: v for k, v in state.items() if not isinstance(v, np.ndarray)} | {
        'sha256': sha256(path), 'fit_id_role': 'natural32k only; no pair/calibration/check/report fit',
        'seconds': time.time()-tick, 'frozen_transform_coefficients': int(state['components'].size)})
    return state


def train(suite, seed, method, fit, check, pairs):
    output = suite/'models'/f'seed{seed}'
    output.mkdir(parents=True, exist_ok=True)
    status = output/(method+'_status.json')
    if status.exists() and read_json(status)['status'] == 'completed':
        return
    state = fit_features(suite, seed, method, fit)
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    model = StatisticsResidual()
    x = torch.from_numpy(transform(fit[method], state))
    px = torch.from_numpy(transform(pairs[method], state))
    cv = torch.from_numpy(transform(check[method], state))
    y, py = [torch.from_numpy(r['targets'][:, TARGETS].astype(np.float32)) for r in [fit, pairs]]
    base, pbase = [torch.from_numpy(r['anchor_logits'][:, TARGETS]) for r in [fit, pairs]]
    th = read_json(suite/'plan.json')['references'][str(seed)]['new']['thresholds']
    anchor = metric(check, check['anchor_logits'], th)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0003, weight_decay=.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=24)
    checkpoint, last = output/(method+'.pt'), output/('last_'+method+'.pt')
    history, first, best, best_epoch = [], 1, anchor['exact'], 0

    def payload(epoch):
        return {'kind': 'frozen_statistics_residual', 'method': method, 'seed': seed,
                'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(), 'torch_rng': torch.get_rng_state(),
                'epoch': epoch, 'best_exact': best, 'best_epoch': best_epoch, 'history': history,
                'trainable_parameters': sum(p.numel() for p in model.parameters()),
                'transform_sha256': sha256(output/(method+'_transform.npz')),
                'selection': 'check locked Exact, weak recall/FPR/AP safeguards; epoch0 fallback'}

    if last.exists():
        old = torch.load(last, map_location='cpu', weights_only=True)
        model.load_state_dict(old['state_dict'])
        optimizer.load_state_dict(old['optimizer'])
        scheduler.load_state_dict(old['scheduler'])
        torch.set_rng_state(old['torch_rng'])
        history, first, best, best_epoch = old['history'], old['epoch']+1, old['best_exact'], old['best_epoch']
    else:
        with torch.inference_mode():
            np.testing.assert_array_equal(compose(check['anchor_logits'], model(cv).numpy()), check['anchor_logits'])
        save_checkpoint(checkpoint, payload(0))
    started = time.time()
    for epoch in range(first, 25):
        tick = time.time()
        model.train()
        natural = draw_indices(fit['targets'], seed, epoch, 'natural')
        chosen, strata = pair_schedule(pairs, seed, epoch, 1)
        losses = 0.
        for step in range(250):
            rows = natural[step*30:(step+1)*30]
            pair_rows = np.array([2*chosen[step], 2*chosen[step]+1])
            xx, yy, bb = [torch.cat([a[rows], b[pair_rows]]) for a, b in [(x, px), (y, py), (base, pbase)]]
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.binary_cross_entropy_with_logits(model(xx)+bb, yy)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=True)
            optimizer.step()
            losses += loss.item()
        scheduler.step()
        model.eval()
        with torch.inference_mode():
            z = compose(check['anchor_logits'], model(cv).numpy())
        m = metric(check, z, th)
        checks = guard(m, anchor)
        if all(checks.values()) and m['exact'] > best+1e-12:
            best, best_epoch = m['exact'], epoch
            improved = True
        else:
            improved = False
        row = {'epoch': epoch, 'optimizer_steps': epoch*250, 'loss': losses/250,
               'check_exact': m['exact'], 'check_weak_ap': m['focus_weak_aupr'],
               'check_guard': checks, 'best_epoch': best_epoch,
               'natural_order_sha256': hashlib.sha256(natural[:7500].astype('<i8').tobytes()).hexdigest(),
               'pair_order_sha256': hashlib.sha256(chosen.astype('<i8').tobytes()).hexdigest(),
               'pair_stratum_counts': np.bincount(strata, minlength=6).tolist(),
               'natural_views': 7500, 'paired_views': 500, 'epoch_seconds': time.time()-tick}
        history.append(row)
        save_checkpoint(last, payload(epoch))
        if improved:
            save_checkpoint(checkpoint, payload(epoch))
        write_json(output/('history_'+method+'.json'), history)
        print({'event': 'train_epoch', 'seed': seed, 'method': method, **row}, flush=True)
    write_json(status, {'status': 'completed', 'epochs': 24, 'optimizer_steps': 6000,
                        'seconds': time.time()-started, 'best_epoch': best_epoch,
                        'parameters': 4194, 'checkpoint_sha256': sha256(checkpoint)})


def predictions(suite, seed, methods, records):
    values = {'anchor_pair1': records['anchor_logits'], 'frozen_natural': records['natural_logits']}
    for method in methods:
        output = suite/'models'/f'seed{seed}'
        payload = torch.load(output/(method+'.pt'), map_location='cpu', weights_only=True)
        assert payload['transform_sha256'] == sha256(output/(method+'_transform.npz'))
        model = StatisticsResidual()
        model.load_state_dict(payload['state_dict'])
        model.eval()
        x = torch.from_numpy(transform(records[method], load_transform(output/(method+'_transform.npz'))))
        with torch.inference_mode():
            values[method] = compose(records['anchor_logits'], model(x).numpy())
        np.testing.assert_array_equal(values[method][:, NON_TARGETS], records['anchor_logits'][:, NON_TARGETS])
    return values


def fit_decisions(records, values, locked):
    y = records['targets'].astype(bool)
    counts = ((probabilities(values['anchor_pair1']) >= locked) & ~y).sum(0)
    result = {}
    for name, logits in values.items():
        proposed, meta = fit_fpr_caps(y, logits, counts)
        capped = np.asarray(locked).copy()
        capped[TARGETS] = np.asarray(proposed)[TARGETS]
        result[name] = {'thresholds': {'locked': list(locked),
                                      'target_f1': target_f1(y, logits, locked),
                                      'matched_calibration_fpr': capped.tolist() if all(meta['empirical_caps_satisfied'][c] for c in TARGETS) else None},
                        'matched_fpr_metadata': meta}
    return {'fit_role': 'calibration only', 'primary': 'locked', 'models': result,
            'matched_fpr_scope': 'Same calibration FP counts; report FPR independently measured'}


def measure(suite, seed, methods, report=False):
    output = suite/'evaluation'/f'seed{seed}'
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    if report:
        report_gate(suite)
        decisions = read_json(output/'decisions.json')
        roles = ['development_test', 'ood_wide', 'pair_report', 'shift0', 'shift1', 'shift3', 'shift7']
    else:
        cal = cache(suite, seed, 'calibration')
        locked = read_json(suite/'plan.json')['references'][str(seed)]['new']['thresholds']
        decisions = fit_decisions(cal, predictions(suite, seed, methods, cal), locked)
        write_json(output/'decisions.json', decisions)
        roles = ['check']
    for role in roles:
        records = cache(suite, seed, role)
        values = predictions(suite, seed, methods, records)
        metrics = {name: {rule: metric(records, logits, th) for rule, th in decisions['models'][name]['thresholds'].items() if th is not None}
                   for name, logits in values.items()}
        fields = {k: records[k] for k in ['targets', 'identities', 'indices', 'weak_mask', 'relative_power_db']}
        if role == 'pair_report':
            fields.update({k: records[k] for k in ['pair_id', 'pair_target', 'pair_side', 'edge_id', 'target_gap_bin']})
        np.savez_compressed(output/('predictions_'+role+'.npz'), **fields,
                            **{'logits_'+k: z for k, z in values.items()})
        write_json(output/('metrics_'+role+'.json'), {'seed': seed, 'role': role, 'metrics': metrics})


def worker(suite, seed, methods, report=False):
    if not report:
        fit, check, pairs = [cache(suite, seed, r) for r in ['fit', 'check', 'pair_fit']]
        for method in methods:
            train(suite, seed, method, fit, check, pairs)
    measure(suite, seed, methods, report)


def collect(suite, seeds, role, methods):
    rows = [read_json(suite/'evaluation'/f'seed{s}'/('metrics_'+role+'.json'))['metrics'] for s in seeds]
    means = {m: mean_metrics([r[m]['locked'] for r in rows]) for m in methods}
    return rows, means


def comparisons(rows, candidate, anchor):
    a, b = [r[candidate]['locked'] for r in rows], [r[anchor]['locked'] for r in rows]
    deltas = {key: contrast([v[key] for v in a], [v[key] for v in b]) for key in KEYS}
    ma, mb = mean_metrics(a), mean_metrics(b)
    d = deltas['exact']
    checks = guard(ma, mb)
    formal = {'exact_gain_0_5pp': d['mean'] >= .005,
              'at_least4_of5_positive': d['positive_seeds'] >= 4,
              'conditional95_lower_positive': d['paired95'][0] > 0,
              **checks}
    fpr = {c: contrast([v['per_label'][c]['negative_fpr'] for v in a],
                       [v['per_label'][c]['negative_fpr'] for v in b]) for c in CLASSES}
    return {'candidate': candidate, 'anchor': anchor, 'deltas': deltas,
            'per_label_fpr_deltas': fpr, 'formal_checks': formal, 'passed': all(formal.values())}
