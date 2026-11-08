"""Isolated ISRJ residual readouts on the immutable segment-statistics candidate."""
import hashlib
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from . import segment_statistics as segment
from .io import read_json, write_json, sha256
from .engine.checkpoint import save_checkpoint
from .staged_training import record_metric, draw_indices, CLASSES
from .metrics import probabilities, fit_thresholds
from .decision import fit_fpr_caps
from .replication_temporal import contrast

METHODS = ['global128', 'segments124']
TARGET = 5
NON_TARGETS = [0, 1, 2, 3, 4]
KEYS = ['exact', 'macro_f1', 'macro_aupr', 'tail_f1', 'mean_fpr',
        'isrj_weak_recall', 'isrj_weak_ap', 'isrj_fpr', 'saw_weak_recall', 'mft_weak_recall']


class ISRJResidual(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(128, 32)
        self.output = nn.Linear(32, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, x):
        return self.output(nn.functional.gelu(self.hidden(x)))


def compose(reference, delta):
    result = np.asarray(reference).copy()
    value = np.asarray(delta)
    if value.shape != (len(result), 1):
        raise ValueError('Expected one ISRJ residual per sample')
    result[:, TARGET] += value[:, 0]
    return result


def attach_reference(suite, seed, records):
    plan = read_json(suite/'plan.json')
    binding = plan['segment_references'][str(seed)]
    for name in ['residual_head', 'transform']:
        assert sha256(binding[name]['path']) == binding[name]['sha256']
    model = segment.StatisticsResidual()
    payload = torch.load(binding['residual_head']['path'], map_location='cpu', weights_only=True)
    model.load_state_dict(payload['state_dict'])
    model.eval().requires_grad_(False)
    state = segment.load_transform(binding['transform']['path'])
    x = torch.from_numpy(segment.transform(records['segments124'], state))
    with torch.inference_mode():
        records['reference_logits'] = segment.compose(records['anchor_logits'], model(x).numpy())
    np.testing.assert_array_equal(records['reference_logits'][:, [0, 1, 3, 5]], records['anchor_logits'][:, [0, 1, 3, 5]])
    return records


def cache(suite, seed, role):
    if role == 'fit':
        plan = read_json(suite/'plan.json')
        source = plan['fit_caches'][str(seed)]
        assert sha256(source['path']) == source['sha256']
        binding = read_json(source['binding'])
        assert binding['backbone_sha256'] == plan['references'][str(seed)]['new']['B']['sha256']
        train_path = str(Path(plan['data']['manifest']).parent/'train.h5')
        assert binding['data_sha256'] == plan['data_hashes'][train_path]
        records = dict(np.load(source['path']))
        for method in METHODS:
            spec = plan['feature_transforms'][str(seed)][method]
            assert sha256(spec['path']) == spec['sha256']
            state = segment.load_transform(spec['path'])
            digest = hashlib.sha256(np.asarray(records['identities'], '<i8').tobytes()).hexdigest()
            assert str(state['fit_id_sha256']) == digest and int(state['fit_samples']) == len(records['targets'])
        print({'event': 'fit_cache_reused', 'seed': seed, 'samples': len(records['targets'])}, flush=True)
    else:
        # Existing extractor enforces the report gate before even reading caches.
        records = segment.cache(suite, seed, role)
    return attach_reference(suite, seed, records)


def features(suite, seed, method, records):
    spec = read_json(suite/'plan.json')['feature_transforms'][str(seed)][method]
    assert sha256(spec['path']) == spec['sha256']
    return segment.transform(records[method], segment.load_transform(spec['path']))


def metric(records, logits, thresholds):
    result = record_metric(records, logits, thresholds)
    weak = result['weak_component']['per_label']
    result.update(isrj_weak_recall=weak['ISRJ']['weak_recall'], isrj_weak_ap=weak['ISRJ']['weak_aupr'],
                  isrj_fpr=weak['ISRJ']['negative_fpr'], mft_weak_recall=weak['MFT']['weak_recall'])
    return result


def mean_metrics(rows):
    out = {key: float(np.mean([r[key] for r in rows])) for key in KEYS}
    out['per_label'] = {c: {'negative_fpr': float(np.mean([r['per_label'][c]['negative_fpr'] for r in rows]))} for c in CLASSES}
    return out


def guard(row, anchor):
    return {'exact_noninferior': row['exact'] >= anchor['exact']-.002-1e-12,
            'isrj_weak_recall': row['isrj_weak_recall'] >= anchor['isrj_weak_recall']-.01-1e-12,
            'isrj_fpr': row['isrj_fpr'] <= anchor['isrj_fpr']+.005+1e-12,
            'isrj_weak_ap': row['isrj_weak_ap'] >= anchor['isrj_weak_ap']-.002-1e-12,
            'tail_f1': row['tail_f1'] >= anchor['tail_f1']-.003-1e-12,
            'class_fpr': all(row['per_label'][c]['negative_fpr'] <= anchor['per_label'][c]['negative_fpr']+.005+1e-12 for c in CLASSES)}


def fit_decisions(records, values, locked):
    y = records['targets'].astype(bool)
    counts = ((probabilities(values['reference']) >= locked) & ~y).sum(0)
    result = {}
    for name, logits in values.items():
        f1 = np.array(locked, float).copy()
        f1[TARGET] = fit_thresholds(y[:, [TARGET]], logits[:, [TARGET]], None)[0]
        proposed, meta = fit_fpr_caps(y, logits, counts)
        capped = np.array(locked, float).copy()
        capped[TARGET] = proposed[TARGET]
        result[name] = {'thresholds': {'locked': list(locked), 'isrj_f1': f1.tolist(),
            'matched_calibration_fpr': capped.tolist() if meta['empirical_caps_satisfied'][TARGET] else None},
            'matched_fpr_metadata': meta}
    return {'fit_role': 'calibration only', 'changed_threshold': TARGET, 'primary': 'locked', 'models': result,
            'matched_fpr_scope': 'same calibration FP budget; report FPR independently measured'}


def train(suite, seed, method, fit, check):
    output = suite/'models'/f'seed{seed}'
    output.mkdir(parents=True, exist_ok=True)
    status, checkpoint, last = output/(method+'_status.json'), output/(method+'.pt'), output/('last_'+method+'.pt')
    if status.exists() and read_json(status)['status'] == 'completed':
        return
    plan = read_json(suite/'plan.json')
    epochs, steps = plan['training']['epochs'], plan['training']['steps_per_epoch']
    x, cv = [torch.from_numpy(features(suite, seed, method, records)) for records in [fit, check]]
    y = torch.from_numpy(fit['targets'][:, [TARGET]].astype(np.float32))
    base = torch.from_numpy(fit['reference_logits'][:, [TARGET]])
    th = plan['references'][str(seed)]['new']['thresholds']
    anchor = metric(check, check['reference_logits'], th)
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    model = ISRJResidual()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0003, weight_decay=.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    history, first, best, best_epoch = [], 1, anchor['exact'], 0

    def payload(epoch):
        return {'kind': 'isrj_frozen_readout', 'seed': seed, 'method': method,
            'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(), 'torch_rng': torch.get_rng_state(),
            'history': history, 'epoch': epoch, 'best_epoch': best_epoch, 'best_exact': best,
            'parameters': 4161, 'transform_sha256': plan['feature_transforms'][str(seed)][method]['sha256'],
            'changed_logit_columns': [TARGET], 'plan_sha256': sha256(suite/'plan.json')}

    if last.exists():
        old = torch.load(last, map_location='cpu', weights_only=True)
        assert old['plan_sha256'] == sha256(suite/'plan.json')
        model.load_state_dict(old['state_dict'])
        optimizer.load_state_dict(old['optimizer'])
        scheduler.load_state_dict(old['scheduler'])
        torch.set_rng_state(old['torch_rng'])
        first, best, best_epoch, history = old['epoch']+1, old['best_exact'], old['best_epoch'], old['history']
    else:
        with torch.inference_mode():
            np.testing.assert_array_equal(compose(check['reference_logits'], model(cv).numpy()), check['reference_logits'])
        save_checkpoint(checkpoint, payload(0))
    started = time.time()
    for epoch in range(first, epochs+1):
        tick = time.time()
        order = draw_indices(fit['targets'], seed, epoch, 'natural', draws=32*steps)
        model.train()
        loss_sum = 0.
        for step in range(steps):
            rows = order[32*step:32*(step+1)]
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.binary_cross_entropy_with_logits(model(x[rows])+base[rows], y[rows])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=True)
            optimizer.step()
            loss_sum += loss.item()
        scheduler.step()
        model.eval()
        with torch.inference_mode():
            z = compose(check['reference_logits'], model(cv).numpy())
        measured = metric(check, z, th)
        checks = guard(measured, anchor)
        improved = all(checks.values()) and measured['exact'] > best+1e-12
        if improved:
            best, best_epoch = measured['exact'], epoch
        row = {'epoch': epoch, 'optimizer_steps': epoch*steps, 'loss': loss_sum/steps,
            'check_exact': measured['exact'], 'check_isrj_weak_recall': measured['isrj_weak_recall'],
            'check_isrj_weak_ap': measured['isrj_weak_ap'], 'check_isrj_fpr': measured['isrj_fpr'],
            'check_guard': checks, 'best_epoch': best_epoch, 'natural_views': 32*steps,
            'natural_order_sha256': hashlib.sha256(order.astype('<i8').tobytes()).hexdigest(),
            'epoch_seconds': time.time()-tick}
        history.append(row)
        save_checkpoint(last, payload(epoch))
        if improved:
            save_checkpoint(checkpoint, payload(epoch))
        write_json(output/('history_'+method+'.json'), history)
        print({'event': 'train_epoch', 'seed': seed, 'method': method, **row}, flush=True)
    write_json(status, {'status': 'completed', 'epochs': epochs, 'optimizer_steps': epochs*steps,
        'parameters': 4161, 'best_epoch': best_epoch, 'seconds': time.time()-started,
        'checkpoint_sha256': sha256(checkpoint)})


def predictions(suite, seed, methods, records):
    values = {'reference': records['reference_logits']}
    for method in methods:
        payload = torch.load(suite/'models'/f'seed{seed}'/(method+'.pt'), map_location='cpu', weights_only=True)
        assert payload['plan_sha256'] == sha256(suite/'plan.json')
        spec = read_json(suite/'plan.json')['feature_transforms'][str(seed)][method]
        assert payload['transform_sha256'] == spec['sha256']
        model = ISRJResidual()
        model.load_state_dict(payload['state_dict'])
        model.eval()
        xx = torch.from_numpy(features(suite, seed, method, records))
        with torch.inference_mode():
            values[method] = compose(records['reference_logits'], model(xx).numpy())
        np.testing.assert_array_equal(values[method][:, NON_TARGETS], values['reference'][:, NON_TARGETS])
    return values


def measure(suite, seed, methods, report=False):
    output = suite/'evaluation'/f'seed{seed}'
    output.mkdir(parents=True, exist_ok=True)
    if report:
        segment.report_gate(suite)
        decisions = read_json(output/'decisions.json')
        roles = ['development_test', 'ood_wide', 'shift0', 'shift1', 'shift3', 'shift7']
    else:
        records = cache(suite, seed, 'calibration')
        locked = read_json(suite/'plan.json')['references'][str(seed)]['new']['thresholds']
        decisions = fit_decisions(records, predictions(suite, seed, methods, records), locked)
        write_json(output/'decisions.json', decisions)
        roles = ['check']
    for role in roles:
        records = cache(suite, seed, role)
        values = predictions(suite, seed, methods, records)
        metrics = {name: {rule: metric(records, z, th) for rule, th in decisions['models'][name]['thresholds'].items() if th is not None}
                   for name, z in values.items()}
        fields = {k: records[k] for k in ['targets', 'identities', 'indices', 'weak_mask', 'relative_power_db']}
        np.savez_compressed(output/('predictions_'+role+'.npz'), **fields, **{'logits_'+k: z for k, z in values.items()})
        write_json(output/('metrics_'+role+'.json'), {'seed': seed, 'role': role, 'metrics': metrics})
        print({'event': 'metrics_completed', 'seed': seed, 'role': role}, flush=True)


def worker(suite, seed, methods, report=False):
    if not report:
        fit, check = cache(suite, seed, 'fit'), cache(suite, seed, 'check')
        for method in methods:
            train(suite, seed, method, fit, check)
    measure(suite, seed, methods, report)


def collect(suite, seeds, role, methods):
    rows = [read_json(suite/'evaluation'/f'seed{s}'/('metrics_'+role+'.json'))['metrics'] for s in seeds]
    return rows, {m: mean_metrics([r[m]['locked'] for r in rows]) for m in methods}


def comparisons(rows, candidate, anchor):
    a, b = [r[candidate]['locked'] for r in rows], [r[anchor]['locked'] for r in rows]
    deltas = {key: contrast([v[key] for v in a], [v[key] for v in b]) for key in KEYS}
    d = deltas['exact']
    checks = guard(mean_metrics(a), mean_metrics(b)) | {'exact_gain0_5pp': d['mean'] >= .005,
        'at_least4_of5_positive': d['positive_seeds'] >= 4, 'paired95_lower_positive': d['paired95'][0] > 0}
    return {'candidate': candidate, 'anchor': anchor, 'deltas': deltas, 'formal_checks': checks, 'passed': all(checks.values())}
