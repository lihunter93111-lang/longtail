"""SawFM-only supervised projection with frozen-feature capacity controls."""
from copy import deepcopy
import hashlib
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from . import segment_statistics as segment, isrj_segment as isrj
from .io import read_json, write_json, sha256
from .engine.checkpoint import save_checkpoint
from .data.dataset import IQDataset
from .data.loaders import make_loader
from .models.temporal_heads import unpooled_features
from .replication_temporal import load_backbone, contrast
from .targeted_validation import candidate_logits, SawMFTHead
from .staged_training import record_metric, draw_indices, CLASSES
from .metrics import probabilities, fit_thresholds
from .decision import fit_fpr_caps

METHODS = ['pca_small', 'pca_wide', 'learned_projection']
SHAPES = {'pca_small': (128, 32), 'pca_wide': (128, 442), 'learned_projection': (1792, 32)}
NON_TARGETS = [0, 1, 3, 4, 5]
REPORT_ROLES = ['development_test', 'ood_wide', 'arrival_ood', 'saw_period_ood', 'shift0', 'shift1', 'shift3', 'shift7']
KEYS = ['exact', 'macro_f1', 'macro_aupr', 'tail_f1', 'mean_fpr',
        'saw_weak_recall', 'saw_weak_ap', 'saw_fpr', 'mft_weak_recall', 'isrj_weak_recall']


class SawHead(nn.Module):
    def __init__(self, dimensions, width):
        super().__init__()
        self.hidden = nn.Linear(dimensions, width)
        self.output = nn.Linear(width, 1)

    def forward(self, x):
        return self.output(nn.functional.gelu(self.hidden(x)))


def initialize_head(method, old, state):
    """Embed the existing Saw row, or fold its PCA into the first layer."""
    model = SawHead(*SHAPES[method])
    with torch.no_grad():
        model.output.weight.zero_()
        model.output.bias.copy_(old['output.bias'][0:1])
        model.output.weight[:, :32].copy_(old['output.weight'][0:1])
        if method == 'learned_projection':
            folded = old['hidden.weight'] @ torch.from_numpy(state['components'])
            model.hidden.weight.copy_(folded)
            model.hidden.bias.copy_(old['hidden.bias']-folded @ torch.from_numpy(state['center']))
        else:
            model.hidden.weight[:32].copy_(old['hidden.weight'])
            model.hidden.bias[:32].copy_(old['hidden.bias'])
    return model


def compose(reference, delta):
    z = np.asarray(reference).copy()
    if np.asarray(delta).shape != (len(z), 1):
        raise ValueError('Expected a single SawFM delta')
    z[:, 2] += np.asarray(delta)[:, 0]
    return z


def physics_cache(suite, seed, role):
    segment.report_gate(suite)
    plan = read_json(suite/'plan.json')
    refs = plan['references'][str(seed)]
    folder = suite/'cache'/f'seed{seed}'
    folder.mkdir(parents=True, exist_ok=True)
    dest, binding = folder/(role+'.npz'), folder/(role+'_source.json')
    if dest.exists() and binding.exists():
        assert sha256(dest) == read_json(binding)['cache_sha256']
        return dict(np.load(dest))
    if dest.exists():
        dest.unlink()  # Incomplete cache without its commit record is rebuildable.
    info = plan['data']['files'][role]
    file = Path(info['path'])
    assert sha256(file) == plan['data_hashes'][str(file)]
    backbone, cfg = load_backbone(Path(refs['new']['B']['path']))
    torch.set_num_threads(4)
    dataset = IQDataset(file, np.arange(info['samples']), cfg['dataloader']['preprocess'], cache='memory')
    fields = {}
    try:
        with torch.inference_mode():
            for batch in make_loader(dataset, cfg['dataloader'], batch_size=64):
                g, t = unpooled_features(backbone, batch['signal'])
                for key, value in [('features', g), ('global128', g), ('segments124', segment.segment_statistics(t)), ('base_logits', backbone.classifier(g))]:
                    fields.setdefault(key, []).append(value.numpy())
                for key in ['target', 'index', 'identity', 'weak_mask', 'relative_power_db']:
                    fields.setdefault(key, []).append(batch[key].numpy())
        rec = {k: np.concatenate(v) for k, v in fields.items()}
        rec['targets'] = rec.pop('target').astype(np.uint8)
        rec['indices'], rec['identities'] = rec.pop('index'), rec.pop('identity')
        expert, _ = load_backbone(Path(refs['FMB']['path']))
        outputs = []
        with torch.inference_mode():
            for batch in make_loader(dataset, cfg['dataloader'], batch_size=64):
                outputs.append(expert(batch['signal']).numpy())
        rec['FMB_logits'] = np.concatenate(outputs)
    finally:
        dataset.close()
    rec['linear_logits'], rec['natural_logits'] = candidate_logits(suite, seed, 'new', rec)
    payload = torch.load(refs['target_head']['path'], map_location='cpu', weights_only=True)
    head = SawMFTHead(payload['initial_state'])
    head.load_state_dict(payload['state_dict'])
    head.eval()
    with torch.inference_mode():
        delta = head(torch.from_numpy(rec['features'])).numpy()
    rec['anchor_logits'] = rec['natural_logits'].copy()
    rec['anchor_logits'][:, [2, 4]] = rec['linear_logits'][:, [2, 4]]+delta
    np.savez(dest, **rec)
    write_json(binding, {'cache_sha256': sha256(dest), 'data_file': str(file), 'data_sha256': sha256(file),
        'backbone_sha256': refs['new']['B']['sha256'], 'role': role, 'physics': info['physics'],
        'input': 'single received I/Q; no component, power, class or pair-difference inputs'})
    print({'event': 'cache_completed', 'seed': seed, 'role': role}, flush=True)
    return rec


def cache(suite, seed, role):
    if role in ['arrival_ood', 'saw_period_ood']:
        rec = isrj.attach_reference(suite, seed, physics_cache(suite, seed, role))
    else:
        rec = isrj.cache(suite, seed, role)
    spec = read_json(suite/'plan.json')['isrj_references'][str(seed)]
    assert sha256(spec['isrj_head']['path']) == spec['isrj_head']['sha256']
    head = isrj.ISRJResidual()
    head.load_state_dict(torch.load(spec['isrj_head']['path'], map_location='cpu', weights_only=True)['state_dict'])
    head.eval()
    x = torch.from_numpy(isrj.features(suite, seed, 'segments124', rec))
    with torch.inference_mode():
        rec['reference_logits'] = isrj.compose(rec['reference_logits'], head(x).numpy())
    return rec


def features(suite, seed, method, records):
    spec = read_json(suite/'plan.json')['feature_transforms'][str(seed)]['segments124']
    assert sha256(spec['path']) == spec['sha256']
    state = segment.load_transform(spec['path'])
    x = records['segments124']
    if method == 'learned_projection':
        return np.ascontiguousarray((x-state['mean'])/state['scale'], dtype=np.float32)
    return segment.transform(x, state)


def metric(records, logits, thresholds):
    result = record_metric(records, logits, thresholds)
    weak = result['weak_component']['per_label']
    result.update(saw_weak_ap=weak['SawFM']['weak_aupr'], saw_fpr=weak['SawFM']['negative_fpr'],
                  mft_weak_recall=weak['MFT']['weak_recall'], isrj_weak_recall=weak['ISRJ']['weak_recall'])
    return result


def mean_metrics(rows):
    out = {k: float(np.mean([r[k] for r in rows])) for k in KEYS}
    out['per_label'] = {c: {'negative_fpr': float(np.mean([r['per_label'][c]['negative_fpr'] for r in rows]))} for c in CLASSES}
    return out


def guard(row, anchor):
    return {'exact_noninferior': row['exact'] >= anchor['exact']-.002-1e-12,
            'saw_weak_recall': row['saw_weak_recall'] >= anchor['saw_weak_recall']-.01-1e-12,
            'saw_fpr': row['saw_fpr'] <= anchor['saw_fpr']+.005+1e-12,
            'saw_weak_ap': row['saw_weak_ap'] >= anchor['saw_weak_ap']-.002-1e-12,
            'tail_f1': row['tail_f1'] >= anchor['tail_f1']-.003-1e-12,
            'class_fpr': all(row['per_label'][c]['negative_fpr'] <= anchor['per_label'][c]['negative_fpr']+.005+1e-12 for c in CLASSES)}


def fit_decisions(records, values, locked):
    y = records['targets'].astype(bool)
    counts = ((probabilities(values['reference']) >= locked) & ~y).sum(0)
    models = {}
    for name, z in values.items():
        f1, cap = np.array(locked, float), np.array(locked, float)
        f1[2] = fit_thresholds(y[:, [2]], z[:, [2]], None)[0]
        proposed, meta = fit_fpr_caps(y, z, counts)
        cap[2] = proposed[2]
        models[name] = {'thresholds': {'locked': list(locked), 'saw_f1': f1.tolist(),
            'matched_calibration_fpr': cap.tolist() if meta['empirical_caps_satisfied'][2] else None}, 'matched_fpr_metadata': meta}
    return {'fit_role': 'calibration only', 'primary': 'locked', 'models': models,
            'matched_fpr_scope': 'same calibration FP count; report actual FPR independently measured'}


def train(suite, seed, method, fit, check):
    output = suite/'models'/f'seed{seed}'
    output.mkdir(parents=True, exist_ok=True)
    status, checkpoint, last = output/(method+'_status.json'), output/(method+'.pt'), output/('last_'+method+'.pt')
    if status.exists() and read_json(status)['status'] == 'completed':
        return
    plan = read_json(suite/'plan.json')
    epochs, steps = plan['training']['epochs'], plan['training']['steps_per_epoch']
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    old = torch.load(plan['segment_references'][str(seed)]['residual_head']['path'], map_location='cpu', weights_only=True)['state_dict']
    state = segment.load_transform(plan['feature_transforms'][str(seed)]['segments124']['path'])
    model = initialize_head(method, old, state)
    initial = deepcopy(model).eval().requires_grad_(False)
    initial_state = {k: v.clone() for k, v in initial.state_dict().items()}
    x, cv = [torch.from_numpy(features(suite, seed, method, r)) for r in [fit, check]]
    y = torch.from_numpy(fit['targets'][:, [2]].astype(np.float32))
    with torch.inference_mode():
        fit_origin, check_origin = initial(x).clone(), initial(cv).clone()
    base = torch.from_numpy(fit['reference_logits'][:, [2]])-fit_origin
    th = plan['references'][str(seed)]['new']['thresholds']
    anchor = metric(check, check['reference_logits'], th)
    optimizer = torch.optim.AdamW([{'params': model.hidden.parameters(), 'lr': .00003},
                                   {'params': model.output.parameters(), 'lr': .0003}], weight_decay=.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    history, first, best, best_epoch = [], 1, anchor['exact'], 0

    def payload(epoch):
        return {'kind': 'anchored_saw_projection', 'method': method, 'seed': seed, 'state_dict': model.state_dict(),
            'initial_state': initial_state, 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
            'torch_rng': torch.get_rng_state(), 'history': history, 'epoch': epoch, 'best_epoch': best_epoch,
            'best_exact': best, 'parameters': sum(p.numel() for p in model.parameters()),
            'plan_sha256': sha256(suite/'plan.json'), 'input_shape': SHAPES[method], 'changed_logit_columns': [2]}

    if last.exists():
        saved = torch.load(last, map_location='cpu', weights_only=True)
        assert saved['plan_sha256'] == sha256(suite/'plan.json')
        model.load_state_dict(saved['state_dict'])
        optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler'])
        torch.set_rng_state(saved['torch_rng'])
        history, first, best, best_epoch = saved['history'], saved['epoch']+1, saved['best_exact'], saved['best_epoch']
    else:
        with torch.inference_mode():
            np.testing.assert_array_equal(compose(check['reference_logits'], (model(cv)-check_origin).numpy()), check['reference_logits'])
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
            z = compose(check['reference_logits'], (model(cv)-check_origin).numpy())
        m = metric(check, z, th)
        checks = guard(m, anchor)
        improved = all(checks.values()) and m['exact'] > best+1e-12
        if improved:
            best, best_epoch = m['exact'], epoch
        row = {'epoch': epoch, 'optimizer_steps': epoch*steps, 'loss': loss_sum/steps,
            'check_exact': m['exact'], 'check_saw_weak_recall': m['saw_weak_recall'],
            'check_saw_weak_ap': m['saw_weak_ap'], 'check_saw_fpr': m['saw_fpr'], 'check_guard': checks,
            'best_epoch': best_epoch, 'natural_views': 32*steps,
            'natural_order_sha256': hashlib.sha256(order.astype('<i8').tobytes()).hexdigest(), 'epoch_seconds': time.time()-tick}
        history.append(row)
        save_checkpoint(last, payload(epoch))
        if improved:
            save_checkpoint(checkpoint, payload(epoch))
        write_json(output/('history_'+method+'.json'), history)
        print({'event': 'train_epoch', 'seed': seed, 'method': method, **row}, flush=True)
    write_json(status, {'status': 'completed', 'epochs': epochs, 'optimizer_steps': epochs*steps,
        'parameters': sum(p.numel() for p in model.parameters()), 'best_epoch': best_epoch,
        'seconds': time.time()-started, 'checkpoint_sha256': sha256(checkpoint)})


def predictions(suite, seed, methods, records):
    values = {'reference': records['reference_logits']}
    for method in methods:
        payload = torch.load(suite/'models'/f'seed{seed}'/(method+'.pt'), map_location='cpu', weights_only=True)
        assert payload['plan_sha256'] == sha256(suite/'plan.json')
        model, initial = SawHead(*SHAPES[method]), SawHead(*SHAPES[method])
        model.load_state_dict(payload['state_dict'])
        initial.load_state_dict(payload['initial_state'])
        model.eval(); initial.eval()
        x = torch.from_numpy(features(suite, seed, method, records))
        with torch.inference_mode():
            values[method] = compose(records['reference_logits'], (model(x)-initial(x)).numpy())
        np.testing.assert_array_equal(values[method][:, NON_TARGETS], values['reference'][:, NON_TARGETS])
    return values


def measure(suite, seed, methods, report=False):
    output = suite/'evaluation'/f'seed{seed}'
    output.mkdir(parents=True, exist_ok=True)
    if report:
        segment.report_gate(suite)
        decisions = read_json(output/'decisions.json')
        roles = REPORT_ROLES
    else:
        rec = cache(suite, seed, 'calibration')
        locked = read_json(suite/'plan.json')['references'][str(seed)]['new']['thresholds']
        decisions = fit_decisions(rec, predictions(suite, seed, methods, rec), locked)
        write_json(output/'decisions.json', decisions)
        roles = ['check']
    for role in roles:
        rec = cache(suite, seed, role)
        values = predictions(suite, seed, methods, rec)
        measured = {name: {rule: metric(rec, z, th) for rule, th in decisions['models'][name]['thresholds'].items() if th is not None} for name, z in values.items()}
        fields = {k: rec[k] for k in ['targets', 'identities', 'indices', 'weak_mask', 'relative_power_db']}
        np.savez_compressed(output/('predictions_'+role+'.npz'), **fields, **{'logits_'+k: z for k, z in values.items()})
        write_json(output/('metrics_'+role+'.json'), {'seed': seed, 'role': role, 'metrics': measured})
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
