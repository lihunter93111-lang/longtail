"""Identity-safe CPU training for replication and pre-pooling residual heads."""
import hashlib
from pathlib import Path
import time

import h5py
import numpy as np
import torch
from torch import nn
from scipy.stats import t
from sklearn.metrics import roc_auc_score

from .io import read_json, write_json, sha256
from .engine.checkpoint import load_checkpoint, save_checkpoint
from .models.factory import build_model
from .models.decision_heads import FiveLabelHead, LocalResidual, residual_logits
from .models.temporal_heads import build_residual, unpooled_features
from .data.dataset import IQDataset
from .data.loaders import make_loader
from .data.sampling import sampler_epoch_seed
from .metrics import fit_thresholds, evaluate_metrics, probabilities
from .decision import fit_fpr_caps
from .staged_training import record_metric, draw_indices, head_logits, CLASSES

METRICS = ['exact', 'macro_f1', 'macro_aupr', 'tail_f1', 'mean_fpr', 'saw_weak_recall',
           'target_weak_recall', 'weak_micro_recall', 'nonweak_recall']


def contrast(a, b):
    delta = np.asarray(a, float) - np.asarray(b, float)
    half = float(t.ppf(.975, len(delta)-1) * delta.std(ddof=1) / np.sqrt(len(delta)))
    return {'mean': float(delta.mean()), 'paired95': [float(delta.mean()-half), float(delta.mean()+half)],
            'per_seed': delta.tolist(), 'positive_seeds': int((delta > 0).sum())}


def source_checkpoint(suite, stage, seed, old=False):
    plan = read_json(suite/'plan.json')
    if stage == 'retrain' and not old:
        return suite/'backbones'/f'seed{seed}/best.pt'
    return Path(plan['references'][str(seed)]['B']['checkpoint'])


def load_backbone(path):
    payload, cfg = load_checkpoint(path)
    model = build_model(cfg['model'], num_classes=6)
    model.load_state_dict(payload['model'], strict=True)
    model.eval().requires_grad_(False)
    return model, cfg


def assert_report_gate(suite, stage):
    frozen = read_json(suite/stage/'frozen.json')
    assert frozen['plan_sha256'] == sha256(suite/'plan.json')
    for path, digest in frozen['files'].items():
        assert sha256(path) == digest, path


def cache_records(suite, stage, seed, tag, role, pair=False, temporal=False, old=False):
    plan = read_json(suite/'plan.json')
    folder = suite/'cache'/stage/f'seed{seed}'
    folder.mkdir(parents=True, exist_ok=True)
    key = ('old_' if old else '') + tag + '_' + ('pair_' if pair else '') + role
    dest = folder/(key+'.npz')
    meta = folder/(key+'_source.json')
    time_file = folder/(key+'_temporal.npy')
    if role == 'development_test' or role.startswith('ood_'):
        assert_report_gate(suite, stage)
    if dest.exists():
        source = read_json(meta)
        assert source['cache_sha256'] == sha256(dest)
        assert source['backbone_sha256'] == sha256(source_checkpoint(suite, stage, seed, old))
        records = dict(np.load(dest))
        if temporal:
            assert source['temporal_sha256'] == sha256(time_file)
            records['temporal'] = np.load(time_file, mmap_mode='r')
        return records
    checkpoint = source_checkpoint(suite, stage, seed, old)
    backbone, cfg = load_backbone(checkpoint)
    if pair:
        manifest = read_json(plan['data']['pair_manifests'][tag])
        file = Path(manifest['files'][role]['path'])
        indices = np.arange(manifest['files'][role]['samples'])
    elif role.startswith('ood_'):
        file = Path(plan['data']['shifts'][role]['path'])
        indices = np.arange(plan['data']['shifts'][role]['samples'])
    else:
        manifest_path = Path(plan['data']['manifests'][tag])
        manifest = read_json(manifest_path)
        split = read_json(suite/'splits'/(tag+'.json'))
        file = manifest_path.parent/manifest['files']['train' if role == 'fit' else 'development']['path']
        indices = np.asarray(split['roles'][role])
    dataset = IQDataset(file, indices, cfg['dataloader']['preprocess'], cache='memory')
    fields, features, logits = {}, [], []
    time_cache = None
    offset = 0
    torch.set_num_threads(4)
    try:
        with torch.inference_mode():
            for batch in make_loader(dataset, cfg['dataloader'], batch_size=64):
                if temporal:
                    feature, time_feature = unpooled_features(backbone, batch['signal'])
                    if time_cache is None:
                        time_cache = np.lib.format.open_memmap(time_file.with_suffix('.partial.npy'), mode='w+',
                                        dtype=np.float32, shape=(len(dataset), *time_feature.shape[1:]))
                    time_cache[offset:offset+len(feature)] = time_feature.numpy()
                    offset += len(feature)
                else:
                    feature = backbone.forward_features(batch['signal'])
                features.append(feature.numpy())
                logits.append(backbone.classifier(feature).numpy())
                for key2 in ['target', 'index', 'identity', 'weak_mask', 'relative_power_db', 'component_jnr_db', 'component_power_db']:
                    if key2 in batch:
                        fields.setdefault(key2, []).append(batch[key2].numpy())
        records = {k: np.concatenate(v) for k, v in fields.items()}
        records['targets'] = records.pop('target').astype(np.uint8)
        records['indices'] = records.pop('index')
        records['identities'] = records.pop('identity')
        records['features'] = np.concatenate(features)
        records['base_logits'] = np.concatenate(logits)
        if role != 'fit':
            expert, _ = load_backbone(Path(plan['references'][str(seed)]['FMB']['checkpoint']))
            values = []
            with torch.inference_mode():
                for batch in make_loader(dataset, cfg['dataloader'], batch_size=64):
                    values.append(expert(batch['signal']).numpy())
            records['FMB_logits'] = np.concatenate(values)
            del expert
    finally:
        dataset.close()
    if pair:
        with h5py.File(file) as f:
            for name in ['pair_id', 'pair_target', 'pair_side', 'edge_id']:
                records[name] = f[name][:].ravel()
    if temporal:
        assert offset == len(dataset)
        time_cache.flush()
        del time_cache
        time_file.with_suffix('.partial.npy').replace(time_file)
    assert np.isfinite(records['features']).all() and np.isfinite(records['base_logits']).all()
    np.savez_compressed(dest, **records)
    write_json(meta, {'cache_sha256': sha256(dest), 'backbone_sha256': sha256(checkpoint),
                     'data_file': str(file), 'data_sha256': sha256(file), 'stage': stage, 'role': role,
                     'temporal_sha256': sha256(time_file) if temporal else None,
                     'input': 'single amplitude-normalized I/Q; no pair difference or truth inputs', 'model_mode': 'eval,frozen'})
    if temporal:
        records['temporal'] = np.load(time_file, mmap_mode='r')
    return records


def linear_checkpoint(suite, stage, seed):
    if stage == 'retrain':
        return suite/'retrain'/f'seed{seed}/linear.pt'
    return Path(read_json(suite/'plan.json')['previous_suite'])/'stage2'/f'seed{seed}/head_natural.pt'


def base_logits(suite, stage, seed, records):
    return head_logits(records, linear_checkpoint(suite, stage, seed))


def fused(records, logits):
    out = logits.copy()
    out[:, 0] = records['FMB_logits'][:, 0]
    return out


def old_candidate(suite, seed, records):
    old = Path(read_json(suite/'plan.json')['previous_suite'])
    base = head_logits(records, old/'stage2'/f'seed{seed}/head_natural.pt')
    payload = torch.load(old/'stage3'/f'seed{seed}/local_pair_bce.pt', map_location='cpu', weights_only=True)
    model = LocalResidual(active=payload['active_targets'])
    model.load_state_dict(payload['state_dict'])
    model.eval()
    with torch.inference_mode():
        return residual_logits(torch.from_numpy(base), model(torch.from_numpy(records['features']))).numpy()


def predict_residual(checkpoint, records, base):
    payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model = build_residual(payload['architecture'])
    model.load_state_dict(payload['state_dict'])
    model.eval()
    source = records['temporal'] if payload['architecture'] == 'temporal' else records['features']
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(source), 64):
            x = torch.from_numpy(np.array(source[start:start+64], copy=True))
            outputs.append(model(x).numpy())
    z = base.copy()
    z[:, [2, 4, 5]] += np.concatenate(outputs)
    return z


def train_component(suite, stage, seed, tag, name, architecture, paired, fit, check, pairs=None):
    output = suite/stage/f'seed{seed}'
    output.mkdir(parents=True, exist_ok=True)
    status = output/(name+'_status.json')
    checkpoint = output/(name+'.pt')
    last = output/('last_'+name+'.pt')
    if status.exists() and read_json(status)['status'] == 'completed':
        return
    torch.set_num_threads(4 if architecture == 'temporal' else 1)
    torch.manual_seed(seed)
    if architecture == 'linear':
        source, _ = load_checkpoint(source_checkpoint(suite, stage, seed))
        model = FiveLabelHead(source['model']['classifier.1.weight'], source['model']['classifier.1.bias'])
    else:
        model = build_residual(architecture)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=24)
    use_time = architecture == 'temporal'
    x = fit['temporal'] if use_time else fit['features']
    cv = check['temporal'] if use_time else check['features']
    y = torch.from_numpy(fit['targets'].astype(np.float32))
    if architecture != 'linear':
        b = torch.from_numpy(base_logits(suite, stage, seed, fit))
        cb = base_logits(suite, stage, seed, check)
    else:
        b = torch.from_numpy(fit['base_logits'])
        cb = check['base_logits']
    if paired:
        assert pairs is not None
        selected = np.flatnonzero(np.isin(pairs['pair_target'][::2], [2, 4, 5]))
        pair_rows = np.ravel(np.c_[2*selected, 2*selected+1])
        px = (pairs['temporal'] if use_time else pairs['features'])[pair_rows]
        py = torch.from_numpy(pairs['targets'][pair_rows].astype(np.float32))
        pb = torch.from_numpy(base_logits(suite, stage, seed, pairs)[pair_rows])
    history, best, first = [], -1., 1
    if last.exists():
        saved = torch.load(last, map_location='cpu', weights_only=True)
        assert saved['architecture'] == architecture and saved['paired'] == paired
        model.load_state_dict(saved['state_dict'])
        optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler'])
        torch.set_rng_state(saved['torch_rng'])
        history, best, first = saved['history'], saved['best_score'], saved['epoch']+1
    started = time.time()
    for epoch in range(first, 25):
        epoch_start = time.time()
        model.train()
        indices = draw_indices(fit['targets'], seed, epoch, 'natural')
        pair_draw = np.random.default_rng(sampler_epoch_seed(seed, epoch, 20)).integers(0, len(selected), size=1000) if paired else np.array([], np.int64)
        loss_sum = 0.
        for step in range(250):
            count = 24 if paired else 32
            rows = indices[step*count:(step+1)*count]
            xx = torch.from_numpy(np.array(x[rows], copy=True))
            yy, bb = y[rows], b[rows]
            if paired:
                group = pair_draw[step*4:(step+1)*4]
                pp = np.ravel(np.c_[2*group, 2*group+1])
                xx = torch.cat([xx, torch.from_numpy(np.array(px[pp], copy=True))])
                yy, bb = torch.cat([yy, py[pp]]), torch.cat([bb, pb[pp]])
            optimizer.zero_grad(set_to_none=True)
            z = model(xx) if architecture == 'linear' else residual_logits(bb, model(xx))[:, 1:]
            loss = nn.functional.binary_cross_entropy_with_logits(z, yy[:, 1:])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=True)
            optimizer.step()
            loss_sum += loss.item()
        scheduler.step()
        model.eval()
        values = []
        with torch.inference_mode():
            for start in range(0, len(cv), 64):
                values.append(model(torch.from_numpy(np.array(cv[start:start+64], copy=True))).numpy())
        prediction = np.concatenate(values)
        if architecture != 'linear':
            z = cb.copy()
            z[:, [2, 4, 5]] += prediction
            prediction = z[:, 1:]
        score = evaluate_metrics(check['targets'][:, 1:], prediction, CLASSES[1:])['macro_aupr']
        row = {'epoch': epoch, 'optimizer_steps': epoch*250, 'samples': 8000,
               'natural_samples': 6000 if paired else 8000, 'paired_views': 2000 if paired else 0,
               'loss': loss_sum/250, 'check_aupr': score, 'epoch_seconds': time.time()-epoch_start,
               'natural_draw_sha256': hashlib.sha256(indices.astype('<i8').tobytes()).hexdigest(),
               'pair_draw_sha256': hashlib.sha256(pair_draw.astype('<i8').tobytes()).hexdigest()}
        history.append(row)
        improved = score > best
        best = max(best, score)
        payload = {'kind': 'replication_temporal_head', 'architecture': architecture, 'state_dict': model.state_dict(),
                   'paired': paired, 'seed': seed, 'stage': stage, 'data_tag': tag, 'epoch': epoch,
                   'optimizer_steps': epoch*250, 'history': history, 'best_score': best,
                   'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'torch_rng': torch.get_rng_state(),
                   'backbone_sha256': sha256(source_checkpoint(suite, stage, seed)),
                   'previous_head_sha256': None if architecture == 'linear' else sha256(linear_checkpoint(suite, stage, seed)),
                   'trainable_parameters': sum(p.numel() for p in model.parameters()), 'checkpoint_selection': 'check five-label Macro-AUPR'}
        save_checkpoint(last, payload)
        if improved:
            save_checkpoint(checkpoint, payload)
        write_json(output/('history_'+name+'.json'), history)
        print({'stage': stage, 'seed': seed, 'method': name, **row}, flush=True)
    write_json(status, {'status': 'completed', 'optimizer_steps': 6000, 'epochs': 24,
                       'trainable_parameters': payload['trainable_parameters'], 'seconds': time.time()-started,
                       'checkpoint_sha256': sha256(checkpoint)})


def methods_for(suite, stage):
    if stage == 'replication':
        return {'natural': ('global32', False), 'pair_bce': ('global32', True)}
    policy = read_json(suite/'replication/selection.json')['paired_policy']
    if stage == 'temporal':
        return {k: (k, policy) for k in ['global32', 'global33', 'temporal']}
    selected = read_json(suite/'temporal/selection.json')['selected']
    return {k: (k, policy) for k in dict.fromkeys(['global32', selected])}


def predictions(suite, stage, seed, tag, records):
    base = base_logits(suite, stage, seed, records)
    out = {'base': fused(records, base)}
    for name, _ in methods_for(suite, stage).items():
        key = tag+'_'+name if stage == 'replication' else name
        out[name] = fused(records, predict_residual(suite/stage/f'seed{seed}'/(key+'.pt'), records, base))
    if stage != 'retrain':
        out['old_candidate'] = fused(records, old_candidate(suite, seed, records))
    else:
        out['backbone_original'] = fused(records, records['base_logits'])
    return out


def measure(suite, stage, seed, tag, report=False):
    output = suite/stage/f'seed{seed}'
    temporal = stage == 'temporal' or stage == 'retrain' and 'temporal' in methods_for(suite, stage)
    decisions_file = output/('decisions_'+tag+'.json')
    if not report:
        cal = cache_records(suite, stage, seed, tag, 'calibration', temporal=temporal)
        logits = predictions(suite, stage, seed, tag, cal)
        if stage == 'retrain':
            old = cache_records(suite, stage, seed, tag, 'calibration', old=True)
            logits['old_candidate'] = fused(old, old_candidate(suite, seed, old))
        thresholds = {}
        for method, z in logits.items():
            thresholds[method] = fit_thresholds(cal['targets'], z, None)
            thresholds[method][0] = .5
        anchor = 'natural' if stage == 'replication' else 'global32'
        caps = ((probabilities(logits[anchor]) >= thresholds[anchor]) & ~cal['targets'].astype(bool)).sum(0)
        matched, matched_meta = {}, {}
        for method, z in logits.items():
            matched[method], matched_meta[method] = fit_fpr_caps(cal['targets'], z, caps)
            matched[method][0] = .5
        write_json(decisions_file, {'fit_role': 'calibration', 'rule': 'per-label F1; FMB fixed0.5', 'thresholds': thresholds,
                   'matched_fpr_thresholds': matched, 'matched_fpr_meta': matched_meta, 'matched_fpr_reference': anchor})
        roles = ['calibration', 'check']
    else:
        thresholds = read_json(decisions_file)['thresholds']
        roles = ['development_test'] + (['ood_low', 'ood_wide'] if stage == 'retrain' else [])
    matched = read_json(decisions_file)['matched_fpr_thresholds']
    for role in roles:
        records = cache_records(suite, stage, seed, tag, role, temporal=temporal)
        logits = predictions(suite, stage, seed, tag, records)
        if stage == 'retrain':
            old = cache_records(suite, stage, seed, tag, role, old=True)
            assert np.array_equal(records['identities'], old['identities'])
            logits['old_candidate'] = fused(old, old_candidate(suite, seed, old))
        metrics = {m: record_metric(records, z, thresholds[m]) for m, z in logits.items()}
        matched_metrics = {m: record_metric(records, z, matched[m]) for m, z in logits.items()}
        np.savez_compressed(output/('predictions_'+tag+'_'+role+'.npz'), **{'logits_'+m: z for m, z in logits.items()},
                            targets=records['targets'], identities=records['identities'], weak_mask=records['weak_mask'], relative_power_db=records['relative_power_db'])
        write_json(output/('metrics_'+tag+'_'+role+'.json'), {'seed': seed, 'stage': stage, 'tag': tag, 'role': role, 'metrics': metrics,
                   'matched_calibration_fpr_metrics': matched_metrics,
                   'matched_fpr_scope': 'Equal calibration FP budgets; achieved report FPR is measured, not guaranteed equal'})
    if report and stage != 'temporal':
        records = cache_records(suite, stage, seed, tag, 'development_test', pair=True, temporal=temporal)
        logits = predictions(suite, stage, seed, tag, records)
        if stage == 'retrain':
            old = cache_records(suite, stage, seed, tag, 'development_test', pair=True, old=True)
            logits['old_candidate'] = fused(old, old_candidate(suite, seed, old))
        metrics = {m: record_metric(records, z, thresholds[m]) for m, z in logits.items()}
        matched_metrics = {m: record_metric(records, z, matched[m]) for m, z in logits.items()}
        diagnostics = {}
        for method, z in logits.items():
            per_target = {}
            for c in [0, 2, 4, 5]:
                rows = np.flatnonzero(records['pair_target'][::2] == c)
                weak_rows = rows[records['weak_mask'][2*rows+1, c].astype(bool)]
                subset = np.ravel(np.c_[2*weak_rows, 2*weak_rows+1])
                pp = probabilities(z)
                per_target[CLASSES[c]] = {'weak_pairs': len(weak_rows),
                    'weak_paired_auroc': float(roc_auc_score(records['targets'][subset, c], z[subset, c])) if len(weak_rows) else None,
                    'weak_positive_recall': float((pp[2*weak_rows+1, c] >= thresholds[method][c]).mean()) if len(weak_rows) else None,
                    'matched_negative_fpr': float((pp[2*weak_rows, c] >= thresholds[method][c]).mean()) if len(weak_rows) else None}
            diagnostics[method] = per_target
        role = 'pair_report'
        np.savez_compressed(output/('predictions_'+tag+'_'+role+'.npz'), **{'logits_'+m: z for m, z in logits.items()},
                            targets=records['targets'], identities=records['identities'], weak_mask=records['weak_mask'], relative_power_db=records['relative_power_db'])
        write_json(output/('metrics_'+tag+'_'+role+'.json'), {'seed': seed, 'stage': stage, 'tag': tag, 'role': role, 'metrics': metrics,
                   'matched_calibration_fpr_metrics': matched_metrics, 'weak_pair_diagnostics': diagnostics,
                   'scope': 'Supplementary held-out pair edges relative to pair fit; no calibration or selection on these pairs'})


def work(suite, stage, seed, report=False):
    if report:
        for tag in (['repA', 'repB'] if stage == 'replication' else ['temporal' if stage == 'temporal' else 'retrain']):
            measure(suite, stage, seed, tag, report=True)
        return
    if stage == 'retrain':
        from .train import run
        output = suite/'backbones'/f'seed{seed}'
        if not (output/'status.json').exists() or read_json(output/'status.json')['status'] != 'completed':
            run(read_json(suite/'configs'/f'backbone_seed{seed}.json'), resume=output/'last.pt' if (output/'last.pt').exists() else None)
        # Bind the newly trained backbone before caching or training heads.
        write_json(suite/'retrain'/f'seed{seed}/backbone_frozen.json', {'checkpoint': str(output/'best.pt'), 'sha256': sha256(output/'best.pt')})
    for tag in (['repA', 'repB'] if stage == 'replication' else ['temporal' if stage == 'temporal' else 'retrain']):
        temporal = stage == 'temporal' or stage == 'retrain' and 'temporal' in methods_for(suite, stage)
        fit = cache_records(suite, stage, seed, tag, 'fit', temporal=temporal)
        check = cache_records(suite, stage, seed, tag, 'check', temporal=temporal)
        if stage == 'retrain':
            train_component(suite, stage, seed, tag, 'linear', 'linear', False, fit, check)
        pairs = None
        for name, (architecture, paired) in methods_for(suite, stage).items():
            if paired and pairs is None:
                pair_tag = tag if stage != 'temporal' else 'repA'
                pairs = cache_records(suite, stage, seed, pair_tag, 'fit', pair=True, temporal=temporal)
            key = tag+'_'+name if stage == 'replication' else name
            train_component(suite, stage, seed, tag, key, architecture, paired, fit, check, pairs)
        measure(suite, stage, seed, tag)


def guard(means, name, reference='base'):
    current, base = means[name], means[reference]
    return {'tail_f1': current['tail_f1'] >= base['tail_f1']-.003,
            'saw_weak': current['saw_weak_recall'] >= base['saw_weak_recall']-.03,
            'isrj_weak': current['weak_component']['per_label']['ISRJ']['weak_recall'] >= base['weak_component']['per_label']['ISRJ']['weak_recall']-.03,
            'class_fpr': max(current['per_label'][c]['negative_fpr']-base['per_label'][c]['negative_fpr'] for c in CLASSES) <= .01}


def mean_metrics(rows):
    result = {k: float(np.mean([r[k] for r in rows])) for k in METRICS}
    result['per_label'] = {c: {'negative_fpr': float(np.mean([r['per_label'][c]['negative_fpr'] for r in rows]))} for c in CLASSES}
    result['weak_component'] = {'per_label': {c: {'weak_recall': float(np.mean([r['weak_component']['per_label'][c]['weak_recall'] for r in rows]))} for c in CLASSES}}
    return result


def collect(suite, stage, tag, role):
    seeds = read_json(suite/'plan.json')['seeds']
    rows = [read_json(suite/stage/f'seed{s}'/('metrics_'+tag+'_'+role+'.json'))['metrics'] for s in seeds]
    return rows, {m: mean_metrics([r[m] for r in rows]) for m in rows[0]}


def select(suite, stage):
    if stage == 'replication':
        details = {}
        passed = []
        for tag in ['repA', 'repB']:
            rows, means = collect(suite, stage, tag, 'check')
            criteria = guard(means, 'pair_bce', 'natural')
            criteria['exact_positive'] = means['pair_bce']['exact'] > means['natural']['exact']
            details[tag] = {'means': means, 'criteria': criteria}
            passed.append(all(criteria.values()))
        selected = 'pair_bce' if all(passed) else 'natural'
        result = {'selected': selected, 'paired_policy': selected == 'pair_bce', 'replications': details,
                  'selection_rule': 'paired only if both independent checks improve Exact and pass all safeguards; otherwise natural'}
    else:
        tag = 'temporal' if stage == 'temporal' else 'retrain'
        rows, means = collect(suite, stage, tag, 'check')
        names = list(methods_for(suite, stage))
        criteria = {m: guard(means, m, 'global32') for m in names}
        eligible = [m for m in names if all(criteria[m].values()) and means[m]['exact'] >= means['global32']['exact']]
        selected = max(eligible, key=lambda m: (means[m]['exact'], m == 'global32')) if eligible else 'global32'
        result = {'selected': selected, 'means': means, 'criteria': criteria,
                  'selection_rule': 'check Exact among local architectures; global32 fallback; tail/FPR/Saw/ISRJ safeguards'}
        if stage == 'retrain':
            result['architecture_from_temporal_check'] = read_json(suite/'temporal/selection.json')['selected']
    result.update(selected_on='check only', report_values_used=False)
    write_json(suite/stage/'selection.json', result)
    return result


def freeze(suite, stage):
    from datetime import datetime
    from zoneinfo import ZoneInfo
    plan = read_json(suite/'plan.json')
    files = {}
    for seed, refs in plan['references'].items():
        for ref in refs.values():
            assert sha256(ref['checkpoint']) == ref['sha256']
            files[ref['checkpoint']] = ref['sha256']
        old = Path(plan['previous_suite'])
        for path in [old/'stage2'/f'seed{seed}/head_natural.pt', old/'stage3'/f'seed{seed}/local_pair_bce.pt']:
            files[str(path)] = sha256(path)
    for path in (suite/stage).rglob('*'):
        if path.suffix == '.pt' and not path.name.startswith('last_') or path.name.startswith('decisions_') or path.name == 'selection.json':
            files[str(path)] = sha256(path)
    if stage == 'retrain':
        for seed in plan['seeds']:
            path = source_checkpoint(suite, stage, seed)
            files[str(path)] = sha256(path)
    write_json(suite/stage/'frozen.json', {'plan_sha256': sha256(suite/'plan.json'), 'files': files,
               'frozen_at': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds'),
               'frozen_before_report': True, 'report_values_used': False})


def summarize(suite, stage):
    tags = ['repA', 'repB'] if stage == 'replication' else ['temporal' if stage == 'temporal' else 'retrain']
    output = {'stage': stage, 'selection': read_json(suite/stage/'selection.json'), 'reports': {}}
    for tag in tags:
        rows, means = collect(suite, stage, tag, 'development_test')
        contrasts = {}
        pairs = [('pair_bce', 'natural')] if stage == 'replication' else [(m, 'global32') for m in methods_for(suite, stage) if m != 'global32']
        pairs += [(m, 'base') for m in methods_for(suite, stage)]
        for candidate, baseline in pairs:
            key = candidate+'_vs_'+baseline
            comparisons = {k: contrast([r[candidate][k] for r in rows], [r[baseline][k] for r in rows]) for k in METRICS}
            guards = guard(means, candidate, baseline)
            comparisons['success'] = {'exact_at_least_half_pp': comparisons['exact']['mean'] >= .005,
                                      'positive_at_least_four_seeds': comparisons['exact']['positive_seeds'] >= 4,
                                      'paired95_lower_positive': comparisons['exact']['paired95'][0] > 0, **guards}
            comparisons['passed'] = all(comparisons['success'].values())
            contrasts[key] = comparisons
        output['reports'][tag] = {'means': means, 'contrasts': contrasts}
    if stage == 'replication':
        data = [collect(suite, stage, tag, 'development_test')[0] for tag in tags]
        # Average over data replications within each model seed, never pretend 10
        # observations are independent when the same backbone seeds are reused.
        a = [np.mean([r[i]['pair_bce']['exact'] for r in data]) for i in range(5)]
        b = [np.mean([r[i]['natural']['exact'] for r in data]) for i in range(5)]
        output['data_replication_averaged_within_seed'] = contrast(a, b)
        output['not_independent_ten_seed_test'] = True
    if stage == 'retrain':
        for shift in ['ood_low', 'ood_wide']:
            _, means = collect(suite, stage, 'retrain', shift)
            output['reports'][shift] = {'means': means, 'thresholds': 'unchanged main calibration; no OOD refitting'}
    if stage != 'temporal':
        output['supplementary_pair_reports'] = {tag: collect(suite, stage, tag, 'pair_report')[1] for tag in tags}
    write_json(suite/stage/'summary.json', output)
    return output
