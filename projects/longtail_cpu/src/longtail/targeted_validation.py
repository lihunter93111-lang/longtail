"""Frozen-candidate calibration and controlled training of SawFM/MFT head copies."""
from datetime import datetime
import hashlib
from pathlib import Path
import time
from zoneinfo import ZoneInfo

import h5py
import numpy as np
import torch
from torch import nn

from .io import read_json, write_json, sha256
from .engine.checkpoint import load_checkpoint, save_checkpoint
from .models.decision_heads import LocalResidual
from .models.temporal_heads import GlobalResidual
from .data.dataset import IQDataset, read_metadata
from .data.loaders import make_loader
from .data.sampling import sampler_epoch_seed
from .staged_training import head_logits, draw_indices, record_metric, CLASSES
from .replication_temporal import load_backbone, contrast
from .calibration_tradeoff import fit_policies, RULES, TARGETS
from .weak_metrics import weak_metrics

METHODS = {'natural': 0, 'pair1': 1, 'pair2': 2}
KEYS = ['exact', 'macro_f1', 'macro_aupr', 'tail_f1', 'mean_fpr', 'saw_weak_recall',
        'mft_weak_recall', 'focus_weak_recall', 'focus_weak_aupr', 'focus_fpr', 'nonweak_recall']


def now():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')


def references(suite, seed):
    return read_json(suite/'plan.json')['references'][str(seed)]


def check_report_gate(suite, stage):
    frozen = read_json(suite/stage/'frozen.json')
    assert frozen['plan_sha256'] == sha256(suite/'plan.json')
    for path, digest in frozen['files'].items():
        assert sha256(path) == digest, path


def candidate_logits(suite, seed, version, records):
    refs = references(suite, seed)[version]
    for kind in ['linear', 'local']:
        assert sha256(refs[kind]['path']) == refs[kind]['sha256']
    base = head_logits(records, refs['linear']['path'])
    payload = torch.load(refs['local']['path'], map_location='cpu', weights_only=True)
    model = GlobalResidual() if version == 'new' else LocalResidual(active=payload['active_targets'])
    model.load_state_dict(payload['state_dict'])
    model.eval().requires_grad_(False)
    with torch.inference_mode():
        residual = model(torch.from_numpy(records['features'])).numpy()
    out = base.copy()
    out[:, [2,4,5]] += residual
    if 'FMB_logits' in records:
        out[:, 0] = records['FMB_logits'][:, 0]
    return base, out


def cache(suite, stage, seed, version, role, pair=False):
    if role in ['development_test', 'ood_wide']:
        check_report_gate(suite, stage)
    plan = read_json(suite/'plan.json')
    folder = suite/'cache'/stage/f'seed{seed}'
    folder.mkdir(parents=True, exist_ok=True)
    tag = version+'_'+('pair_' if pair else '')+role
    dest, origin = folder/(tag+'.npz'), folder/(tag+'_source.json')
    refs = references(suite, seed)
    if dest.exists():
        saved = read_json(origin)
        assert saved['cache_sha256'] == sha256(dest)
        assert saved['B_sha256'] == refs[version]['B']['sha256']
        return dict(np.load(dest))
    if pair:
        manifest = read_json(plan['data']['pair_manifest'])
        file = Path(manifest['files'][role]['path'])
        indices = np.arange(manifest['files'][role]['samples'])
    elif role == 'ood_wide':
        info = plan['data']['shifts']['ood_wide']
        file, indices = Path(info['path']), np.arange(info['samples'])
    else:
        manifest = Path(plan['data']['manifests'][stage])
        file = manifest.parent/('train.h5' if role == 'fit' else 'development.h5')
        indices = np.asarray(read_json(suite/'splits'/(stage+'.json'))['roles'][role])
    reused = None
    if role == 'fit' and version == 'new' and not pair:
        previous = Path(plan['previous_suite'])/'cache/retrain'/f'seed{seed}'
        old = previous/'retrain_fit.npz'
        binding = read_json(previous/'retrain_fit_source.json')
        assert sha256(old) == binding['cache_sha256']
        assert binding['backbone_sha256'] == refs['new']['B']['sha256']
        assert sha256(file) == binding['data_sha256']
        records = dict(np.load(old))
        yy, identities, _ = read_metadata(file)
        np.testing.assert_array_equal(records['identities'], identities[records['indices']])
        np.testing.assert_array_equal(records['targets'], yy[records['indices']])
        reused = str(old)
    else:
        torch.set_num_threads(4)
        assert sha256(refs[version]['B']['path']) == refs[version]['B']['sha256']
        backbone, cfg = load_backbone(Path(refs[version]['B']['path']))
        dataset = IQDataset(file, indices, cfg['dataloader']['preprocess'], cache='memory')
        fields, features, logits = {}, [], []
        try:
            with torch.inference_mode():
                for batch in make_loader(dataset, cfg['dataloader'], batch_size=64):
                    x = backbone.forward_features(batch['signal'])
                    features.append(x.numpy())
                    logits.append(backbone.classifier(x).numpy())
                    for key in ['target', 'index', 'identity', 'weak_mask', 'relative_power_db']:
                        fields.setdefault(key, []).append(batch[key].numpy())
            records = {k: np.concatenate(v) for k, v in fields.items()}
            records['targets'] = records.pop('target').astype(np.uint8)
            records['indices'], records['identities'] = records.pop('index'), records.pop('identity')
            records.update(features=np.concatenate(features), base_logits=np.concatenate(logits))
            if role != 'fit':
                counterpart = folder/('new_'+('pair_' if pair else '')+role+'.npz')
                if version == 'old' and counterpart.exists():
                    companion = dict(np.load(counterpart))
                    np.testing.assert_array_equal(records['identities'], companion['identities'])
                    records['FMB_logits'] = companion['FMB_logits']
                else:
                    assert sha256(refs['FMB']['path']) == refs['FMB']['sha256']
                    expert, _ = load_backbone(Path(refs['FMB']['path']))
                    values = []
                    with torch.inference_mode():
                        for batch in make_loader(dataset, cfg['dataloader'], batch_size=64):
                            values.append(expert(batch['signal']).numpy())
                    records['FMB_logits'] = np.concatenate(values)
                    del expert
        finally:
            dataset.close()
        del backbone
    if pair:
        with h5py.File(file) as f:
            for key in ['pair_id', 'pair_side', 'pair_target', 'target_gap_bin', 'edge_id']:
                records[key] = f[key][:].ravel()
    records['linear_logits'], records['candidate_logits'] = candidate_logits(suite, seed, version, records)
    np.savez_compressed(dest, **records)
    write_json(origin, {'cache_sha256': sha256(dest), 'data_file': str(file), 'data_sha256': sha256(file),
                        'B_sha256': refs[version]['B']['sha256'], 'role': role, 'version': version,
                        'reused_exact_frozen_features': reused, 'pair': pair,
                        'fit_FMB_not_used': role == 'fit', 'input': 'single normalized I/Q; no gap metadata, latent components or pair differences'})
    return records


class SawMFTHead(nn.Module):
    def __init__(self, state):
        super().__init__()
        self.hidden = nn.Linear(128, 32)
        self.output = nn.Linear(32, 2)
        with torch.no_grad():
            self.hidden.weight.copy_(state['hidden.weight'])
            self.hidden.bias.copy_(state['hidden.bias'])
            self.output.weight.copy_(state['output.weight'][[0,1]])
            self.output.bias.copy_(state['output.bias'][[0,1]])

    def forward(self, x):
        return self.output(torch.nn.functional.gelu(self.hidden(x)))


def compose_targets(records, delta):
    out = records['candidate_logits'].copy()
    out[:, TARGETS] = records['linear_logits'][:, TARGETS] + delta
    return out


def pair_schedule(pairs, seed, epoch, pairs_per_step):
    strata = (pairs['pair_target'][::2] == 4).astype(int)*3+pairs['target_gap_bin'][::2]
    groups = [np.flatnonzero(strata == k) for k in range(6)]
    assert all(len(g) for g in groups)
    count = 250*pairs_per_step
    sequence = (np.arange(count)+(epoch-1)*count) % 6
    rng = np.random.default_rng(sampler_epoch_seed(seed, epoch, 31))
    rng.shuffle(sequence)
    selected = np.array([rng.choice(groups[k]) for k in sequence], np.int64)
    return selected, sequence


def predict_copy(suite, seed, method, records):
    payload = torch.load(suite/'targeted'/f'seed{seed}'/(method+'.pt'), map_location='cpu', weights_only=True)
    model = SawMFTHead(payload['initial_state'])
    model.load_state_dict(payload['state_dict'])
    model.eval()
    with torch.inference_mode():
        delta = model(torch.from_numpy(records['features'])).numpy()
    return compose_targets(records, delta)


def metric(records, logits, thresholds):
    m = record_metric(records, logits, thresholds)
    focus = [m['weak_component']['per_label'][c] for c in ['SawFM', 'MFT']]
    m.update(mft_weak_recall=focus[1]['weak_recall'], focus_weak_recall=float(np.mean([v['weak_recall'] for v in focus])),
             focus_weak_aupr=float(np.mean([v['weak_aupr'] for v in focus])), focus_fpr=float(np.mean([v['negative_fpr'] for v in focus])))
    return m


def train_copy(suite, seed, method, fit, check, pairs):
    output = suite/'targeted'/f'seed{seed}'
    output.mkdir(parents=True, exist_ok=True)
    status = output/(method+'_status.json')
    checkpoint, last = output/(method+'.pt'), output/('last_'+method+'.pt')
    if status.exists() and read_json(status)['status'] == 'completed':
        return
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    reference = references(suite, seed)['new']['local']
    initial = torch.load(reference['path'], map_location='cpu', weights_only=True)['state_dict']
    model = SawMFTHead(initial)
    x, y = torch.from_numpy(fit['features']), torch.from_numpy(fit['targets'][:, TARGETS].astype(np.float32))
    linear = torch.from_numpy(fit['linear_logits'][:, TARGETS])
    cv = torch.from_numpy(check['features'])
    cl = torch.from_numpy(check['linear_logits'][:, TARGETS])
    with torch.inference_mode():
        initial_z = model(cv)+cl
    torch.testing.assert_close(initial_z, torch.from_numpy(check['candidate_logits'][:, TARGETS]), rtol=1e-5, atol=2e-6)
    px = torch.from_numpy(pairs['features'])
    py = torch.from_numpy(pairs['targets'][:, TARGETS].astype(np.float32))
    pl = torch.from_numpy(pairs['linear_logits'][:, TARGETS])
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0003, weight_decay=.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=24)
    first, best, history = 1, -1., []
    if last.exists():
        old = torch.load(last, map_location='cpu', weights_only=True)
        model.load_state_dict(old['state_dict'])
        optimizer.load_state_dict(old['optimizer'])
        scheduler.load_state_dict(old['scheduler'])
        torch.set_rng_state(old['torch_rng'])
        first, best, history = old['epoch']+1, old['best_score'], old['history']
    n_pairs = METHODS[method]
    started = time.time()
    for epoch in range(first, 25):
        tick = time.time()
        model.train()
        natural = draw_indices(fit['targets'], seed, epoch, 'natural')
        chosen, strata = pair_schedule(pairs, seed, epoch, n_pairs) if n_pairs else (np.array([], np.int64), np.array([], np.int64))
        count = 32-2*n_pairs
        loss_sum = 0.
        for step in range(250):
            rows = natural[step*count:(step+1)*count]
            xx, yy, ll = x[rows], y[rows], linear[rows]
            if n_pairs:
                selected = chosen[step*n_pairs:(step+1)*n_pairs]
                pp = np.ravel(np.c_[2*selected, 2*selected+1])
                xx, yy, ll = torch.cat([xx, px[pp]]), torch.cat([yy, py[pp]]), torch.cat([ll, pl[pp]])
            assert len(xx) == 32
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.binary_cross_entropy_with_logits(model(xx)+ll, yy)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=True)
            optimizer.step()
            loss_sum += loss.item()
        scheduler.step()
        model.eval()
        with torch.inference_mode():
            z = compose_targets(check, model(cv).numpy())
        weak = weak_metrics(check['targets'], z, CLASSES, [.5]*6, check['weak_mask'], check['relative_power_db'])
        score = float(np.mean([weak['per_label'][c]['weak_aupr'] for c in ['SawFM', 'MFT']]))
        row = {'epoch': epoch, 'optimizer_steps': epoch*250, 'samples': 8000, 'natural_views': count*250,
               'paired_views': 2*n_pairs*250, 'pair_stratum_counts': np.bincount(strata, minlength=6).tolist(),
               'loss': loss_sum/250, 'check_focus_weak_aupr': score, 'epoch_seconds': time.time()-tick,
               'natural_order_sha256': hashlib.sha256(natural[:count*250].astype('<i8').tobytes()).hexdigest(),
               'pair_order_sha256': hashlib.sha256(chosen.astype('<i8').tobytes()).hexdigest()}
        history.append(row)
        improved = score > best
        best = max(best, score)
        payload = {'kind': 'saw_mft_head_copy', 'state_dict': model.state_dict(), 'initial_state': initial,
                   'seed': seed, 'method': method, 'epoch': epoch, 'optimizer_steps': epoch*250,
                   'history': history, 'best_score': best, 'optimizer': optimizer.state_dict(),
                   'scheduler': scheduler.state_dict(), 'torch_rng': torch.get_rng_state(),
                   'initial_checkpoint_sha256': reference['sha256'], 'trainable_parameters': sum(p.numel() for p in model.parameters()),
                   'selection': 'check mean weak AUPR of SawFM/MFT', 'changed_logit_columns': TARGETS}
        save_checkpoint(last, payload)
        if improved:
            save_checkpoint(checkpoint, payload)
        write_json(output/('history_'+method+'.json'), history)
        print({'stage': 'targeted', 'seed': seed, 'method': method, **row}, flush=True)
    write_json(status, {'status': 'completed', 'epochs': 24, 'optimizer_steps': 6000, 'seconds': time.time()-started,
                       'parameters': payload['trainable_parameters'], 'checkpoint_sha256': sha256(checkpoint),
                       'reference_checkpoints_unchanged': True})


def logits_all(suite, stage, seed, records):
    z = {v: records[v]['candidate_logits'] for v in ['new', 'old']}
    if stage == 'targeted':
        z.update({m: predict_copy(suite, seed, m, records['new']) for m in METHODS})
    return z


def measure(suite, stage, seed, report=False):
    output = suite/stage/f'seed{seed}'
    output.mkdir(parents=True, exist_ok=True)
    if not report:
        cal = {v: cache(suite, stage, seed, v, 'calibration') for v in ['new', 'old']}
        np.testing.assert_array_equal(cal['new']['identities'], cal['old']['identities'])
        z = logits_all(suite, stage, seed, cal)
        refs = references(suite, seed)
        locked = {m: refs['old' if m == 'old' else 'new']['thresholds'] for m in z}
        decisions = fit_policies(cal['new']['targets'], z, cal['new']['weak_mask'], locked)
        write_json(output/'decisions.json', decisions)
        roles = ['calibration', 'check']
    else:
        decisions = read_json(output/'decisions.json')
        roles = ['development_test'] + (['ood_wide', 'pair_report'] if stage == 'targeted' else [])
    for role in roles:
        records = {v: cache(suite, stage, seed, v, 'development_test' if role == 'pair_report' else role, pair=role == 'pair_report') for v in ['new', 'old']}
        np.testing.assert_array_equal(records['new']['identities'], records['old']['identities'])
        z = logits_all(suite, stage, seed, records)
        common = records['new']
        metrics = {}
        for model, values in z.items():
            metrics[model] = {rule: metric(common, values, th) for rule, th in decisions['models'][model]['thresholds'].items() if th is not None}
        fields = {k: common[k] for k in ['targets', 'identities', 'indices', 'weak_mask', 'relative_power_db']}
        if role == 'pair_report':
            fields.update({k: common[k] for k in ['pair_id', 'pair_target', 'target_gap_bin', 'pair_side', 'edge_id']})
        np.savez_compressed(output/('predictions_'+role+'.npz'), **fields, **{'logits_'+m: values for m, values in z.items()})
        write_json(output/('metrics_'+role+'.json'), {'stage': stage, 'seed': seed, 'role': role, 'metrics': metrics})


def worker(suite, stage, seed, report=False):
    if not report and stage == 'targeted':
        fit = cache(suite, stage, seed, 'new', 'fit')
        check = cache(suite, stage, seed, 'new', 'check')
        pairs = cache(suite, stage, seed, 'new', 'fit', pair=True)
        for method in METHODS:
            train_copy(suite, seed, method, fit, check, pairs)
    measure(suite, stage, seed, report)


def mean_metrics(rows):
    out = {k: float(np.mean([r[k] for r in rows])) for k in KEYS}
    out['per_label'] = {c: {'negative_fpr': float(np.mean([r['per_label'][c]['negative_fpr'] for r in rows]))} for c in CLASSES}
    return out


def guard(candidate, anchor, fpr=.01, weak=.01):
    return {'exact_noninferior': candidate['exact'] >= anchor['exact']-.002,
            'saw_weak': candidate['saw_weak_recall'] >= anchor['saw_weak_recall']-weak,
            'mft_weak': candidate['mft_weak_recall'] >= anchor['mft_weak_recall']-weak,
            'class_fpr': max(candidate['per_label'][c]['negative_fpr']-anchor['per_label'][c]['negative_fpr'] for c in CLASSES) <= fpr,
            'tail_f1': candidate['tail_f1'] >= anchor['tail_f1']-.003}


def collect(suite, stage, role):
    seeds = read_json(suite/'plan.json')['seeds']
    rows = [read_json(suite/stage/f'seed{s}'/('metrics_'+role+'.json'))['metrics'] for s in seeds]
    available = {m: sorted(set.intersection(*(set(r[m]) for r in rows))) for m in rows[0]}
    means = {m: {rule: mean_metrics([r[m][rule] for r in rows]) for rule in available[m]} for m in available}
    return rows, means


def select(suite, stage):
    rows, means = collect(suite, stage, 'check')
    if stage == 'calibration':
        common = set(means['new']) & set(means['old'])
        eligibility, scores = {}, {}
        for rule in common:
            checks = {m: guard(means[m][rule], means[m]['locked']) for m in ['new', 'old']}
            eligibility[rule] = checks
            if all(all(c.values()) for c in checks.values()):
                scores[rule] = float(np.mean([means[m][rule]['exact'] for m in ['new', 'old']]))
        winner = max(scores, key=lambda r: (scores[r], r == 'locked'))
        result = {'selected_rule': winner, 'selected': winner, 'check_scores': scores,
                  'eligibility': eligibility, 'check_means': means, 'model': 'shared operating rule for new and old fixed weights'}
    else:
        desired = read_json(suite/'calibration/selection.json')['selected_rule']
        feasible_reference = desired in means['new'] and desired in means['old']
        rule = desired if feasible_reference else 'f1_8064'
        eligibility = {m: guard(means[m][rule], means['new'][rule]) for m in ['new']+list(METHODS) if rule in means[m]}
        scores = {m: means[m][rule]['exact'] for m in eligibility if all(eligibility[m].values())}
        winner = max(scores, key=lambda m: (scores[m], m == 'new'))
        result = {'selected': winner, 'selected_rule': rule, 'desired_rule': desired,
                  'global_rule_fallback': not feasible_reference, 'eligibility': eligibility, 'check_scores': scores,
                  'check_means': means, 'baseline': 'new immutable candidate at same rule',
                  'infeasible_candidates_excluded': [m for m in METHODS if rule not in means[m]]}
    result.update(selected_on='check only', report_values_used=False, selected_at=now())
    write_json(suite/stage/'selection.json', result)
    return result


def freeze(suite, stage):
    plan = read_json(suite/'plan.json')
    files = {}
    for refs in plan['references'].values():
        for version in ['new', 'old']:
            for kind in ['B', 'linear', 'local']:
                ref = refs[version][kind]
                assert sha256(ref['path']) == ref['sha256']
                files[ref['path']] = ref['sha256']
        ref = refs['FMB']
        assert sha256(ref['path']) == ref['sha256']
        files[ref['path']] = ref['sha256']
    for path in (suite/stage).rglob('*'):
        if path.name in ['decisions.json', 'selection.json'] or path.suffix == '.pt' and not path.name.startswith('last_'):
            files[str(path)] = sha256(path)
    write_json(suite/stage/'frozen.json', {'plan_sha256': sha256(suite/'plan.json'), 'files': files,
               'frozen_at': now(), 'all_weights_and_decisions_frozen_before_report': True})


def comparison(rows, candidate, baseline):
    a, arule = candidate
    b, brule = baseline
    ca, cb = [r[a][arule] for r in rows], [r[b][brule] for r in rows]
    delta = {k: contrast([r[k] for r in ca], [r[k] for r in cb]) for k in KEYS}
    guards = guard(mean_metrics(ca), mean_metrics(cb))
    exact = delta['exact']
    primary = {'exact_gain_at_least_half_pp': exact['mean'] >= .005, 'four_positive_seeds': exact['positive_seeds'] >= 4,
               'paired95_lower_positive': exact['paired95'][0] > 0, **guards}
    tradeoff = {'focus_weak_gain_at_least_two_pp': delta['focus_weak_recall']['mean'] >= .02,
                'weak_paired95_lower_positive': delta['focus_weak_recall']['paired95'][0] > 0,
                **guard(mean_metrics(ca), mean_metrics(cb), fpr=.005)}
    return {'candidate': candidate, 'baseline': baseline, 'deltas': delta, 'exact_success': primary,
            'exact_passed': all(primary.values()), 'weak_tradeoff_success': tradeoff, 'weak_tradeoff_passed': all(tradeoff.values())}


def summarize(suite, stage):
    rows, means = collect(suite, stage, 'development_test')
    selection = read_json(suite/stage/'selection.json')
    rule = selection['selected_rule']
    comparisons = {}
    if stage == 'calibration':
        for m in ['new', 'old']:
            comparisons[m+'_selected_vs_locked'] = comparison(rows, (m, rule), (m, 'locked'))
            comparisons[m+'_large_vs_small'] = comparison(rows, (m, 'f1_8064'), (m, 'f1_1008'))
        comparisons['new_vs_old_selected_rule'] = comparison(rows, ('new', rule), ('old', rule))
    else:
        winner = selection['selected']
        comparisons['primary_selected_vs_frozen_new'] = comparison(rows, (winner, rule), ('new', rule))
        for method in METHODS:
            if rule in means[method]:
                comparisons[method+'_vs_frozen_new'] = comparison(rows, (method, rule), ('new', rule))
                if method != 'natural' and rule in means['natural']:
                    comparisons[method+'_vs_natural_continuation'] = comparison(rows, (method, rule), ('natural', rule))
        comparisons['selected_vs_frozen_old'] = comparison(rows, (winner, rule), ('old', rule))
    result = {'stage': stage, 'selection': selection, 'report_means': means, 'comparisons': comparisons,
              'conditional_intervals': 'five paired model seeds on fixed pools; exploratory comparisons not multiplicity corrected'}
    if stage == 'targeted':
        result['supplementary'] = {role: collect(suite, stage, role)[1] for role in ['ood_wide', 'pair_report']}
    write_json(suite/stage/'summary.json', result)
    return result
