"""Two-label operating points; the other four thresholds stay explicitly locked."""
import numpy as np
from .metrics import fit_thresholds, probabilities
from .decision import fit_fpr_caps

TARGETS = [2, 4]
RULES = ['locked', 'f1_1008', 'f1_8064', 'fpr03', 'fpr05', 'fpr07', 'weak_guard']


def small_calibration_blocks(targets):
    code = np.asarray(targets, int) @ (1 << np.arange(6))
    blocks = [[] for _ in range(8)]
    for c in range(1, 64):
        rows = np.flatnonzero(code == c)
        if len(rows) != 128:
            raise ValueError('Expected128 calibration identities per combination')
        for i in range(8):
            blocks[i].extend(rows[16*i:16*(i+1)].tolist())
    return [np.asarray(sorted(v), np.int64) for v in blocks]


def target_f1(y, z, locked):
    out = np.array(locked, float).copy()
    out[TARGETS] = fit_thresholds(np.asarray(y)[:, TARGETS], np.asarray(z)[:, TARGETS], None)
    return out.tolist()


def two_target_exact(targets, logits, weak, locked, reference, max_fpr, weak_floor, recall_drop=.01):
    """Calibration-only coordinate ascent with explicit infeasibility, no relaxation."""
    y = np.asarray(targets, bool)
    p = probabilities(logits)
    weak = np.asarray(weak, bool)
    reference = np.asarray(reference, float)
    if p.shape != y.shape or weak.shape != y.shape or np.any(weak & ~y):
        raise ValueError('Invalid calibration truth')
    ref_pred = p >= reference
    candidates, predictions = {}, {}
    details, infeasible = {}, []
    for c in TARGETS:
        pos, neg, weak_count = y[:, c].sum(), (~y[:, c]).sum(), weak[:, c].sum()
        if min(pos, neg, weak_count) == 0:
            raise ValueError('Target calibration requires positives, negatives and weak positives')
        values = np.unique(np.r_[p[:, c], reference[c], locked[c], .5, 1.])
        pred = p[:, c, None] >= values
        fpr = (pred & ~y[:, c, None]).sum(0)/neg
        recall = (pred & y[:, c, None]).sum(0)/pos
        wr = (pred & weak[:, c, None]).sum(0)/weak_count
        recall_floor = float((ref_pred[:, c] & y[:, c]).sum()/pos-recall_drop)
        valid = (fpr <= max_fpr[c]+1e-12) & (wr >= weak_floor[c]-1e-12) & (recall >= recall_floor-1e-12)
        candidates[c], predictions[c] = values[valid], pred[:, valid]
        details[str(c)] = {'max_fpr': float(max_fpr[c]), 'weak_floor': float(weak_floor[c]),
                           'recall_floor': recall_floor, 'feasible_thresholds': int(valid.sum())}
        if not valid.any():
            infeasible.append(c)
    meta = {'feasible': not infeasible, 'infeasible_labels': infeasible, 'constraints': details,
            'fit_role': 'calibration', 'objective': 'six-label Exact; only SawFM/MFT thresholds change',
            'search': 'two starts, up to6 coordinate passes; no claim of global optimum'}
    if infeasible:
        return None, meta
    solutions = []
    for origin in [reference, np.full(6, .5)]:
        th = np.array(locked, float).copy()
        for c in TARGETS:
            th[c] = candidates[c][np.argmin(np.abs(candidates[c]-origin[c]))]
        pred = p >= th
        trace = [int((pred == y).all(1).sum())]
        for _ in range(6):
            before = th.copy()
            for c in TARGETS:
                others = np.delete(pred == y, c, axis=1).all(1)
                exact = ((predictions[c] == y[:, c, None]) & others[:, None]).sum(0)
                best = np.lexsort((candidates[c], np.abs(candidates[c]-reference[c]), -exact))[0]
                th[c] = candidates[c][best]
                pred[:, c] = predictions[c][:, best]
            trace.append(int((pred == y).all(1).sum()))
            if np.array_equal(before, th):
                break
        solutions.append((trace[-1], float(np.abs(th[TARGETS]-reference[TARGETS]).sum()), th, trace))
    winner = min(solutions, key=lambda s: (-s[0], s[1], tuple(s[2])))
    meta.update(calibration_exact=winner[0]/len(y), traces=[s[3] for s in solutions])
    return winner[2].tolist(), meta


def fit_policies(targets, logits_by_model, weak, locked_by_model):
    y = np.asarray(targets, bool)
    weak = np.asarray(weak, bool)
    blocks = small_calibration_blocks(y)
    f1 = {m: target_f1(y, z, locked_by_model[m]) for m, z in logits_by_model.items()}
    # Shared constraints are defined only by the two immutable references.
    cap = np.ones(6)
    floor = np.zeros(6)
    for c in TARGETS:
        fp, recall = [], []
        for m in ['new', 'old']:
            pp = probabilities(logits_by_model[m])[:, c] >= f1[m][c]
            fp.append(float(pp[~y[:, c]].mean()))
            recall.append(float(pp[weak[:, c]].mean()))
        cap[c] = min(1., max(fp)+.005)
        floor[c] = max(0., max(recall)-.01)
    models = {}
    for m, z in logits_by_model.items():
        locked = np.asarray(locked_by_model[m], float)
        small = [target_f1(y[block], z[block], locked) for block in blocks]
        thresholds = {'locked': locked.tolist(), 'f1_1008': small[0], 'f1_8064': f1[m]}
        metadata = {}
        for name, rate in [('fpr03', .03), ('fpr05', .05), ('fpr07', .07)]:
            counts = np.floor((~y).sum(0)*rate).astype(int)
            proposed, meta = fit_fpr_caps(y, z, counts)
            th = locked.copy()
            th[TARGETS] = np.asarray(proposed)[TARGETS]
            feasible = all(meta['empirical_caps_satisfied'][c] for c in TARGETS)
            thresholds[name] = th.tolist() if feasible else None
            metadata[name] = {'nominal_target_fpr': rate, **meta,
                              'target_caps_satisfied': feasible,
                              'scope': 'Only target calibration FP counts capped; actual check/report FPR may differ.'}
        thresholds['weak_guard'], metadata['weak_guard'] = two_target_exact(y, z, weak, locked, f1[m], cap, floor)
        for th in thresholds.values():
            if th is not None:
                np.testing.assert_array_equal(np.asarray(th)[[0,1,3,5]], locked[[0,1,3,5]])
        models[m] = {'thresholds': thresholds, 'metadata': metadata, 'small_block_thresholds': small}
    return {'fit_role': 'calibration only', 'target_columns': TARGETS, 'models': models,
            'shared_max_fpr': cap.tolist(), 'shared_min_weak_recall': floor.tolist(),
            'small_block_indices': [v.tolist() for v in blocks], 'primary_small_block': 0,
            'infeasibility_policy': 'No relaxed per-seed fallback. Exclude non-common methods; stage2 reference infeasibility falls back to f1_8064 for the whole comparison.'}
