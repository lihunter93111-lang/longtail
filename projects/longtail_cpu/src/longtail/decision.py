"""Small, predeclared joint threshold search using calibration identities only."""
import numpy as np
from .metrics import probabilities, fit_thresholds


def fit_fpr_caps(targets, logits, allowed_false_positives):
    """Lowest observed-negative threshold under a calibration-only FP budget.

    Score ties can make the cap conservative. Saturated probabilities at one
    may make it infeasible; report the achieved count instead of hiding this.
    """
    y = np.asarray(targets, dtype=bool); p = probabilities(logits)
    counts = np.asarray(allowed_false_positives)
    if y.shape != p.shape or counts.shape != (y.shape[1],) or not np.isfinite(p).all():
        raise ValueError('Invalid FPR calibration arrays')
    thresholds, achieved, negative_counts = [], [], []
    for c, count in enumerate(counts):
        neg = np.sort(p[~y[:, c], c])[::-1]
        if not len(neg) or count < 0 or count != int(count):
            raise ValueError('Invalid calibration FP budget')
        threshold = 0. if count >= len(neg) else min(float(np.nextafter(neg[int(count)], np.inf)), 1.)
        thresholds.append(threshold); achieved.append(int((p[~y[:, c], c] >= threshold).sum()))
        negative_counts.append(len(neg))
    return thresholds, {'fit_role': 'calibration', 'allowed_false_positives': counts.astype(int).tolist(),
                        'achieved_false_positives': achieved, 'negative_counts': negative_counts,
                        'empirical_caps_satisfied': [bool(a <= b) for a, b in zip(achieved, counts)]}


def constrained_exact_thresholds(targets, logits, recall_drop=0.03, fpr_rise=0.02,
                                 grid=None, max_passes=4):
    """Coordinate ascent, not a claim of globally optimal Exact thresholds.

    Constraints refer to calibration F1 decisions for each label. Lexicographic
    ties prefer thresholds close to 0.5. Both starts and the grid are fixed;
    neither check nor reporting labels enter this function.
    """
    y = np.asarray(targets, dtype=bool)
    p = probabilities(logits)
    if y.shape != p.shape or not np.isfinite(p).all() or not len(y):
        raise ValueError('Invalid calibration arrays')
    pos, neg = y.sum(0), (~y).sum(0)
    if (pos == 0).any() or (neg == 0).any() or recall_drop < 0 or fpr_rise < 0 or max_passes < 1:
        raise ValueError('Calibration needs positive/negative examples and valid constraints')
    anchor = np.asarray(fit_thresholds(y, logits, None))
    base = p >= anchor
    base_recall = (base & y).sum(0) / pos
    base_fpr = (base & ~y).sum(0) / neg
    grid = np.linspace(.2, .8, 13) if grid is None else np.asarray(grid, dtype=float)
    if not np.isfinite(grid).all() or ((grid <= 0) | (grid >= 1)).any():
        raise ValueError('Threshold grid must be finite and inside (0,1)')

    def distance(values):
        clipped = np.clip(values, 1e-9, 1 - 1e-9)
        return np.abs(np.log(clipped / (1 - clipped)))

    candidates, predictions = [], []
    for c in range(y.shape[1]):
        values = np.unique(np.r_[grid, .5, anchor[c]])
        pred = p[:, c, None] >= values
        recall = (pred & y[:, c, None]).sum(0) / pos[c]
        fpr = (pred & ~y[:, c, None]).sum(0) / neg[c]
        feasible = (recall >= base_recall[c] - recall_drop - 1e-12) & (fpr <= base_fpr[c] + fpr_rise + 1e-12)
        candidates.append(values[feasible])
        predictions.append(pred[:, feasible])
    projected = np.array([v[np.argmin(distance(v))] for v in candidates])
    solutions = []
    for start in [anchor, projected]:
        thresholds = start.copy()
        pred = p >= thresholds
        trace = [int(np.all(pred == y, axis=1).sum())]
        for _ in range(max_passes):
            previous = thresholds.copy()
            for c in range(y.shape[1]):
                others = np.delete(pred == y, c, axis=1).all(1)
                exact = ((predictions[c] == y[:, c, None]) & others[:, None]).sum(0)
                best = np.lexsort((candidates[c], distance(candidates[c]), -exact))[0]
                thresholds[c] = candidates[c][best]
                pred[:, c] = predictions[c][:, best]
            trace.append(int(np.all(pred == y, axis=1).sum()))
            if np.array_equal(previous, thresholds):
                break
        solutions.append((trace[-1], float(distance(thresholds).sum()), thresholds, trace))
    winner = min(solutions, key=lambda v: (-v[0], v[1], tuple(v[2])))
    thresholds = winner[2]
    pred = p >= thresholds
    return thresholds.tolist(), {'method': 'constrained_exact_coordinate_v1', 'fit_role': 'calibration',
        'grid': grid.tolist(), 'starts': ['f1_exact', 'feasible_nearest_0.5'], 'max_passes': max_passes,
        'recall_drop_max_vs_f1': recall_drop, 'fpr_rise_max_vs_f1': fpr_rise,
        'tie_break': 'minimum absolute threshold log-odds, then lower threshold',
        'calibration_samples': len(y), 'calibration_exact_f1': float(np.all(base == y, axis=1).mean()),
        'calibration_exact_fitted': winner[0] / len(y), 'search_exact_count_traces': [s[3] for s in solutions],
        'reference_recall': base_recall.tolist(), 'reference_fpr': base_fpr.tolist(),
        'fitted_recall': ((pred & y).sum(0) / pos).tolist(), 'fitted_fpr': ((pred & ~y).sum(0) / neg).tolist()}
