"""Weak-positive ranking against true negatives; strong positives are excluded."""
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from .metrics import probabilities


def weak_metrics(targets, logits, classes, thresholds, weak_mask=None, relative_power_db=None, label_counts=None):
    if weak_mask is None:
        return {'available': False, 'reason': 'No verified per-component power truth; sample-mean nominal JNR is insufficient.'}
    y = np.asarray(targets, dtype=bool)
    weak = np.asarray(weak_mask, dtype=bool)
    if weak.shape != y.shape or np.any(weak & ~y):
        raise ValueError('Weak positives must be a subset of true positives')
    p = probabilities(logits)
    pred = p >= np.asarray(thresholds)
    per_label = {}
    for i, name in enumerate(classes):
        selected = weak[:, i] | ~y[:, i]
        truth = weak[selected, i]
        support = int(weak[:, i].sum())
        per_label[name] = {
            'weak_positive_count': support,
            'weak_recall': float(pred[weak[:, i], i].mean()) if support else None,
            'weak_aupr': float(average_precision_score(truth, p[selected, i])) if support else None,
            'weak_auroc': float(roc_auc_score(truth, p[selected, i])) if support and (~y[:, i]).any() else None,
            'negative_fpr': float(pred[~y[:, i], i].mean()) if (~y[:, i]).any() else None}
    valid = [v['weak_recall'] for v in per_label.values() if v['weak_recall'] is not None]
    valid_aupr = [v['weak_aupr'] for v in per_label.values() if v['weak_aupr'] is not None]
    result = {'available': True, 'weak_positive_count': int(weak.sum()),
              'weak_micro_recall': float(pred[weak].mean()) if weak.any() else None,
              'weak_macro_recall': float(np.mean(valid)) if valid else None,
              'weak_macro_aupr': float(np.mean(valid_aupr)) if valid_aupr else None,
              'per_label': per_label}
    if label_counts is not None:
        counts = np.asarray(label_counts)
        if counts.shape != (len(classes),) or (counts < 0).any():
            raise ValueError('Invalid training label counts')
        tail = (np.lexsort((np.arange(len(classes)), counts))[:max(1, len(classes) // 3)]
                if counts.max() > counts.min() else np.array([], dtype=int))
        tail_mask = weak[:, tail]
        result['tail_labels'] = [classes[i] for i in tail]
        result['tail_weak_positive_count'] = int(tail_mask.sum())
        result['tail_weak_recall'] = float(pred[:, tail][tail_mask].mean()) if tail_mask.any() else None
    if relative_power_db is not None:
        gap = -np.asarray(relative_power_db)
        if gap.shape != y.shape:
            raise ValueError('Relative-power shape mismatch')
        result['power_gap_bins'] = []
        for lo, hi in [(0, 6), (6, 10), (10, 15), (15, 20), (20, 30), (30, float('inf'))]:
            mask = y & (gap >= lo - 1e-6) & (gap < hi - 1e-6)
            result['power_gap_bins'].append({'lower_db': lo, 'upper_db': hi if np.isfinite(hi) else None,
                 'positive_count': int(mask.sum()), 'recall': float(pred[mask].mean()) if mask.any() else None})
    return result
