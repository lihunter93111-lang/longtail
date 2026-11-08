"""Multi-label, combination and probability metrics shared by train and eval."""
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from .data.splits import codes


def probabilities(logits):
    return 1 / (1 + np.exp(-np.clip(np.asarray(logits, dtype=np.float64), -40, 40)))


def fit_thresholds(targets, logits, grid):
    p = probabilities(logits)
    thresholds = []
    candidates = np.linspace(*grid) if grid is not None else None
    for column in range(targets.shape[1]):
        if grid is None:
            # Search all observable operating points, including tied scores.
            # Choose the lowest threshold on an F1 tie, matching grid behavior.
            order = np.argsort(-p[:, column], kind='stable')
            scores = p[order, column]
            truth = np.asarray(targets)[order, column].astype(bool)
            ends = np.flatnonzero(np.r_[scores[1:] != scores[:-1], True])
            tp = np.cumsum(truth)[ends]
            denominator = truth.sum() + ends + 1
            f1 = np.divide(2 * tp, denominator, out=np.zeros_like(tp, dtype=float), where=denominator > 0)
            best = np.flatnonzero(f1 == f1.max())[-1]
            thresholds.append(float(scores[ends[best]]))
            continue
        truth = targets[:, column:column + 1].astype(bool)
        pred = p[:, column:column + 1] >= candidates[None, :]
        tp = (pred & truth).sum(0)
        denominator = truth.sum() + pred.sum(0)
        scores = np.divide(2 * tp, denominator, out=np.zeros_like(tp, dtype=float), where=denominator > 0)
        thresholds.append(float(candidates[int(np.argmax(scores))]))
    return thresholds


def _mean(values):
    valid = [v for v in values if v is not None]
    return float(np.mean(valid)) if valid else None


def evaluate_metrics(targets, logits, classes, thresholds=None, combination_counts=None, ece_bins=15):
    y = np.asarray(targets, dtype=np.uint8)
    p = probabilities(logits)
    thresholds = np.full(y.shape[1], 0.5) if thresholds is None else np.asarray(thresholds)
    if y.shape != p.shape or len(classes) != y.shape[1] or thresholds.shape != (y.shape[1],):
        raise ValueError('Invalid metric shapes')
    if not np.isfinite(p).all() or not np.isfinite(thresholds).all() or not ((thresholds >= 0) & (thresholds <= 1)).all():
        raise ValueError('Invalid probabilities or thresholds')
    pred = p >= thresholds
    exact = np.all(y == pred, axis=1)
    per_label = {}
    for index, name in enumerate(classes):
        true = y[:, index].astype(bool)
        positive = int(true.sum())
        tp = int((true & pred[:, index]).sum())
        fp = int((~true & pred[:, index]).sum())
        fn = positive - tp
        ece = 0.0
        for lo, hi in zip(np.linspace(0, 1, ece_bins + 1)[:-1], np.linspace(0, 1, ece_bins + 1)[1:]):
            mask = (p[:, index] >= lo) & ((p[:, index] < hi) if hi < 1 else p[:, index] <= hi)
            if mask.any():
                ece += float(mask.mean() * abs(p[mask, index].mean() - true[mask].mean()))
        per_label[name] = {'positive_count': positive, 'tp': tp, 'fp': fp, 'fn': fn,
                           'precision': tp / max(tp + fp, 1), 'recall': tp / max(positive, 1),
                           'f1': 2 * tp / max(2 * tp + fp + fn, 1),
                           'negative_fpr': fp / max(len(y) - positive, 1),
                           'aupr': float(average_precision_score(true, p[:, index])) if positive else None,
                           'auroc': float(roc_auc_score(true, p[:, index])) if 0 < positive < len(y) else None,
                           'ece': ece, 'brier': float(np.mean((p[:, index] - true) ** 2))}
    combo = codes(y)
    per_combo = [{'code': int(code), 'count': int((combo == code).sum()),
                  'exact_accuracy': float(exact[combo == code].mean())} for code in np.unique(combo)]
    result = {'samples': len(y), 'subset_accuracy': float(exact.mean()),
              'macro_combination_accuracy': float(np.mean([v['exact_accuracy'] for v in per_combo])),
              'combination_coverage': len(per_combo), 'hamming_loss': float(np.mean(y != pred)),
              **{'macro_' + key: _mean([v[key] for v in per_label.values()]) for key in ('f1', 'aupr', 'auroc')},
              'ece': _mean([v['ece'] for v in per_label.values()]),
              'brier': _mean([v['brier'] for v in per_label.values()]),
              'thresholds': thresholds.tolist(), 'per_label': per_label, 'per_combination': per_combo,
              'cardinality': {str(k): {'count': int((y.sum(1) == k).sum()),
                                     'exact_accuracy': float(exact[y.sum(1) == k].mean())} for k in np.unique(y.sum(1))}}
    if combination_counts is not None:
        order = np.lexsort((np.arange(1, 64), combination_counts)) + 1
        for name, selected in [('rare', order[:13]), ('common', order[-13:])]:
            mask = np.isin(combo, selected)
            result[name + '_combination_codes'] = selected.tolist()
            result[name + '_combination_count'] = int(mask.sum())
            result[name + '_combination_accuracy'] = float(exact[mask].mean()) if mask.any() else None
    return result
