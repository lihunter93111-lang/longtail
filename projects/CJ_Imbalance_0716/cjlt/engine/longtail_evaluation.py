"""AMA multilabel metrics and calibration with explicit data roles."""
import numpy as np
from scipy.special import expit
from sklearn.metrics import average_precision_score, roc_auc_score
from scipy.stats import t as student_t
from cjlt.models.classwise_fusion import AMA_CLASSES


def fit_thresholds(targets, logits, source_role="calibration", fpr_caps=None, grid=None):
    if source_role != "calibration":
        raise ValueError("Thresholds may only be fitted on calibration identities")
    targets = np.asarray(targets, dtype=bool)
    probabilities = expit(np.asarray(logits, dtype=np.float64))
    thresholds = []
    for column in range(6):
        if grid is not None:
            candidates = np.asarray(grid, dtype=np.float64)
            if candidates.ndim != 1 or not len(candidates) or not np.isfinite(candidates).all() or np.any((candidates < 0) | (candidates > 1)):
                raise ValueError("Invalid threshold grid")
            prediction = probabilities[:, column, None] >= candidates
            truth = targets[:, column, None]
            tp = (prediction & truth).sum(0)
            fp = (prediction & ~truth).sum(0)
            f1 = 2 * tp / np.maximum(truth.sum() + prediction.sum(0), 1)
            allowed = np.ones(len(candidates), dtype=bool) if fpr_caps is None else fp / max(int((~truth).sum()), 1) <= fpr_caps[column] + 1e-12
            thresholds.append(float(candidates[np.argmax(np.where(allowed, f1, -1))]) if allowed.any() else 1.0)
            continue
        order = np.argsort(-probabilities[:, column], kind="stable")
        scores = probabilities[order, column]
        truth = targets[order, column]
        ends = np.flatnonzero(np.r_[scores[1:] != scores[:-1], True])
        tp = np.cumsum(truth)[ends]
        f1 = 2 * tp / np.maximum(truth.sum() + ends + 1, 1)
        allowed = np.ones(len(ends), dtype=bool)
        if fpr_caps is not None:
            fp = ends + 1 - tp
            allowed = fp / max((~truth).sum(), 1) <= fpr_caps[column] + 1e-12
        if not allowed.any():
            thresholds.append(1.0)
        else:
            best = np.flatnonzero(allowed & (f1 == f1[allowed].max()))[-1]
            thresholds.append(float(scores[ends[best]]))
    return thresholds


def evaluate_predictions(targets, logits, thresholds, weak_mask=None, relative_power_db=None):
    targets = np.asarray(targets, dtype=bool)
    logits = np.asarray(logits, dtype=np.float64)
    probabilities = expit(logits)
    thresholds = np.asarray(thresholds, dtype=np.float64)
    if logits.shape != targets.shape or targets.ndim != 2 or targets.shape[1] != 6 or thresholds.shape != (6,):
        raise ValueError("Expected matching six-label predictions and thresholds")
    if not np.isfinite(logits).all() or not np.isfinite(thresholds).all() or np.any((thresholds < 0) | (thresholds > 1)):
        raise ValueError("Invalid scores or thresholds")
    prediction = probabilities >= thresholds
    exact = (prediction == targets).all(1)
    result = dict(samples=len(targets), exact=float(exact.mean()),
                  raw_bce=float((np.logaddexp(0, logits) - targets * logits).mean()),
                  hamming_loss=float((prediction != targets).mean()), thresholds=thresholds.tolist(), per_label={})
    if weak_mask is not None:
        weak_mask = np.asarray(weak_mask, dtype=bool)
        if weak_mask.shape != targets.shape or np.any(weak_mask & ~targets):
            raise ValueError("Weak positives must be a subset of true positives")
    for column, name in enumerate(AMA_CLASSES):
        truth = targets[:, column]
        predicted = prediction[:, column]
        tp = int((truth & predicted).sum())
        fp = int((~truth & predicted).sum())
        fn = int((truth & ~predicted).sum())
        row = dict(tp=tp, fp=fp, fn=fn, recall=tp / max(int(truth.sum()), 1),
                   f1=2 * tp / max(2 * tp + fp + fn, 1), negative_fpr=fp / max(int((~truth).sum()), 1),
                   aupr=float(average_precision_score(truth, probabilities[:, column])) if truth.any() else None)
        if weak_mask is not None:
            weak = weak_mask[:, column]
            selected = weak | ~truth
            row.update(weak_recall=float(predicted[weak].mean()) if weak.any() else None,
                       weak_aupr=float(average_precision_score(weak[selected], probabilities[selected, column])) if weak.any() else None,
                       weak_auroc=float(roc_auc_score(weak[selected], probabilities[selected, column])) if weak.any() and (~truth).any() else None)
        result["per_label"][name] = row
    result["macro_f1"] = float(np.mean([row["f1"] for row in result["per_label"].values()]))
    result["macro_aupr"] = float(np.mean([row["aupr"] for row in result["per_label"].values() if row["aupr"] is not None]))
    codes = targets.astype(np.int64) @ (1 << np.arange(6))
    result["per_combination"] = {str(code): dict(count=int((codes == code).sum()), exact=float(exact[codes == code].mean()))
                                 for code in np.unique(codes)}
    result["per_cardinality"] = {str(count): dict(count=int((targets.sum(1) == count).sum()),
                                                exact=float(exact[targets.sum(1) == count].mean()))
                                 for count in np.unique(targets.sum(1))}
    if relative_power_db is not None:
        gap = -np.asarray(relative_power_db)
        result["power_gap_bins"] = []
        for low, high in ((0, 6), (6, 10), (10, 15), (15, 20), (20, 30), (30, float("inf"))):
            mask = targets & (gap >= low - 1e-6) & (gap < high - 1e-6)
            result["power_gap_bins"].append(dict(low_db=low, high_db=high if np.isfinite(high) else None,
                                                count=int(mask.sum()), recall=float(prediction[mask].mean()) if mask.any() else None))
    return result


def preserves_reference(candidate, reference, targets, weak_drop=0.01, fpr_rise=0.005, ap_drop=0.002):
    for name in AMA_CLASSES:
        if candidate["per_label"][name]["negative_fpr"] > reference["per_label"][name]["negative_fpr"] + fpr_rise:
            return False
    for name in targets:
        left, right = candidate["per_label"][name], reference["per_label"][name]
        if right.get("weak_recall") is not None and left["weak_recall"] < right["weak_recall"] - weak_drop:
            return False
    valid = [name for name in targets if reference["per_label"][name].get("weak_aupr") is not None]
    if valid and np.mean([candidate["per_label"][name]["weak_aupr"] for name in valid]) < np.mean([
            reference["per_label"][name]["weak_aupr"] for name in valid]) - ap_drop:
        return False
    # AMA/SinFM are the declared tail labels of this protocol.
    tail = lambda row: np.mean([row["per_label"][name]["f1"] for name in ("AMA", "SinFM")])
    return bool(tail(candidate) >= tail(reference) - 0.003)


def compare_predictions(candidate_paths, reference_paths):
    """Paired seed intervals from saved, identity-aligned frozen predictions."""
    if not candidate_paths or set(candidate_paths) != set(reference_paths):
        raise ValueError("Comparison requires the same nonempty seed set")
    exact, bce, corrected, harmed = [], [], [], []
    for seed in sorted(candidate_paths):
        with np.load(candidate_paths[seed], allow_pickle=False) as left, \
             np.load(reference_paths[seed], allow_pickle=False) as right:
            for key in ("identity", "target"):
                if not np.array_equal(left[key], right[key]):
                    raise ValueError("Paired predictions differ in {} for seed {}".format(key, seed))
            if len(np.unique(left["identity"])) != len(left["identity"]):
                raise ValueError("Comparison identities must be unique")
            a, b = [((expit(values["logits"].astype(np.float64)) >= values["thresholds"]) == values["target"]).all(1)
                    for values in (left, right)]
            exact.append(float(a.mean() - b.mean()))
            losses = [(np.logaddexp(0, values["logits"].astype(np.float64)) - values["target"] * values["logits"]).mean()
                      for values in (left, right)]
            bce.append(float(losses[0] - losses[1]))
            corrected.append(int((a & ~b).sum()))
            harmed.append(int((~a & b).sum()))
    result = dict(seeds=sorted(candidate_paths), corrected=corrected, harmed=harmed)
    for key, values in (("exact_delta", exact), ("raw_bce_delta", bce)):
        mean = float(np.mean(values))
        radius = float(student_t.ppf(0.975, len(values) - 1) * np.std(values, ddof=1) / np.sqrt(len(values))) if len(values) > 1 else None
        result[key] = dict(mean=mean, values=values, t95=None if radius is None else [mean - radius, mean + radius])
    return result
