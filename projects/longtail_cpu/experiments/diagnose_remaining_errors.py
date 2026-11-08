"""Use consumed results to propose new hypotheses, never to claim new tests."""
import argparse
import json
from pathlib import Path
import numpy as np
from longtail.metrics import probabilities, evaluate_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', required=True, type=Path)
    args = parser.parse_args(); suite = args.suite.resolve()
    plan = json.loads((suite / 'plan.json').read_text())
    labels = ['FMB', 'AMA', 'SawFM', 'SinFM', 'MFT', 'ISRJ']
    routing, errors = [], []
    for seed in plan['protocol']['seeds']:
        base = suite / f'bce_seed{seed}'; mix = suite / f'mixed_sampler_seed{seed}'
        hroot = suite / f'hybrid_seed{seed}'
        t = json.loads((base / 'decisions_v2.json').read_text())['thresholds']['f1']; t[0] = .5
        with np.load(hroot / 'evaluation/check/predictions.npz') as f:
            h = {k: f[k] for k in ['targets', 'logits', 'identities']}
        with np.load(mix / 'evaluation/check/predictions.npz') as f:
            m = {k: f[k] for k in ['targets', 'logits', 'identities']}
        for k in ['targets', 'identities']:
            np.testing.assert_array_equal(h[k], m[k])
        th = t.copy(); th[2] = json.loads((mix / 'decisions_v2.json').read_text())['thresholds']['f1'][2]
        z = h['logits'].copy(); z[:, 2] = m['logits'][:, 2]
        old = evaluate_metrics(h['targets'], h['logits'], labels, t)
        new = evaluate_metrics(h['targets'], z, labels, th)
        routing.append({'seed': seed, 'two_expert_exact': old['subset_accuracy'], 'three_expert_exact': new['subset_accuracy'],
                        'two_expert_macro_f1': old['macro_f1'], 'three_expert_macro_f1': new['macro_f1']})
        with np.load(hroot / 'evaluation/development_test/predictions.npz') as f:
            y = f['targets'].astype(bool); p = probabilities(f['logits']) >= t
            weak = f['weak_mask'] & y
        fn = ~p & y; err = p != y; bad = err.any(1)
        errors.append({'seed': seed, 'weak_positive_share': float(weak.sum() / y.sum()),
            'weak_fn_share_all_fn': float((fn & weak).sum() / fn.sum()),
            'weak_recall': float(1 - (fn & weak).sum() / weak.sum()),
            'nonweak_recall': float(1 - (fn & ~weak).sum() / (y & ~weak).sum()),
            'only_weak_fn_share_bad_samples': float((bad & np.all(err == (fn & weak), 1)).sum() / bad.sum()),
            'any_weak_fn_share_bad_samples': float((fn & weak).any(1).sum() / bad.sum()),
            'label_error_share': (err.sum(0) / err.sum()).tolist(),
            'per_label_weak_recall': (1 - (fn & weak).sum(0) / weak.sum(0)).tolist(),
            'per_label_nonweak_recall': (1 - (fn & ~weak).sum(0) / (y & ~weak).sum(0)).tolist()})
    result = {'suite': str(suite), 'labels': labels,
        'purpose': 'Development diagnosis to design future experiments, using already consumed evidence.',
        'new_independent_test_claim': False,
        'weak_definition': 'Present component <= -10dB relative to strongest; saved generator truth, not inference input.',
        'three_expert_rule': 'FMB combo fixed0.5; SawFM mixed F1; other four BCE F1; check only.',
        'three_expert_check_rows': routing,
        'three_expert_check_means': {k: float(np.mean([r[k] for r in routing])) for k in routing[0] if k != 'seed'},
        'weak_diagnosis_role': 'consumed development_test, predeclared hybrid', 'weak_diagnosis_rows': errors,
        'weak_diagnosis_means': {k: np.mean([r[k] for r in errors], axis=0).tolist() for k in errors[0] if k != 'seed'},
        'next_primary_hypothesis': 'Preserve frozen combo FMB, improve the other-five expert using diverse and targeted weak-component fit identities.',
        'confirmation_requirement': 'Fixed methods/budgets and new reporting identities; no tuning against consumed report.'}
    out = Path(plan['report_output']) / 'next_validation_diagnosis.json'
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'output': str(out), 'three_expert_check_means': result['three_expert_check_means'],
                      'weak_diagnosis_means': result['weak_diagnosis_means']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
