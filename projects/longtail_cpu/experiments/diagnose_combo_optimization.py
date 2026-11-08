"""One fixed check-only diagnostic and analytic sampling budgets; no training."""
import argparse
from datetime import datetime
import hashlib
from html import escape
import json
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', required=True, type=Path)
    args = parser.parse_args()
    suite = args.suite.resolve()
    plan = read(suite / 'plan.json')
    out = Path(plan['report_output'])
    rows, provenance = [], []
    for seed in plan['protocol']['seeds']:
        loaded = {}
        for method in ['bce', 'combo_sampler']:
            run = suite / f'{method}_seed{seed}'
            predictions = run / 'evaluation/check/predictions.npz'
            with np.load(predictions) as z:
                loaded[method] = {k: z[k] for k in ['targets', 'identities', 'logits']}
            loaded[method]['decision'] = read(run / 'decision.json')
            provenance.append({'seed': seed, 'method': method, 'role': 'check',
                               'checkpoint_sha256': digest(run / 'best.pt'),
                               'predictions_sha256': digest(predictions),
                               'decision_sha256': digest(run / 'decision.json')})
        b, c = loaded['bce'], loaded['combo_sampler']
        np.testing.assert_array_equal(b['identities'], c['identities'])
        np.testing.assert_array_equal(b['targets'], c['targets'])
        truth = b['targets'].astype(bool)
        bp = 1 / (1 + np.exp(-np.clip(b['logits'].astype(float), -40, 40)))
        bpred = bp >= np.array(b['decision']['thresholds'])
        cpred = c['logits'] >= 0
        hybrid = bpred.copy()
        hybrid[:, 0] = cpred[:, 0]

        def metrics(pred):
            tp = (pred & truth).sum(0)
            fp = (pred & ~truth).sum(0)
            fn = (~pred & truth).sum(0)
            f1 = 2 * tp / (2 * tp + fp + fn)
            return {'exact': float(np.all(pred == truth, axis=1).mean()),
                    'macro_f1': float(f1.mean()), 'tail_f1': float(f1[[1, 3]].mean())}

        rows.append({'seed': seed, 'samples': len(truth), 'bce_calibrated': metrics(bpred),
                     'combo_fixed': metrics(cpred), 'hybrid': metrics(hybrid)})
    aggregate = {method: {metric: {'mean': float(np.mean([r[method][metric] for r in rows])),
                                  'std': float(np.std([r[method][metric] for r in rows], ddof=1))}
                          for metric in ['exact', 'macro_f1', 'tail_f1']}
                 for method in ['bce_calibrated', 'combo_fixed', 'hybrid']}
    split = read(suite / 'bce_seed2050/split.json')
    counts = np.array(split['fit_combination_counts'])
    combinations = ((np.arange(1, 64)[:, None] & (1 << np.arange(6))) > 0).astype(float)
    budget, k = int(counts.sum()), len(counts)
    expectations = []
    for fraction in [0, .25, .5, .75, 1]:
        combo_draws = int(fraction * budget)
        no_combo_draw = np.exp(combo_draws * np.log1p(-1 / (k * counts)))
        unique = budget - fraction * np.sum(counts * no_combo_draw)
        combo_distribution = (1 - fraction) * counts / budget + fraction / k
        expectations.append({'combo_fraction': fraction, 'natural_draws_without_replacement': budget - combo_draws,
                             'combo_draws_with_replacement': combo_draws,
                             'expected_unique': float(unique),
                             'expected_label_fractions': (combo_distribution @ combinations).tolist()})
    result = {'created': datetime.now().astimezone().isoformat(timespec='seconds'), 'suite': str(suite),
              'purpose': 'Develop next hypotheses using only saved check predictions; no fresh report evaluation.',
              'rule': 'FMB decision from combo model fixed0.5; other five from BCE frozen per-label F1 thresholds.',
              'role': 'check', 'threshold_fitting': 'none; reuse frozen decisions',
              'selection_caveat': 'Hypothesis informed by previously consumed report results; check is also used for epoch selection. This is exploratory, not independent confirmation.',
              'inference_cost': 'two frozen models per seed; no new model training',
              'rows': rows, 'aggregate': aggregate, 'input_provenance': provenance,
              'mixed_sampling_expectations': expectations,
              'sampling_caveat': 'Analytic expectations, not measured accuracy gains. Natural subset sampled without replacement, combined with independent combo replacement draws, then shuffled.',
              'next_protocol': 'Fix seed+epoch stream reuse; freeze methods and decisions on development, then evaluate new reporting identities once.'}
    (out / 'optimization_diagnosis.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    def table(headers, data):
        return '<table><tr>' + ''.join('<th>' + escape(str(v)) + '</th>' for v in headers) + '</tr>' + ''.join('<tr>' + ''.join('<td>' + escape(str(v)) + '</td>' for v in row) + '</tr>' for row in data) + '</table>'
    text = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>组合采样后续优化：开发诊断</title><style>body{font:16px/1.8 system-ui;max-width:1000px;margin:30px auto;padding:20px;color:#234}table{border-collapse:collapse;width:100%;font-size:14px}td,th{border:1px solid #ccd;padding:8px}th{background:#e7f2f3}.note{background:#fff4df;padding:16px}a{color:#16758a}</style>'
    text += '<h1>组合采样后续优化：固定模型互补与采样预算</h1><p class="note">仅使用已保存的check预测。规则根据已消费结果提出，check也参与过选模，因此属于探索性开发诊断，不能视为新测试成绩。未更新权重、未重新拟合阈值，也未评测新报告身份。</p>'
    text += '<p>固定融合规则：FMB采用组合采样模型的0.5决策，其余五类采用自然采样BCE模型的既有F1阈值。需要两个模型推理。五个种子均分别配对。</p>'
    text += table(['方案', 'check Exact %', 'check Macro-F1 %', 'check 尾类F1 %'], [[name] + [f"{aggregate[m][key]['mean']*100:.3f} ± {aggregate[m][key]['std']*100:.3f}" for key in ['exact', 'macro_f1', 'tail_f1']] for m, name in [('bce_calibrated', 'BCE+校准'), ('combo_fixed', '组合采样+0.5'), ('hybrid', '固定按类融合')]])
    text += '<p>该结果支持两类训练模型具有互补性，尚不能证明同一主干上增加两个分类头即可得到同样收益；两者的表征也可能不同。下一步可先用新报告身份验证双模型规则，再研究按类蒸馏或分阶段训练。</p>'
    text += '<h2>混合采样的理论覆盖</h2><p>每轮仍为8000次：自然部分无放回，组合部分有放回，两部分合并后打乱。表中为数学期望，尚未进行训练验证。</p>'
    text += table(['组合采样比例', '自然抽样数', '组合抽样数', '期望独立身份', 'FMB期望曝光 %'], [[f"{e['combo_fraction']:.0%}", e['natural_draws_without_replacement'], e['combo_draws_with_replacement'], f"{e['expected_unique']:.1f}", f"{100*e['expected_label_fractions'][0]:.2f}"] for e in expectations])
    text += '<p>50/50混合将期望覆盖由3966.7提升至5316.2，FMB曝光由50.79%调整至67.45%。意图是在FMB负例学习与其他类表征之间取得平衡，准确率是否提高仍需公平对照。</p>'
    text += '<h2>后续顺序</h2><ol><li>修复采样随机流派生；保留相同训练预算和新报告身份。</li><li>用冻结模型比较固定0.5、F1校准与带逐类召回/误报约束的Exact校准。</li><li>自然采样、纯组合采样、50/50混合三组同预算五种子，先单独定位采样收益。</li><li>通过新身份验证互补后，再研究分阶段训练或按类蒸馏，降低双模型推理成本。</li></ol>'
    text += '<p><a href="optimization_diagnosis.json">逐种子数字与输入哈希</a> · <a href="analysis.html">上一轮完整分析</a></p></html>'
    (out / 'optimization_diagnosis.html').write_text(text)
    print(json.dumps({'role': 'check only, exploratory', 'aggregate': aggregate,
                       'report': str(out / 'optimization_diagnosis.html')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
