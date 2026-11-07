"""Frozen natural and identity-paired evaluation, verification, and delivery."""
import csv
import json
import math
import shutil
from pathlib import Path

import torch
import numpy as np
from scipy.special import expit


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def summarize(rows, keys, metrics):
    groups = {}
    for row in rows:
        group = tuple(row[key] for key in keys)
        groups.setdefault(group, []).append(row)
    output = []
    for group, values in groups.items():
        record = dict(zip(keys, group))
        record["seeds"] = len(values)
        for metric in metrics:
            numbers = np.array([float(row[metric]) for row in values])
            record[metric + "_mean"] = float(numbers.mean())
            record[metric + "_sd"] = float(numbers.std(ddof=1))
        output.append(record)
    return output


def natural_slices(args, manifest, freeze, helper):
    records = read(args.source / "development_identities.json")
    power = np.array([[np.nan if value is None else value for value in row["component_JNR_dB"]] for row in records])
    active = np.isfinite(power)
    gap = np.full_like(power, np.nan)
    for label in range(6):
        others = power.copy()
        others[:, label] = np.nan
        maxima = np.max(np.where(np.isfinite(others), others, -np.inf), axis=1)
        eligible = active[:, label] & np.isfinite(maxima)
        gap[eligible, label] = maxima[eligible] - power[eligible, label]
    target_mask = active & (gap >= 10)
    target_mask[:, [0, 1, 3]] = False
    cardinality = active.sum(axis=1)
    slices = []
    weak_rows = []
    loss_parts = []
    for name, calibration in freeze["models"].items():
        arm, seed = name.split("_")
        saved = np.load(args.output / name / "evaluation_logits.npz")
        z, y = saved["logits"], saved["labels"]
        np.testing.assert_array_equal(y.astype(bool), active)
        losses = helper.binary_loss(z, y)
        prediction = expit(z) >= np.array(calibration["thresholds"]["raw"])
        missed = ~prediction & active
        exact = (prediction == active).all(axis=1)
        fixed_wrong = (z >= 0) != active
        loss_parts.append({"arm": arm, "seed": int(seed),
            "correct_bit_BCE_contribution": float(np.where(~fixed_wrong, losses, 0).mean()),
            "wrong_bit_BCE_contribution": float(np.where(fixed_wrong, losses, 0).mean()),
            "one_bit_error_fraction_of_failed_samples": float(((prediction != active).sum(1) == 1).sum() / max((~exact).sum(), 1))})
        weak_rows.append({"arm": arm, "seed": int(seed), "positive_bits": int(target_mask.sum()),
            "FN": int(missed[target_mask].sum()), "FNR": float(missed[target_mask].mean()),
            "fixed05_FNR": float((z[target_mask] < 0).mean()), "BCE": float(losses[target_mask].mean())})
        for count in range(1, 7):
            mask = cardinality == count
            slices.append({"arm": arm, "seed": int(seed), "slice": "cardinality", "group": count,
                "label": "all", "n": int(mask.sum()), "BCE": float(losses[mask].mean()),
                "exact": float(exact[mask].mean()), "FNR": float(missed[mask].sum()/active[mask].sum())})
        masks = [("JNR", f"{low}-{low+5}", active & (power >= low) & (power < low+5)) for low in (10, 15, 20, 25)]
        masks += [("gap", "strongest", active & (gap <= 0)),
            ("gap", "0-5", active & (gap > 0) & (gap < 5)),
            ("gap", "5-10", active & (gap >= 5) & (gap < 10)),
            ("gap", "10+", active & (gap >= 10))]
        for kind, group, mask in masks:
            for label in [None] + list(range(6)):
                selected = mask.copy()
                if label is not None:
                    selected[:, np.arange(6) != label] = False
                assert selected.any()
                slices.append({"arm": arm, "seed": int(seed), "slice": kind, "group": group,
                    "label": "all" if label is None else helper.LABELS[label], "n": int(selected.sum()),
                    "BCE": float(losses[selected].mean()), "FNR": float(missed[selected].mean()),
                    "fixed05_FNR": float((z[selected] < 0).mean())})
    helper.table(args.output / "natural_slices.csv", slices)
    helper.table(args.output / "weak_target_natural.csv", weak_rows)
    helper.table(args.output / "loss_decomposition.csv", loss_parts)
    return slices, weak_rows


def paired_evaluation(args, manifest, freeze, helper):
    helper.base.write(args.output / "paired_evaluation_access.json", {
        "freeze_sha256": helper.base.sha(args.output / "freeze.json"),
        "all_models_and_calibrations_frozen": True, "role": "confirmation identity-paired interventions"})
    datasets = {}
    for gap in (0, 5, 10, 15):
        x, positions, targets = helper.load_views(args.source / f"diagnostic_gap{gap}.mat")
        datasets[gap] = (x, positions, targets)
    np.testing.assert_array_equal(datasets[0][1], datasets[5][1])
    np.testing.assert_array_equal(datasets[0][1], datasets[10][1])
    np.testing.assert_array_equal(datasets[0][2], datasets[10][2])
    subset = np.searchsorted(datasets[0][1], datasets[15][1])
    np.testing.assert_array_equal(datasets[0][1][subset], datasets[15][1])
    np.testing.assert_array_equal(datasets[0][2][subset], datasets[15][2])
    rows = []
    effects = []
    fit_rows = []
    fit_x, fit_y = helper.base.load(manifest["fit_path"],manifest["fit_A"])
    fit_x = fit_x.cuda()
    for name, calibration in freeze["models"].items():
        arm, seed = name.split("_")
        folder = args.output / name
        assert helper.base.sha(folder / "best_BCE.pth") == calibration["checkpoint_sha256"]
        model = helper.base.build_backbone("dfms_iq", num_classes=6, width=16, dropout=.2).cuda()
        model.load_state_dict(torch.load(folder / "best_BCE.pth", map_location="cuda", weights_only=False)["model_state_dict"])
        fit_z = helper.logits(model,fit_x).cpu().numpy().astype(np.float64)
        fit_prediction = expit(fit_z)>=np.asarray(calibration["thresholds"]["raw"])
        fit_rows.append({"arm":arm,"seed":int(seed),"BCE":float(helper.binary_loss(fit_z,fit_y).mean()),
            "fixed05_exact":float(((fit_z>=0)==fit_y).all(1).mean()),
            "f1_exact":float((fit_prediction==fit_y).all(1).mean())})
        predictions = {}
        for gap, (x, positions, targets) in datasets.items():
            path = folder / f"paired_gap{gap}_logits.npz"
            if path.exists():
                saved = np.load(path)
                np.testing.assert_array_equal(saved["positions"], positions)
                np.testing.assert_array_equal(saved["targets"], targets)
                z = saved["logits"]
            else:
                z = helper.logits(model, x.cuda()).cpu().numpy().astype(np.float64)
                np.savez_compressed(path, logits=z, positions=positions, targets=targets)
            target_z = z[np.arange(len(z)), targets]
            threshold = np.asarray(calibration["thresholds"]["raw"])[targets]
            probabilities = expit(target_z)
            predictions[gap] = {"BCE": np.logaddexp(0, -target_z), "FNR": probabilities < threshold,
                "fixed05_FNR": target_z < 0, "probability": probabilities}
            populations = [("gap15_feasible", np.arange(len(z)) if gap == 15 else subset)]
            if gap != 15:
                populations.append(("gap0_5_10_feasible", np.arange(len(z))))
            for population, indices in populations:
                for label in [None, 2, 4, 5]:
                    selected = indices if label is None else indices[targets[indices] == label]
                    assert len(selected) > 0
                    rows.append({"arm": arm, "seed": int(seed), "population": population, "gap": gap,
                        "label": "all_targets" if label is None else helper.LABELS[label], "n": len(selected),
                        **{key: float(value[selected].mean()) for key, value in predictions[gap].items()}})
        for gap in (10, 15):
            baseline_indices = np.arange(len(datasets[0][1])) if gap == 10 else subset
            targets = datasets[gap][2]
            for label in [None, 2, 4, 5]:
                selected = np.arange(len(targets)) if label is None else np.flatnonzero(targets == label)
                strong = predictions[0]["FNR"][baseline_indices[selected]]
                weak = predictions[gap]["FNR"][selected]
                effects.append({"arm": arm, "seed": int(seed), "gap": gap,
                    "label": "all_targets" if label is None else helper.LABELS[label], "n": len(selected),
                    "FNR_at_gap0": float(strong.mean()), "FNR_at_weak_gap": float(weak.mean()),
                    "weak_miss_strong_detect": int((weak & ~strong).sum()),
                    "weak_detect_strong_miss": int((~weak & strong).sum()),
                    "target_BCE_increase": float((predictions[gap]["BCE"][selected]-predictions[0]["BCE"][baseline_indices[selected]]).mean())})
        print("PAIRED_EVALUATION_COMPLETE", name, flush=True)
    helper.table(args.output / "paired_power_metrics.csv", rows)
    helper.table(args.output / "paired_power_effects.csv", effects)
    helper.table(args.output / "fit_checkpoint_metrics.csv", fit_rows)
    return rows, effects


def verify(args, config, freeze, helper):
    errors = []
    checked_rows = 0
    pair_records = read(args.source / "fit_pairing_identities.json")
    eligible_count = sum(row["eligible"] for row in pair_records)
    for seed in args.seeds:
        receipts = [read(args.output / f"{arm}_{seed}" / "receipt.json") for arm in args.arms]
        assert len({row["initial_state_sha256"] for row in receipts}) == 1
        assert all(row["order_sha256"] == receipts[0]["order_sha256"] for row in receipts)
        for arm, receipt in zip(args.arms, receipts):
            folder = args.output / f"{arm}_{seed}"
            history = read(folder / "history.json")
            assert len(history) == 40 and [row["epoch"] for row in history] == list(range(1, 41))
            assert receipt["epochs"] == 40 and receipt["updates"] == 10000 and receipt["unique_fit_per_epoch"] == 8000
            assert receipt["best_BCE_epoch"] == min(history, key=lambda row: row["early_BCE"])["epoch"]
            assert sum(row["augmented_presentations"] for row in history) == (20*eligible_count if arm == "W" else 0)
            assert helper.base.sha(folder / "best_BCE.pth") == receipt["checkpoint_sha256"]
            checked_rows += len(history)
            saved = np.load(folder / "calibration_logits.npz")
            calibration = freeze["models"][f"{arm}_{seed}"]
            assert helper.fit_thresholds(saved["logits"], saved["labels"]) == calibration["thresholds"]["raw"]
            scaled_loss = float(helper.binary_loss(saved["logits"] / calibration["T"], saved["labels"]).mean())
            assert abs(scaled_loss - calibration["calibration_BCE_scaled"]) < 1e-12
    with (args.output / "evaluation_metrics.csv").open(encoding="utf-8-sig") as stream:
        metrics = list(csv.DictReader(stream))
    for row in metrics:
        folder = args.output / f'{row["arm"]}_{row["seed"]}'
        saved = np.load(folder / "evaluation_logits.npz")
        calibration = read(folder / "calibration.json")
        z = saved["logits"] / (calibration["T"] if row["variant"] == "temperature" else 1)
        values, _, _, _ = helper.metrics(z, saved["labels"], calibration["thresholds"][row["variant"]], config["tail_codes"])
        for key, value in values.items():
            errors.append(abs(value - float(row[key])))
    assert max(errors) < 1e-12
    assert helper.source_hashes() == config["source_sha256"]
    checks = {"passed": True, "history_rows": checked_rows, "evaluated_metric_rows": len(metrics),
        "maximum_metric_recompute_error": max(errors), "matched_initialization_and_epoch_orders": True,
        "all_8000_identities_once_per_epoch": True, "weak_view_eligible_identities": eligible_count,
        "weak_presentations_per_W_run": 20*eligible_count, "total_presentations_per_run": 320000,
        "source_hashes_unchanged": True, "calibration_thresholds_recomputed": True, "locked_test_used": False}
    helper.base.write(args.output / "verification.json", checks)
    return metrics, checks


def plots(args, summaries, slices, paired, helper):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"A": "#64748b", "B": "#187bcd", "W": "#db6b22"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for arm in args.arms:
        histories = [read(args.output / f"{arm}_{seed}" / "history.json") for seed in args.seeds]
        for axis, metric, title in zip(axes, ("train_hard_BCE", "early_BCE"),
                ("Training hard BCE (views differ for W)", "Natural early-stop BCE")):
            values = np.array([[row[metric] for row in history] for history in histories])
            mean, sd = values.mean(0), values.std(0, ddof=1)
            axis.plot(np.arange(1, 41), mean, label=arm, color=colors[arm])
            axis.fill_between(np.arange(1, 41), mean-sd, mean+sd, alpha=.12, color=colors[arm])
            axis.set(xlabel="Epoch", ylabel="BCE", title=title)
            axis.grid(alpha=.2)
            axis.legend()
    fig.tight_layout()
    fig.savefig(args.output / "loss_curves.png", dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for arm in args.arms:
        rows = [row for row in summaries if row["arm"] == arm and row["variant"] == "raw"]
        axes[0].bar(arm, rows[0]["BCE_mean"], yerr=rows[0]["BCE_sd"], color=colors[arm], capsize=4)
        counts = range(1, 7)
        exact = [np.mean([row["exact"] for row in slices if row["arm"] == arm and row["slice"] == "cardinality" and row["group"] == count]) for count in counts]
        axes[1].plot(list(counts), np.array(exact)*100, "o-", label=arm, color=colors[arm])
        fnr = [np.mean([row["FNR"] for row in slices if row["arm"] == arm and row["slice"] == "gap" and row["group"] == group and row["label"] == "all"]) for group in ("strongest", "0-5", "5-10", "10+")]
        axes[2].plot(["strongest", "0-5", "5-10", "10+"], np.array(fnr)*100, "o-", label=arm, color=colors[arm])
    axes[0].set(title="Fresh confirmation raw BCE", ylabel="Mean +/- seed SD")
    axes[1].set(title="Natural combination accuracy", xlabel="Active components", ylabel="Exact match (%)")
    axes[2].set(title="Natural positive-bit miss rate", xlabel="Gap to strongest other (dB)", ylabel="FNR (%)")
    for axis in axes[1:]:
        axis.legend()
        axis.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(args.output / "natural_confirmation.png", dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for axis, population, gaps in zip(axes, ("gap0_5_10_feasible", "gap15_feasible"), ((0, 5, 10), (0, 5, 10, 15))):
        n = next(row["n"] for row in paired if row["population"] == population and row["label"] == "all_targets")
        for arm in args.arms:
            means = [np.mean([row["FNR"] for row in paired if row["arm"] == arm and row["population"] == population and row["gap"] == gap and row["label"] == "all_targets"]) for gap in gaps]
            axis.plot(gaps, np.array(means)*100, "o-", label=arm, color=colors[arm])
        axis.set(title=f"Same-identity power intervention (n={n})", xlabel="Target gap to strongest other (dB)", ylabel="Target FNR (%)", xticks=gaps)
        axis.legend()
        axis.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(args.output / "paired_power_response.png", dpi=170)
    plt.close(fig)


def write_report(args, config, decision, summaries, weak, effects, checks, helper):
    fit_distribution = read(args.source / "fit_A_distribution.json")
    pairing = read(args.source / "fit_pairing_identities.json")
    runs = read(args.output / "runs.json")
    reasons = {reason: sum(row["reason"] == reason for row in pairing) for reason in sorted({row["reason"] for row in pairing})}
    lines = ["余弦复验与弱分量覆盖训练研究结果", "", "一、研究协议与证据范围",
        "本轮全新合成身份和五个新模型种子。三组共用预先冻结的角色划分，全部模型与校准冻结后，一次性评估 4032 个自然确认身份及其物理配对视图。",
        "原 5% 筛选失败记录保留。本轮预先采用 3% BCE 实用收益门槛，并增加 FMB 误报和弱分量漏检约束。两个主比较分别报告。",
        "既有 locked-test 未访问。本轮确认集现已消费，后续参数选择需要新的独立确认数据。",
        "", "二、数据、网络与训练预算",
        "生成 fit 候选 25200、dev 候选 6048。fit 使用 8000，early_stop 1008，calibration 1008，confirmation 4032。所有 63 个非空组合保留。",
        f"训练六标签正样本数 {fit_distribution['positive']}，标签 IF=3。训练身份占候选池 31.75%，选中 8000 个身份在每轮全部使用。",
        "固定 F2 dfms_iq，257511 参数，全网络微调。输入为 [2,3000] RMS 归一化 I/Q。共享历史骨干初始化，分类头按种子初始化。",
        "A 固定 lr=1e-4，B 与 W 为 40 轮余弦 1e-4 至 1e-6。硬 BCE、AdamW、wd=1e-4、batch32、clip5、FP32、TF32 关闭。",
        "每组每种子 40 轮、10000 次优化更新、320000 次身份呈现。五种子为 2060–2064。选择 early_stop 最低 raw BCE 检查点。",
        "", "三、弱视图覆盖与物理验证",
        f"可生成训练弱视图的独立身份 {checks['weak_view_eligible_identities']} 个，占 8000 身份 {checks['weak_view_eligible_identities']/80:.2f}%。",
        f"每个 W 运行弱视图呈现 {checks['weak_presentations_per_W_run']} 次，原始视图呈现 {320000-checks['weak_presentations_per_W_run']} 次。独立训练身份仍为 8000。",
        "各身份可行性统计 " + json.dumps(reasons, ensure_ascii=False),
        "各目标干预身份数 " + json.dumps({helper.LABELS[label]: sum(row['target_label'] == label for row in pairing) for label in (2,4,5)}, ensure_ascii=False),
        "目标有效 JNR=max_other-10，始终在 10–30 dB。已弱或不可行身份使用原始视图。每个可干预身份恰好 20 个原始轮次和 20 个弱视图轮次。",
        "fit 和 diagnostic 原始重建、单分量幅度差分、种子及标签核验结果如下。",
        json.dumps(read(args.source / "fit_pairing_receipt.json"), ensure_ascii=False),
        json.dumps(read(args.source / "diagnostic_pairing_receipt.json"), ensure_ascii=False),
        "", "四、自然确认指标，五种子均值 ± 标准差",
        "组别  视图  BCE  macroAP  F1阈值exact  tail exact  ECE"]
    for row in summaries:
        lines.append(f"{row['arm']}  {row['variant']}  {row['BCE_mean']:.6f}±{row['BCE_sd']:.6f}  "
            f"{row['macro_AP_mean']*100:.3f}%  {row['f1_exact_mean']*100:.3f}%  "
            f"{row['tail_exact_mean']*100:.3f}%  {row['ECE_mean']:.5f}")
    lines += ["", "逐种子主指标及最佳轮次"]
    for run in runs:
        saved = np.load(args.output / f"{run['arm']}_{run['seed']}" / "evaluation_logits.npz")
        bce = helper.binary_loss(saved["logits"], saved["labels"]).mean()
        lines.append(f"{run['arm']} seed={run['seed']} rawBCE={bce:.6f} best_epoch={run['best_BCE_epoch']} earlyBCE={run['best_early_BCE']:.6f}")
    lines += ["", "五、预注册主比较与晋级判定"]
    for key in ("B_raw_vs_A_raw", "W_raw_vs_B_raw"):
        result = decision["comparisons"][key]
        lines += [key, f"平均相对 BCE 降低 {result['relative_reduction']*100:.3f}%，逐种子 BCE 收益 {result['paired_BCE_gain']}",
            f"配对身份 bootstrap 95% 区间 {result['identity_bootstrap95_BCE_gain']}",
            f"五种子配对均值 t95% 区间 {result['seed_t95_BCE_gain']}",
            "各门槛 " + json.dumps(result["checks"], ensure_ascii=False),
            "晋级 " + str(result["passes"])]
    lines += ["身份 bootstrap 条件于本次五个模型，种子 t 区间反映有限训练种子间波动。它们不构成所有分布和所有随机种子的保证。",
        "", "六、弱分量自然与配对诊断"]
    for arm in args.arms:
        selected = [row for row in weak if row["arm"] == arm]
        lines.append(f"{arm} 自然目标三类 gap>=10，{selected[0]['positive_bits']} 个正标签位，平均 FNR={np.mean([row['FNR'] for row in selected])*100:.3f}%，BCE={np.mean([row['BCE'] for row in selected]):.6f}")
        for gap in (10,15):
            selected = [row for row in effects if row["arm"] == arm and row["gap"] == gap and row["label"] == "all_targets"]
            lines.append(f"{arm} 配对 n={selected[0]['n']}，gap0 FNR={np.mean([row['FNR_at_gap0'] for row in selected])*100:.3f}% → gap{gap} FNR={np.mean([row['FNR_at_weak_gap'] for row in selected])*100:.3f}%")
    lines += ["gap0/5/10 与 gap15 的可行人群不同。图表在各自固定身份子集内比较，gap15 的对应 gap0/5/10 同样限制在 max_other>=25 dB。",
        "这些干预改变目标分量功率，控制同身份其他分量和接收噪声，支持功率敏感性的模型层证据。结论适用范围为当前合成器、10–30 dB 和可行子集。",
        "", "七、研究结论与下一步"]
    b_pass = decision["comparisons"]["B_raw_vs_A_raw"]["passes"]
    w_pass = decision["comparisons"]["W_raw_vs_B_raw"]["passes"]
    if w_pass:
        lines.append("W 达到预注册晋级条件。下一阶段以独立身份和预先固定的新功率区间检验可迁移性，再决定是否作为正式方法。")
    elif b_pass:
        lines.append("B 达到预注册晋级条件，作为改进基线。W 未达到弱分量方法的完整门槛，其失败分项与逐类代价须保留。下一步围绕配对诊断暴露的表示和判别问题设计新研究。")
    else:
        lines.append("主比较未形成可直接采用的改进基线与弱分量方法。保留各分项收益及失败约束，下一步应在新研究中明确单一机制假设，避免围绕本次确认集反复调整。")
    lines += ["网络规模判断需要同时考虑原始 fit 拟合、自然确认误差与同身份功率响应。本轮固定网络检验了训练策略和数据覆盖，容量最优性仍需匹配预算的结构对照。",
        "当前五种子使用同一新身份划分，因此结果反映该划分下的训练随机性与确认身份抽样波动。未执行多份独立数据划分重复，也未执行外部真实采集验证。",
        "", "八、执行与可复核性",
        f"全部训练累计墙钟计时 {sum(run['elapsed_seconds'] for run in runs)/60:.2f} 分钟，生成、校准和配对评估另计。",
        json.dumps(checks, ensure_ascii=False),
        "实验数值、逐类和组合指标、功率切片、温度校准、全部 loss 历史见附带 CSV/JSON。完整权重、原始 MAT 与预测位于外部存储，轻量交付清单记录其 SHA-256。",
        str(args.output.resolve()), ""]
    with (args.output / "fit_checkpoint_metrics.csv").open(encoding="utf-8-sig") as stream:
        fit_rows = list(csv.DictReader(stream))
    lines += ["最佳检查点在原始训练身份上的重新评估"]
    for arm in args.arms:
        rows = [row for row in fit_rows if row["arm"]==arm]
        lines.append(f"{arm} 平均原始 fit BCE={np.mean([float(row['BCE']) for row in rows]):.6f}，固定0.5 exact={np.mean([float(row['fixed05_exact']) for row in rows])*100:.3f}%")
    lines += ["", "执行中修复记录", json.dumps(read(helper.HERE / "execution_fixes.json"),ensure_ascii=False,indent=2)]
    (args.output / "RESULTS.txt").write_text("\n".join(lines), encoding="utf-8")


def analyze(args, manifest, config, freeze, decision, helper):
    slices, weak = natural_slices(args, manifest, freeze, helper)
    paired, effects = paired_evaluation(args, manifest, freeze, helper)
    natural_fnr = {arm: float(np.mean([row["FNR"] for row in weak if row["arm"] == arm])) for arm in args.arms}
    paired_fnr = {arm: float(np.mean([row["FNR"] for row in paired if row["arm"] == arm and row["gap"] == 10
        and row["population"] == "gap0_5_10_feasible" and row["label"] == "all_targets"])) for arm in args.arms}
    comparison = decision["comparisons"]["W_raw_vs_B_raw"]
    comparison["natural_weak_target_FNR"] = natural_fnr
    comparison["paired_gap10_target_FNR"] = paired_fnr
    comparison["checks"]["natural_weak_target_relative_FNR_reduction_ge5pct"] = natural_fnr["W"] <= .95*natural_fnr["B"]
    comparison["checks"]["paired_gap10_target_FNR_noninferior"] = paired_fnr["W"] <= paired_fnr["B"]
    comparison["passes"] = all(comparison["checks"].values())
    decision["selected_training_candidate"] = "W" if comparison["passes"] else (
        "B" if decision["comparisons"]["B_raw_vs_A_raw"]["passes"] else None)
    decision["paired_weak_checks_pending"] = False
    decision["confirmation_consumed"] = True
    decision["status"] = "completed"
    metrics, checks = verify(args, config, freeze, helper)
    summaries = summarize(metrics, ("arm", "variant"), ("BCE", "macro_AP", "fixed05_exact", "f1_exact", "tail_exact", "ECE", "Brier", "macro_F1"))
    helper.table(args.output / "summary_metrics.csv", summaries)
    helper.base.write(args.output / "decision.json", decision)
    plots(args, summaries, slices, paired, helper)
    write_report(args, config, decision, summaries, weak, effects, checks, helper)
    artifacts = {str(path.relative_to(args.output)): helper.base.sha(path) for path in args.output.rglob("*")
        if path.is_file() and path.name != "artifact_sha256.json" and not path.name.startswith("status")}
    helper.base.write(args.output / "artifact_sha256.json", artifacts)
    delivery = helper.HERE / "results"
    delivery.mkdir(exist_ok=True)
    for path in args.output.iterdir():
        if path.is_file() and path.suffix in (".json", ".csv", ".txt", ".png"):
            shutil.copy2(path, delivery / path.name)
    for name in freeze["models"]:
        (delivery / name).mkdir(exist_ok=True)
        for filename in ("history.json", "receipt.json", "calibration.json"):
            shutil.copy2(args.output / name / filename, delivery / name / filename)
    for path in args.source.glob("*receipt.json"):
        shutil.copy2(path, delivery / path.name)
    shutil.copy2(args.source / "preflight.json", delivery / "data_preflight.json")
    shutil.copy2(helper.HERE / "execution_fixes.json", delivery / "execution_fixes.json")
    shutil.copy2(args.output / "RESULTS.txt", helper.HERE / "RESULTS.txt")
    helper.base.write(helper.HERE / "delivery_sha256.json", {str(path.relative_to(helper.HERE)): helper.base.sha(path)
        for path in delivery.rglob("*") if path.is_file()})
    helper.base.write(args.output / "status.json", {"status": "completed", "selected": decision["selected_training_candidate"], "verification": checks})
    print("RESEARCH_GOAL_COMPLETE", json.dumps(decision), flush=True)
