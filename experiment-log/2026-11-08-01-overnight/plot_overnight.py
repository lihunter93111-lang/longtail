"""Standalone scientific figures from immutable experiment outputs."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def table(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def moments(rows, metric):
    values = np.array([float(row[metric]) for row in rows])
    return values.mean(), values.std(ddof=1)


def finish(figure, path, caption):
    figure.text(.02, .015, caption, fontsize=8, color="#475569")
    figure.tight_layout(rect=(0, .05, 1, .95))
    figure.savefig(path, dpi=220, facecolor="white")
    plt.close(figure)


def training_curves(root, output):
    figure, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True, sharey=True)
    for row, distribution in enumerate(("IF3", "balanced")):
        for column, width in enumerate((16, 32)):
            axis = axes[row, column]
            histories = [read(root / f"capacity_{distribution}_w{width}/M_{seed}/history.json")
                         for seed in range(2080, 2085)]
            for metric, label, color in (("train_hard_BCE", "训练目标（含弱视图）", "#94a3b8"),
                                         ("fit_eval_BCE", "训练身份自然视图", "#0f766e"),
                                         ("early_BCE", "early_stop", "#b45309")):
                epochs = [i + 1 for i, record in enumerate(histories[0]) if record[metric] is not None]
                values = np.array([[history[i - 1][metric] for i in epochs] for history in histories])
                average, deviation = values.mean(axis=0), values.std(axis=0, ddof=1)
                axis.plot(epochs, average, label=label, color=color, linewidth=1.8)
                axis.fill_between(epochs, np.maximum(average - deviation, 1e-5), average + deviation,
                                  color=color, alpha=.13, linewidth=0)
            axis.set(title=f"{'IF3' if distribution == 'IF3' else '近均衡'} · width{width}",
                     xlabel="训练轮次", ylabel="硬标签 BCE", yscale="log", xlim=(1, 40))
            axis.grid(alpha=.15)
            axis.legend(fontsize=8)
    figure.suptitle("同预算容量对照的训练与 early_stop 曲线", fontsize=15)
    finish(figure, output / "01_training_curves.png", "五种子均值与±1标准差。自然训练视图每5轮评估。各组均完成40轮，检查点由early_stop最低BCE选择。")


def capacity_matrix(root, output):
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.8))
    for axis, metric, label, scale in ((axes[0], "BCE", "开发集 raw BCE", 1),
                                       (axes[1], "f1_exact", "开发集 exact（%）", 100)):
        for index, width in enumerate((16, 32)):
            means, deviations = [], []
            for distribution in ("IF3", "balanced"):
                rows = [row for row in table(root / f"capacity_{distribution}_w{width}/evaluation_metrics.csv")
                        if row["variant"] == "raw"]
                mean, deviation = moments(rows, metric)
                means.append(mean * scale)
                deviations.append(deviation * scale)
            locations = np.arange(2) + (index - .5) * .3
            axis.bar(locations, means, width=.27, yerr=deviations, capsize=4,
                     color=("#94a3b8", "#0369a1")[index], label=f"width{width}")
            for location, mean, deviation in zip(locations, means, deviations):
                axis.text(location, mean + deviation + (.002 if metric == "BCE" else .8),
                          f"{mean:.4f}" if metric == "BCE" else f"{mean:.2f}", ha="center", fontsize=9)
        axis.set(xticks=[0, 1], xticklabels=["IF3训练", "近均衡训练"], ylabel=label)
        axis.grid(axis="y", alpha=.15)
        axis.set_axisbelow(True)
        axis.legend()
    axes[0].set_ylim(0, .18)
    axes[1].set_ylim(0, 100)
    figure.suptitle("容量 × 训练分布 · 相同4032开发身份", fontsize=15)
    finish(figure, output / "02_capacity_matrix.png", "五种子均值±标准差。两种宽度均随机初始化、M覆盖、40轮。近均衡改变标签边际、组合共现和部分训练身份。")


def stage_comparison(root, output):
    specifications = [
        ("B 自然视图", "stage2", "B"), ("W gap10", "stage2", "W"), ("M gap10/15", "stage2", "M"),
        ("M + top8读出", "stage4_topk", "M"), ("M + 全局统计旁路", "stage4_global_control", "M"),
        ("M + 特征金字塔", "stage4_pyramid", "M"), ("M width16 随机初始化", "capacity_IF3_w16", "M"),
        ("M width32 随机初始化", "capacity_IF3_w32", "M")]
    figure, axes = plt.subplots(1, 2, figsize=(11, 6), sharey=True)
    for position, (label, folder, arm) in enumerate(specifications):
        rows = [r for r in table(root / folder / "evaluation_metrics.csv") if r["variant"] == "raw" and r["arm"] == arm]
        for axis, metric, scale in ((axes[0], "BCE", 1), (axes[1], "f1_exact", 100)):
            mean, deviation = moments(rows, metric)
            color = "#0369a1" if "width32" in label else "#64748b"
            axis.errorbar(mean * scale, position, xerr=deviation * scale, fmt="o", color=color, capsize=3)
            axis.text(mean * scale, position - .17, f"{mean * scale:.4f}" if scale == 1 else f"{mean * scale:.2f}",
                      ha="center", fontsize=8, color=color)
    axes[0].set_yticks(range(len(specifications)), [row[0] for row in specifications])
    axes[0].invert_yaxis()
    axes[0].set_xlabel("开发集 raw BCE")
    axes[1].set_xlabel("开发集 exact（%）")
    for axis in axes:
        axis.grid(axis="x", alpha=.2)
    figure.suptitle("主要训练分支 · 同一IF3数据与40轮预算", fontsize=15)
    finish(figure, output / "03_stage_comparison.png", "点为五种子均值，误差线为标准差。结构分支沿用共享F2初始化，容量两组使用匹配随机初始化。完整晋级还需逐类保护指标。")


def weak_target(root, output):
    settings = [("W", "stage5_counterfactual_main", "W", "#94a3b8"),
                ("M", "stage5_counterfactual_main", "M", "#0f766e"),
                ("M width16 scratch", "stage5_counterfactual_scratch16", "M", "#b45309"),
                ("M width32 scratch", "stage5_counterfactual_scratch32", "M", "#0369a1")]
    labels = ["all_targets", "SawFM", "MFT", "ISRJ"]
    figure, axes = plt.subplots(1, 2, figsize=(11, 5))
    for index, (name, folder, arm, color) in enumerate(settings):
        records = table(root / folder / "metrics.csv")
        for axis, metric in zip(axes, ("TPR", "matched_absent_FPR")):
            average, deviation = [], []
            for label in labels:
                rows = [r for r in records if r["arm"] == arm and int(r["gap"]) == 15 and r["label"] == label]
                mean, sd = moments(rows, metric)
                average.append(mean * 100)
                deviation.append(sd * 100)
            axis.bar(np.arange(4) + (index - 1.5) * .18, average, width=.17, color=color,
                     yerr=deviation, capsize=2, label=name)
    for axis in axes:
        axis.set_xticks(range(4), ["三类合计", "SawFM", "MFT", "ISRJ"])
        axis.grid(axis="y", alpha=.15)
        axis.set_axisbelow(True)
    axes[0].set(ylabel="目标存在时的召回率（%）", ylim=(0, 100))
    axes[1].set(ylabel="对应目标移除后的误报率（%）", ylim=(0, 15))
    axes[1].legend(fontsize=8)
    figure.suptitle("同身份目标存在 / 移除诊断 · gap15", fontsize=15)
    finish(figure, output / "04_weak_target_counterfactual.png", "1686个gap15可行身份，阈值来自calibration。柱为五种子均值，误差线为标准差。每个存在/移除样本共享其他分量与噪声。")


def final_confirmation(root, output):
    path = root / "final_confirmation_results/metrics.csv"
    if not path.exists():
        return
    rows = table(path)
    figure, axes = plt.subplots(1, 3, figsize=(11, 4.7))
    for axis, metric, label, scale in ((axes[0], "BCE", "raw BCE", 1),
                                       (axes[1], "f1_exact", "exact（%）", 100),
                                       (axes[2], "macro_AP", "macro AP（%）", 100)):
        reference = sorted([r for r in rows if r["group"] == "reference" and r["variant"] == "raw"], key=lambda r: r["seed"])
        candidate = sorted([r for r in rows if r["group"] == "candidate" and r["variant"] == "raw"], key=lambda r: r["seed"])
        for a, b in zip(reference, candidate):
            axis.plot([0, 1], [float(a[metric]) * scale, float(b[metric]) * scale], "o-", color="#94a3b8", alpha=.8)
        means = [np.mean([float(r[metric]) for r in group]) * scale for group in (reference, candidate)]
        axis.plot([0, 1], means, "o-", color="#0369a1", linewidth=3, markersize=8)
        axis.set(xticks=[0, 1], xticklabels=["参考 W16", "候选"], ylabel=label, xlim=(-.25, 1.25))
        axis.grid(axis="y", alpha=.2)
    figure.suptitle("预留新身份最终确认 · 五个模型种子", fontsize=15)
    finish(figure, output / "05_final_confirmation.png", "4032个新身份，每组合64。模型与全部校准在确认访问前冻结。灰线为单种子，蓝线为均值。参考和候选按最终freeze文件定义。")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    font_path = Path("C:/Windows/Fonts/msyh.ttc")
    font_manager.fontManager.addfont(str(font_path))
    plt.rcParams.update({"font.family": font_manager.FontProperties(fname=str(font_path)).get_name(),
                         "axes.unicode_minus": False, "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False})
    args.output.mkdir(parents=True, exist_ok=True)
    training_curves(args.root, args.output)
    capacity_matrix(args.root, args.output)
    stage_comparison(args.root, args.output)
    weak_target(args.root, args.output)
    final_confirmation(args.root, args.output)
    print("FIGURES_WRITTEN", str(args.output), flush=True)


if __name__ == "__main__":
    main()
