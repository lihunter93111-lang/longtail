"""Build a deadline-safe report and light archive from completed evidence."""
import csv
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

import training
from verify_day import verify

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / "2026-11-08-01-overnight"
ROOT = Path("E:/LJQ_code/LJQ_code_experiment_storage/daytime_v1/20261109_01")
base = training.base


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def csv_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def plots():
    font_path = "C:/Windows/Fonts/msyh.ttc"
    font_manager.fontManager.addfont(font_path)
    plt.rcParams.update({"font.family": font_manager.FontProperties(fname=font_path).get_name(), "axes.unicode_minus": False})
    destination = ROOT / "figures"
    destination.mkdir(exist_ok=True)
    head_names = ["原始宽骨干", "标准硬BCE头", "标准均衡头", "低步长硬头", "低步长均衡头"]
    if all((ROOT / f"guarded_heads_{split}/metrics.csv").exists() for split in ("primary", "replication")):
        figure, axes = plt.subplots(1, 2, figsize=(12, 5))
        previous = Path("E:/LJQ_code/LJQ_code_experiment_storage/overnight_v1/20261108_01")
        for axis, split, model in zip(axes, ("primary", "replication"), ("capacity_IF3_w32", "replication_M32")):
            baseline = [r for r in csv_rows(previous / model / "evaluation_metrics.csv") if r["variant"] == "raw"]
            groups = [baseline]
            for prefix in ("wide_heads", "guarded_heads"):
                saved = csv_rows(ROOT / f"{prefix}_{split}/metrics.csv")
                groups.extend([[r for r in saved if r["variant"] == variant] for variant in ("hard", "balanced")])
            means = [np.mean([float(r["BCE"]) for r in group]) for group in groups]
            deviations = [np.std([float(r["BCE"]) for r in group], ddof=1) for group in groups]
            axis.bar(range(5), means, yerr=deviations, capsize=3, color=["#0369a1", "#b45309", "#d97706", "#64748b", "#94a3b8"])
            for position, mean, deviation in zip(range(5), means, deviations):
                axis.text(position, mean + deviation + .002, f"{mean:.4f}", ha="center", fontsize=9)
            axis.set(xticks=range(5), xticklabels=head_names, ylabel="开发集raw BCE", ylim=(0, .15),
                     title="第一身份划分" if split == "primary" else "第二身份划分")
            axis.tick_params(axis="x", rotation=22)
            axis.grid(axis="y", alpha=.2)
            axis.set_axisbelow(True)
        figure.suptitle("宽骨干分类头续做 · 五种子均值±标准差")
        figure.tight_layout()
        figure.savefig(destination / "wide_head_comparison.png", dpi=200, facecolor="white")
        plt.close(figure)
    available = [epochs for epochs in (40, 80, 160) if (ROOT / f"budget_e{epochs}/summary_metrics.csv").exists()]
    if available:
        figure, axes = plt.subplots(1, len(available), figsize=(5 * len(available), 4.7), squeeze=False, sharey=True)
        for axis, epochs in zip(axes[0], available):
            histories = [read(ROOT / f"budget_e{epochs}/M_{seed}/history.json") for seed in (2100, 2101, 2102)]
            for key, label, color in (("early_BCE", "early_stop", "#b45309"), ("fit_eval_BCE", "自然fit", "#0f766e")):
                indices = [i for i, row in enumerate(histories[0]) if row[key] is not None]
                values = np.array([[history[i][key] for i in indices] for history in histories])
                x = np.array(indices) + 1
                average = values.mean(0)
                axis.plot(x, average, color=color, label=label)
                axis.fill_between(x, values.min(0), values.max(0), color=color, alpha=.15)
            axis.set(title=f"独立{epochs}轮预算", xlabel="训练轮次", ylabel="硬标签BCE", yscale="log")
            axis.grid(alpha=.2)
            axis.legend()
        figure.suptitle("width32训练预算诊断 · 三种子均值与取值范围")
        figure.tight_layout()
        figure.savefig(destination / "budget_curves.png", dpi=200, facecolor="white")
        plt.close(figure)
        figure, axis = plt.subplots(figsize=(9, 4.8))
        for epochs, color in zip(available, ("#0369a1", "#b45309", "#0f766e")):
            histories = [read(ROOT / f"budget_e{epochs}/M_{seed}/history.json") for seed in (2100, 2101, 2102)]
            values = np.array([[row["early_macro_AP"] for row in history[9:]] for history in histories]) * 100
            x = np.arange(10, epochs + 1)
            mean, sd = values.mean(0), values.std(0, ddof=1)
            axis.plot(x, mean, label=f"{epochs}轮预算", color=color)
            axis.fill_between(x, mean - sd, mean + sd, color=color, alpha=.12)
        axis.set(title="第10轮后的early_stop AP · 三种子均值与标准差", xlabel="训练轮次", ylabel="macro AP (%)")
        axis.legend()
        axis.grid(alpha=.2)
        figure.tight_layout()
        figure.savefig(destination / "budget_AP_curves.png", dpi=200, facecolor="white")
        plt.close(figure)
    if (ROOT / "final_confirmation_results/metrics.csv").exists():
        rows = [r for r in csv_rows(ROOT / "final_confirmation_results/metrics.csv") if r["variant"] == "raw"]
        figure, axes = plt.subplots(1, 2, figsize=(9, 4.7))
        for axis, metric, label, scale in ((axes[0], "BCE", "raw BCE", 1), (axes[1], "f1_exact", "exact (%)", 100)):
            reference = sorted([r for r in rows if r["group"] == "reference"], key=lambda r: r["seed"])
            candidate = sorted([r for r in rows if r["group"] == "candidate"], key=lambda r: r["seed"])
            for a, b in zip(reference, candidate):
                axis.plot([0, 1], [float(a[metric]) * scale, float(b[metric]) * scale], "o-", color="#64748b", alpha=.65)
            axis.set(xticks=[0, 1], xticklabels=["匹配参考", "候选"], ylabel=label, xlim=(-.2, 1.2))
            axis.grid(alpha=.2)
        figure.suptitle("新的预留确认 · 五种子独立模型")
        figure.tight_layout()
        figure.savefig(destination / "confirmation.png", dpi=200, facecolor="white")
        plt.close(figure)
    comparisons = [("主对照", ROOT / "sampling_primary_comparison.json"),
                   ("独立训练划分复验", ROOT / "sampling_replication_comparison.json")]
    comparisons = [(name, path) for name, path in comparisons if path.exists()]
    if comparisons:
        figure, axes = plt.subplots(len(comparisons), 2, figsize=(10, 4.3 * len(comparisons)), squeeze=False)
        for row_index, (name, path) in enumerate(comparisons):
            comparison = read(path)
            for axis, metric, scale, label in ((axes[row_index, 0], "BCE", 1, "raw BCE"),
                                               (axes[row_index, 1], "f1_exact", 100, "exact (%)")):
                reference = [record[metric] * scale for record in comparison["metrics"]["reference"]]
                candidate = [record[metric] * scale for record in comparison["metrics"]["candidate"]]
                for left, right in zip(reference, candidate):
                    axis.plot([0, 1], [left, right], "o-", color="#94a3b8", alpha=.7)
                axis.plot([0, 1], [np.mean(reference), np.mean(candidate)], "o-", color="#0369a1", linewidth=3)
                axis.set(title=f"{name} · {len(reference)}种子", xticks=[0, 1], xticklabels=["自然无放回", "分层混合采样"],
                         ylabel=label, xlim=(-.2, 1.2))
                axis.grid(alpha=.2)
        figure.suptitle("相同身份与更新预算的采样对照")
        figure.tight_layout()
        figure.savefig(destination / "sampling_comparison.png", dpi=200, facecolor="white")
        plt.close(figure)


def main():
    verification = verify(ROOT)
    final_path = ROOT / "final_confirmation_results/decision.json"
    if final_path.exists():
        sys.path.append(str(OLD))
        from build_delivery import verify_final
        verify_final(ROOT)
    plots()
    lines = ["宽骨干续做与长尾探索结果", f"更新时间 {datetime.now(timezone.utc).astimezone().isoformat()}",
        "约定截止为2026-11-09 17:00。优化截止16:00，独立监督进程16:05触发收尾。", "",
        f"已完成完整网络训练{verification['full_runs']}次，冻结分类头拟合{verification['head_fits']}次。",
        f"独立重算{verification['scalar_metrics']}个标量指标，最大误差{verification['max_metric_error']:.3g}。", "",
        "数据与参考", "延续昨晚两套合法fit/early_stop/calibration/development角色，主fit8000身份，全部63组合。",
        "强参考为width32随机初始化、硬BCE、M覆盖。此前最终确认已消费。今日新的确认使用预留种子3109310900，只有合格候选才触发生成。",
        "头诊断每轮8000身份、40轮、batch32、weight_decay1e-4、clip5。标准学习率1e-3，追加低步长对照使用1e-4并包含原始epoch0检查点。",
        "完整预算诊断使用2100–2102种子、独立40/80/160轮，AdamW1e-4余弦到1e-6，其他设置匹配。", "", "阶段结果"]
    used_sources = sorted({record["source"] for record in verification["completed_runs"]})
    unique_identities = set()
    for source in used_sources:
        for record in read(Path(source) / "fit_A_identities.json"):
            unique_identities.add((record["source_file_hash"], record["raw_row"]))
    full_records = [record for record in verification["completed_runs"] if record["kind"] == "full"]
    head_records = [record for record in verification["completed_runs"] if record["kind"] == "head"]
    lines.extend(["已完成并复核运行的样本消费",
        f"完整网络训练累计{sum(record['epochs'] for record in full_records)}轮、{sum(record['updates'] for record in full_records)}次更新、{sum(record['sample_presentations'] for record in full_records)}次身份呈现。",
        f"冻结头累计{sum(record['updates'] for record in head_records)}次更新、{sum(record['sample_presentations'] for record in head_records)}次身份呈现。",
        f"这些运行的训练身份并集为{len(unique_identities)}个，物理功率视图按原身份归组。", ""])
    for name in ("wide_heads_primary", "wide_heads_replication", "guarded_heads_primary", "guarded_heads_replication"):
        path = ROOT / name / "metrics.csv"
        if path.exists():
            rows = csv_rows(path)
            for variant in ("hard", "balanced"):
                chosen = [r for r in rows if r["variant"] == variant]
                bce = np.mean([float(r["BCE"]) for r in chosen])
                exact = np.mean([float(r["f1_exact"]) for r in chosen])
                selected_epochs = [int(r["best_epoch"]) for r in chosen]
                lines.append(f"{name}/{variant} raw BCE={bce:.6f}, exact={exact:.6f}, 选中轮次={selected_epochs}")
    for epochs in (40, 80, 160):
        path = ROOT / f"budget_e{epochs}/summary_metrics.csv"
        if path.exists():
            row = next(r for r in csv_rows(path) if r["variant"] == "raw")
            lines.append(f"独立{epochs}轮 raw BCE={float(row['BCE_mean']):.6f}, AP={float(row['macro_AP_mean']):.6f}, exact={float(row['f1_exact_mean']):.6f}")
    for epochs in (40, 80, 160):
        path = ROOT / f"ap_selection_e{epochs}/evaluation_metrics.csv"
        if path.exists():
            records = csv_rows(path)
            for variant in ("raw", "temperature"):
                selected = [row for row in records if row["variant"] == variant]
                values = {key: np.mean([float(row[key]) for row in selected]) for key in ("BCE", "macro_AP", "f1_exact")}
                lines.append(f"{epochs}轮AP选点 {variant} BCE={values['BCE']:.6f}, AP={values['macro_AP']:.6f}, exact={values['f1_exact']:.6f}，作为检查点目标诊断。")
    if (ROOT / "budget_decision.json").exists():
        lines.extend(["", "预算选择", json.dumps(read(ROOT / "budget_decision.json"), ensure_ascii=False, indent=2)])
    if (ROOT / "stage3_plan.json").exists():
        lines.extend(["", "采样对照与时间分支", json.dumps(read(ROOT / "stage3_plan.json"), ensure_ascii=False, indent=2)])
    if (ROOT / "smoothing_plan.json").exists():
        lines.extend(["", "条件性置信正则化分支", json.dumps(read(ROOT / "smoothing_plan.json"), ensure_ascii=False, indent=2),
            "epsilon=0.01时训练软目标为0.995/0.005，所有检查点选择与评价仍使用硬标签BCE。第二划分参考沿用已完成的匹配40轮模型并核验初始状态及全部样本顺序。"])
    if (ROOT / "combined_plan.json").exists():
        lines.extend(["", "采样与置信正则组合分支", json.dumps(read(ROOT / "combined_plan.json"), ensure_ascii=False, indent=2),
            "组合分支在模型训练前依据已完成的校准对照修订，保留03C原方案及branch_revision_03D.json。自然遍历加平滑单独分支延期，当前组合同时包含固定混合采样与epsilon0.01。"])
    if (ROOT / "axis_pilot_plan.json").exists():
        lines.extend(["", "受控数据轴初筛计划状态", json.dumps(read(ROOT / "axis_pilot_plan.json"), ensure_ascii=False, indent=2)])
    if (ROOT / "axis_pilot_review.json").exists():
        axis_review = read(ROOT / "axis_pilot_review.json")
        lines.extend(["", "三种子受控数据轴初筛",
            f"参考BCE {axis_review['reference_BCE']:.6f}，候选数据条件BCE {axis_review['candidate_BCE']:.6f}，相对下降{axis_review['relative_BCE_reduction'] * 100:.4f}%。",
            f"配对收益{axis_review['paired_gains']}，种子t95区间{axis_review['seed_t95']}。",
            f"两条件共享{axis_review['shared_fit_identities']}个训练身份。该比较属于数据条件开发初筛，配额语义归属和部分身份发生变化。"])
    for filename, label in (("sampling_primary_comparison.json", "五种子主采样对照"),
                             ("sampling_replication_comparison.json", "独立训练划分采样复验"),
                             ("smoothing_primary_comparison.json", "五种子置信正则化主对照"),
                             ("smoothing_replication_comparison.json", "置信正则化独立训练划分复验"),
                             ("combined_primary_comparison.json", "五种子采样加平滑主对照"),
                             ("combined_incremental_comparison.json", "相对采样硬BCE的增量对照"),
                             ("combined_replication_comparison.json", "采样加平滑独立训练划分复验")):
        if (ROOT / filename).exists():
            comparison = read(ROOT / filename)
            lines.extend(["", label,
                f"参考raw BCE {comparison['reference_BCE']:.6f}，候选{comparison['candidate_BCE']:.6f}，相对下降{comparison['relative_BCE_reduction'] * 100:.4f}%。",
                f"种子 {comparison['seeds']}，配对收益 {comparison['paired_BCE_gain']}。",
                f"种子t95区间 {comparison['seed_t95']}，身份bootstrap区间 {comparison['identity_bootstrap95']}。",
                json.dumps(comparison["checks"], ensure_ascii=False, indent=2)])
    sampled = [row for row in verification["completed_runs"] if (ROOT / row["path"] / "sample_counts.npz").exists()]
    if (ROOT / "sampling_probability_review.json").exists():
        lines.extend(["", "采样与概率校准对照", json.dumps(read(ROOT / "sampling_probability_review.json"), ensure_ascii=False, indent=2)])
    if sampled:
        lines.extend(["", "采样实际消费"])
        for record in sampled:
            receipt = read(ROOT / record["path"] / "receipt.json")
            unique = receipt["unique_fit_per_epoch"]
            lines.append(f"{record['path']} 每轮唯一身份均值{np.mean(unique):.2f}，范围{min(unique)}–{max(unique)}，"
                         f"累计唯一身份{receipt['total_unique_identities']}，实际访问有效样本量{receipt['realized_effective_sample_size']:.2f}。")
    analysis_notes = HERE / "STAGE_ANALYSIS.txt"
    if analysis_notes.exists():
        lines.extend(["", "阶段分析", analysis_notes.read_text(encoding="utf-8")])
    if final_path.exists():
        final = read(final_path)
        lines.extend(["", "最终独立确认", json.dumps(final, ensure_ascii=False, indent=2)])
    elif (ROOT / "confirmation_decision.json").exists():
        lines.extend(["", "确认状态", json.dumps(read(ROOT / "confirmation_decision.json"), ensure_ascii=False, indent=2)])
    else:
        lines.extend(["", "确认尚待候选冻结，此文件为阶段快照。"])
    if (ROOT / "supervisor_jobs.json").exists():
        lines.extend(["", "运行状态"])
        for name, item in read(ROOT / "supervisor_jobs.json").items():
            lines.append(f"{name}  {item['status']}  {item.get('started_utc', '')}  {item.get('ended_utc', '')}")
        lines.append("完整命令、阶段提示词hash与运行收据保存在supervisor_jobs.json。")
    if (ROOT / "closeout_recovered_jobs.json").exists():
        lines.extend(["", "收尾阶段评价恢复", json.dumps(read(ROOT / "closeout_recovered_jobs.json"), ensure_ascii=False, indent=2)])
    lines.extend(["", "证据范围与交付", "五种子在各划分内部共享身份，种子区间和身份bootstrap分别报告。",
        "40/80/160轮、2%BCE晋级线及30次完整训练上限为本轮自主制定的配置。所有失败、未完成与保护项取舍保留在阶段记录。",
        "长期路线和逐阶段Goal见LONG_TERM_PLAN.txt及long_term_goals。", f"完整数据、模型和预测位于{ROOT}。",
        "results保存轻量协议、曲线、结果和收据，delivery_manifest.json记录副本hash。"])
    (HERE / "RESULTS.txt").write_text("\n".join(lines), encoding="utf-8")
    destination = HERE / "results"
    destination.mkdir(exist_ok=True)
    copied = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in (".json", ".csv", ".png"):
            continue
        if "identities" in path.name or path.name == "generation_manifest.json" or path.stat().st_size > 5 * 1024**2:
            continue
        relative = path.relative_to(ROOT)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        digest = base.sha(target)
        assert digest == base.sha(path)
        copied.append({"path": str(relative), "bytes": target.stat().st_size, "sha256": digest})
    base.write(HERE / "delivery_manifest.json", {"status": "verified", "files": copied,
        "generated_utc": datetime.now(timezone.utc).isoformat(), "external_root": str(ROOT)})
    print("REPORT_AND_ARCHIVE_COMPLETE", len(copied), flush=True)


if __name__ == "__main__":
    main()
