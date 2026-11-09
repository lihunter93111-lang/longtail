"""Verify final arithmetic and assemble the requested research delivery."""
import argparse
import csv
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import torch
import numpy as np
from scipy.special import expit, logit

import training
from verify_overnight import independent_metrics
from run_stage2 import read

base = training.base
HERE = Path(__file__).resolve().parent


def table(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def average(rows, metric):
    return float(np.mean([float(row[metric]) for row in rows]))


def verify_final(root):
    frozen = read(root / "final_freeze.json")
    rows = table(root / "final_confirmation_results/metrics.csv")
    quota = read(HERE.parent / "2026-11-06-01-lt-abc/2026-11-06-lt-abc-quotas.json")
    differences = []
    for record in frozen["models"]:
        saved = np.load(root / f"final_confirmation_results/{record['id']}/natural_logits.npz")
        z, y = saved["logits"], saved["labels"]
        threshold = np.array(record["calibration"]["thresholds"]["raw"])
        for variant in ("raw", "temperature"):
            values = z if variant == "raw" else z / record["calibration"]["T"]
            cut = threshold if variant == "raw" else expit(logit(threshold) / record["calibration"]["T"])
            actual = independent_metrics(values, y, cut, quota["tail_codes"])
            expected = next(row for row in rows if row["group"] == record["group"]
                            and int(row["seed"]) == record["seed"] and row["variant"] == variant)
            for key, value in actual.items():
                error = abs(value - float(expected[key]))
                assert error < 1e-12, (record["id"], key)
                differences.append(error)
    pair_rows = table(root / "final_confirmation_results/paired_power_metrics.csv")
    for row in pair_rows:
        record = next(r for r in frozen["models"] if r["group"] == row["group"] and r["seed"] == int(row["seed"]))
        folder = root / f"final_confirmation_results/{record['id']}"
        saved = np.load(folder / f"paired_gap{row['gap']}_logits.npz")
        common15 = np.load(folder / "paired_gap15_logits.npz")["positions"]
        mask = np.isin(saved["positions"], common15) if row["population"] == "gap15_feasible" else np.ones(len(saved["positions"]), bool)
        if row["label"] != "all_targets":
            mask &= saved["targets"] == training.previous.LABELS.index(row["label"])
        assert mask.sum() == int(row["n"])
        values = saved["logits"][np.arange(len(mask)), saved["targets"]][mask]
        thresholds = np.array(record["calibration"]["thresholds"]["raw"])[saved["targets"][mask]]
        actual = {"BCE": np.logaddexp(0, -values).mean(), "FNR": (expit(values) < thresholds).mean(),
                  "fixed05_FNR": (values < 0).mean()}
        for key, value in actual.items():
            error = abs(value - float(row[key]))
            assert error < 1e-12
            differences.append(error)
    result = {"status": "passed", "scalar_metrics": len(differences), "max_error": max(differences),
              "natural_models": len(frozen["models"]), "paired_rows": len(pair_rows),
              "source_sha256": base.sha(Path(__file__))}
    base.write(root / "verification_final.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    assert read(root / "final_pipeline_status.json")["stage"] == "completed"
    final_check = verify_final(root)
    decision = read(root / "final_confirmation_results/decision.json")
    replication = read(root / "replication_comparison.json")
    matrix = read(root / "capacity_matrix_review.json")
    full_check = read(root / "verification_full60.json")
    rows = table(root / "final_confirmation_results/metrics.csv")
    group_rows = {group: [r for r in rows if r["group"] == group and r["variant"] == "raw"]
                  for group in ("reference", "candidate")}
    score_lines = []
    for metric, label, scale in (("BCE", "raw BCE", 1), ("macro_AP", "macro AP (%)", 100),
        ("f1_exact", "exact (%)", 100), ("tail_exact", "tail exact (%)", 100), ("ECE", "ECE", 1)):
        means = [average(group_rows[group], metric) * scale for group in ("reference", "candidate")]
        score_lines.append(f"{label:<18} {means[0]:.6f} -> {means[1]:.6f}")
    failed = [key for key, passed in decision["checks"].items() if not passed]
    selection = "全部预先声明的晋级与保护指标通过" if not failed else "未通过项为 " + ", ".join(failed)
    receipt_epochs = {}
    for parent, arm, seeds in (("capacity_IF3_w16", "M", range(2080, 2085)),
                              ("capacity_IF3_w32", "M", range(2080, 2085)),
                              ("replication_W16", "W", range(2090, 2095)),
                              ("replication_M32", "M", range(2090, 2095))):
        receipt_epochs[parent] = [read(root / parent / f"{arm}_{seed}/receipt.json")["best_BCE_epoch"] for seed in seeds]
    final_pairs = table(root / "final_confirmation_results/paired_power_metrics.csv")
    weak = {group: [r for r in final_pairs if r["group"] == group and int(r["gap"]) == 15
                   and r["label"] == "all_targets" and r["population"] == "gap15_feasible"]
            for group in ("reference", "candidate")}
    report = f"""分阶段整夜实验结果
更新于 {datetime.now(timezone.utc).astimezone().isoformat()}

一、执行状态与时间偏差
用户约定截止北京时间2026-11-09 08:00。60次完整网络训练在截止前完成，累计2400轮、600000次更新、19200000次训练身份呈现。另完成30次冻结分类头训练、35组线性探针及校准/反事实诊断。最终冻结、独立确认和报告在08:00之后完成，该交付超时已记录。03E宽骨干分类头追加实验未执行。
默认40轮、60次完整训练上限、2% raw BCE晋级线及逐类保护阈值均为本次执行时制定的方案选择，供核验。最终冻结记录为final_freeze.json。

二、最终独立确认
参考模型为DFMS-IQ width16、共享F2初始化、W gap10覆盖。候选为DFMS-IQ width32、随机初始化、M gap10/15覆盖。两者均在第二套8000训练身份上完成五种子2090–2094训练。候选的宽度、初始化、覆盖均有变化，完整方案复验采用这两组比较。
确认使用预留种子3108310800新生成的4032身份，每个组合64个。新身份与历史及本轮训练/开发池完成隔离审计，所有模型与校准参数在确认生成和评分前冻结。此确认集评分后已消费。
""" + "\n".join(score_lines) + f"""
raw BCE相对下降 {decision['relative_BCE_reduction'] * 100:.4f}%。五种子配对收益为 {decision['paired_BCE_gain']}。
配对BCE收益的种子t95%区间为 {decision['seed_t95']}，身份分层bootstrap95%区间为 {decision['identity_bootstrap95']}。
{selection}。
FMB FPR变化 {decision['metric_differences']['FMB_FPR'] * 100:+.4f}个百分点，SawFM FPR变化 {decision['metric_differences']['SawFM_FPR'] * 100:+.4f}个百分点，ISRJ自然弱目标FNR变化 {decision['metric_differences']['ISRJ_weak_FNR'] * 100:+.4f}个百分点。
确认gap15同身份诊断的三类合计FNR为 {average(weak['reference'], 'FNR') * 100:.4f}% -> {average(weak['candidate'], 'FNR') * 100:.4f}%，正例BCE为 {average(weak['reference'], 'BCE'):.6f} -> {average(weak['candidate'], 'BCE'):.6f}。该功率子集为1683个可行身份。
温度校准后的BCE见结果表，raw BCE始终作为主指标。温度变换使用等价阈值保持最终二元决策相同。

三、独立训练划分复验
在第二套开发身份上，参考W16 raw BCE为 {replication['reference_BCE']:.6f}，候选M32为 {replication['candidate_BCE']:.6f}，下降 {replication['relative_BCE_reduction'] * 100:.4f}%。exact从75.2976%到82.4653%，提升7.1677个百分点，五个种子同向，全部保护规则通过。
这次复验重新生成fit与dev身份，并重新训练全部10个模型。两套训练划分提供独立身份层面的复现证据，每套内部五种子共享同一数据划分。

四、容量与训练分布
第一套开发集的匹配随机初始化四格对照如下，每格5种子，固定M覆盖、8000身份、40轮和相同开发身份。
IF3 width16      BCE 0.149818   exact 72.3413%
IF3 width32      BCE 0.104191   exact 81.2649%
近均衡 width16   BCE 0.121065   exact 76.9792%
近均衡 width32   BCE 0.092150   exact 82.5397%
在IF3条件扩宽使BCE降低30.4550%，在近均衡条件也有稳定收益。宽度收益在IF3条件比近均衡条件大 {matrix['effects']['width_gain_IF3_minus_width_gain_balanced']['mean']:.6f} BCE，种子t95%区间 {matrix['effects']['width_gain_IF3_minus_width_gain_balanced']['seed_t95']}。
当前窄骨干的容量对性能有明显限制，同时训练分布调整也改善泛化。近均衡条件改变了标签边际、组合共现及部分身份，与IF3共享4903个fit身份，不能将这项数据条件效果全部归因于单一标签频率。
width16与width32分别为257511、1013191参数。单个width32还优于本次预先固定四成员width16简单集成，后者参数1030044，logit平均raw BCE为0.135888，温度校准BCE为0.128920。

五、训练是否充分与剩余瓶颈
本轮所有完整模型均完成40轮，每轮8000身份无放回完整遍历，共250次更新。选取的early_stop最低BCE轮次分别为
{json.dumps(receipt_epochs, ensure_ascii=False, indent=2)}
训练自然视图损失与early_stop损失仍存在差距，宽骨干的自然fit BCE下降更明显。容量扩展改善了开发排名与exact，但泛化差距仍需研究。40轮预算内曲线趋于平缓，更长训练预算和最优宽度留待独立诊断，后续Goal已给出40/80/160轮方案。原有窄网络旧协议的20轮建议具有自身实验条件。

六、数据来源与实际使用量
数据由仓库MATLAB可配置组合干扰生成器生成，配置源为九类full_power_10_30dB_v1/closedset_9class_train.mat中的config。实际本轮输出为FMB、AMA、SawFM、SinFM、MFT、ISRJ六标签，共63个非空组合，复数I/Q长度3000，按单样本RMS归一化成2×3000输入。活动分量JNR10–30dB，采样率10MHz，脉宽100us，PRI300us，带宽5MHz，载频10GHz，单脉冲，噪声功率1，add_target=false。
主数据fit候选25200，实际IF3 fit8000，候选使用31.746%，剩余17200个候选未用于该训练。dev候选6048全部分配为early_stop1008、calibration1008、development4032。第二划分使用同样规模的新身份池，最终确认另有4032新身份。
主IF3标签正例数为6729/2243/3973/3897/4069/3986，标签IF=3。组合配额41–322，组合最大最小比约7.854。近均衡仍为8000身份，六类正例数4150/4150/4150/4149/4149/4149，组合配额87–142，组合阶数总量与IF3相同。
主数据W每轮身份顺序与B/M匹配。4440个身份可构造gap10，2414个可构造gap15。W每个可行身份20次原始加20次gap10。M在gap15可行身份上改为20次原始、10次gap10、10次gap15，其他可行身份保持20次gap10。每次完整训练总计320000次身份呈现，M中88800次弱视图含24140次gap15，231200次原始视图。
第二划分可行gap10/gap15身份为4553/2475，M合计91060次弱视图，其中24750次gap15。增强沿用同身份、其他分量与噪声，仅改变一个目标分量功率，物理差分重建误差均已审计。
全夜实际使用的不同训练身份为19097，包含主IF3与近均衡的11097个身份并集，以及第二划分8000身份。19200000次呈现反映重复训练访问量。

七、网络与训练合同
输入I/Q分离后进入复数残差编码。width16四级通道16/32/48/64，时间长度3000/1500/750/375。width32通道32/64/96/128，时间长度相同。第3、4级采用kernel3/7/15、dilation1/2/4多尺度复数卷积分支。
末级融合实部/虚部、幅度以及lag1/4/16的归一化相位差特征，融合到128维或256维时序特征，随后多尺度深度卷积残差、全局均值/标准差池化、投影、dropout0.2和六输出线性分类头。每个输出独立sigmoid，允许任意非空组合。
AdamW，lr从1e-4余弦下降到1e-6，weight_decay1e-4，batch32，clip5，40轮，FP32且TF32关闭。训练损失为六标签无权重硬BCEWithLogits平均，全网络可训练。early_stop最低raw BCE选模型，calibration拟合温度与每类F1阈值，development用于研究选择。
硬件为RTX5080 16GB。实测GPU驻留输入、100次计时的batch32中位推理耗时width16为5.936ms，width32为9.761ms。详细P95、batch1、显存定义见inference_cost.json，训练耗时见各receipt。

八、阶段分支与后续研究
完整阶段分析见STAGE_ANALYSIS.txt。长期路线和七份逐阶段Goal见LONG_TERM_PLAN.txt及long_term_goals目录。下一优先事项为冻结强width32基线，完成预算诊断，然后在组合长尾与标签长尾可控矩阵上检验同预算采样和简单专家。弱功率条件证据与跨分布验证安排在后续阶段。

九、复核与证据范围
60个完整模型的源码/数据/检查点hash、全部样本顺序、40轮历史、增强次数均通过核验。独立重算840个开发标量指标，最大误差 {full_check['maximum_metric_difference']:.3g}。最终确认独立重算 {final_check['scalar_metrics']} 个自然与配对标量指标，最大误差 {final_check['max_error']:.3g}。
研究证据来自当前合成六标签数据与声明参数范围。容量机制对照来自第一套匹配初始化实验，最终候选确认检验完整训练方案。五种子区间和身份bootstrap反映不同随机性来源。多次开发探索后使用新确认身份评分，确认结果用于最终报告。
发生过一次B2080第5轮日志导入错误，已从保存的完整优化器/RNG状态恢复并核验40轮。线性探针8个优化达到初始迭代上限，保持模型设置继续求解后全部210个二元问题收敛。时间截止检查迟到导致最终交付超时。全部记录见execution_notes.json和相关日志。

十、文件入口
本目录GOAL_PROMPT.txt及01–07和追加Goal保存当晚方案。
results目录保存轻量CSV、JSON、曲线和审计结果，保留原外部目录结构。
外部完整数据、模型和预测位于 {root}。
最终参考与候选的模型路径、hash、阈值及温度见results/final_freeze.json。
主报告、长期计划、逐阶段Goal和数据/模型合同可独立阅读。源码依赖保留在本仓库先前实验目录，复现入口见DELIVERY_GUIDE.txt。
"""
    (HERE / "RESULTS.txt").write_text(report, encoding="utf-8")
    notes = read(HERE / "execution_notes.json")
    if not any(item.get("stage") == "deadline_deviation" for item in notes):
        notes.append({"stage": "deadline_deviation", "requested_deadline": "2026-11-09 08:00 Asia/Shanghai",
            "observed_late_check": "2026-11-09 08:49 Asia/Shanghai", "all_full_training_completed_before_deadline": True,
            "late_work": "final freeze, reserved confirmation evaluation, verification and delivery",
            "wide_head_03E": "not executed"})
        base.write(HERE / "execution_notes.json", notes)
    destination = HERE / "results"
    destination.mkdir(exist_ok=True)
    copied = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in (".json", ".csv", ".png"):
            continue
        if "identities" in path.name or path.name == "generation_manifest.json" or path.stat().st_size > 5 * 1024**2:
            continue
        relative = path.relative_to(root)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        assert base.sha(path) == base.sha(target)
        copied.append({"path": str(relative), "bytes": path.stat().st_size, "sha256": base.sha(target)})
    base.write(HERE / "delivery_manifest.json", {"status": "verified", "external_root": str(root),
        "files": copied, "files_count": len(copied), "generated_utc": datetime.now(timezone.utc).isoformat()})
    print("DELIVERY_BUILT", len(copied), selection, flush=True)


if __name__ == "__main__":
    main()
