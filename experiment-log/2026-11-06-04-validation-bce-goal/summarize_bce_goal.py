"""Create measured curves, comparison tables, and a readable experiment report."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def summarize(output):
    protocol = load_json(output/"protocol.json")
    decision = load_json(output/"decision.json")
    with (output/"evaluation_metrics.csv").open(encoding="utf-8-sig",newline="") as stream:
        rows = list(csv.DictReader(stream))
    arms = protocol["arms"]
    seeds = protocol["seeds"]
    histories = {
        arm: [load_json(output/f"{arm}_{seed}/history.json") for seed in seeds]
        for arm in arms
    }
    plt.rcParams.update({"font.family":"Microsoft YaHei","font.size":11,"axes.unicode_minus":False,"axes.spines.top":False,"axes.spines.right":False,"savefig.dpi":180})
    colors = {"A":"#52677d","B":"#277eb2","C":"#dd7839","D":"#24866b"}
    names = {"A":"A 标准 BCE","B":"B 余弦衰减","C":"C 标签平滑","D":"D 衰减 + 平滑"}
    fig,axes = plt.subplots(1,3,figsize=(17,5.1),constrained_layout=True)
    for arm in arms:
        for axis,key,title in zip(axes,["train_hard_BCE","early_BCE","early_macro_AP"],["训练硬标签 BCE（训练模式）","选模集 BCE（eval 模式）","选模集 Macro-AP"]):
            values = np.array([[r[key] for r in run] for run in histories[arm]])
            if key == "early_macro_AP":
                values *= 100
            mean = values.mean(axis=0)
            deviation = values.std(axis=0,ddof=1)
            axis.plot(np.arange(1,41),mean,color=colors[arm],label=names[arm],linewidth=1.8)
            axis.fill_between(np.arange(1,41),mean-deviation,mean+deviation,color=colors[arm],alpha=.10)
            axis.set(title=title,xlabel="训练轮次",xlim=(1,40))
            axis.grid(alpha=.2)
    axes[0].legend(fontsize=9)
    axes[2].set_ylabel("Macro-AP (%)")
    role = "探索性筛选" if protocol["stage"]=="screen" else "新身份独立确认"
    fig.suptitle(f"验证 BCE 实验 · {role} · {len(seeds)} 种子均值与标准差",fontsize=15)
    fig.savefig(output/"training_curves.png")
    plt.close(fig)
    summary = []
    for arm in arms:
        for variant in ("raw","temperature"):
            group = [r for r in rows if r["arm"]==arm and r["variant"]==variant]
            record = {"arm":arm,"variant":variant,"seeds":len(group)}
            for key in ("BCE","macro_AP","fixed05_exact","f1_exact","macro_F1","tail_exact","ECE","Brier","confident_wrong_fraction","confident_wrong_loss_share"):
                values = np.array([float(r[key]) for r in group])
                record[key+"_mean"] = float(values.mean())
                record[key+"_sd"] = float(values.std(ddof=1))
            summary.append(record)
    with (output/"summary.csv").open("w",newline="",encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream,fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    fig,axes = plt.subplots(1,2,figsize=(11,4.7),constrained_layout=True)
    for axis,key,title in zip(axes,["BCE","f1_exact"],["评价集硬标签 BCE","F1 校准阈值下的组合 exact"]):
        scale = 100 if key=="f1_exact" else 1
        for offset,variant,label,color in [(-.18,"raw","原始概率","#4a7292"),(.18,"temperature","温度校准概率","#d88c4a")]:
            group = [next(r for r in summary if r["arm"]==arm and r["variant"]==variant) for arm in arms]
            means = [r[key+"_mean"]*scale for r in group]
            deviations = [r[key+"_sd"]*scale for r in group]
            axis.bar(np.arange(len(arms))+offset,means,width=.34,yerr=deviations,capsize=3,label=label,color=color)
        axis.set_xticks(np.arange(len(arms)),arms)
        axis.set(title=title,ylabel="%" if scale==100 else "BCE")
        axis.grid(axis="y",alpha=.15)
        axis.set_axisbelow(True)
    axes[0].legend(fontsize=9)
    fig.suptitle(f"{role} · 独立校准参数冻结后评价",fontsize=14)
    fig.savefig(output/"bce_comparison.png")
    plt.close(fig)
    lines = [f"验证 BCE 实验结果 — {role}","",f"训练设置：{len(seeds)}种子，每组40轮，fit8000，独立选模1008，校准1008，评价4032。", "checkpoint按选模集硬标签BCE最低选择。", ""]
    for record in summary:
        lines.append(f"{record['arm']} / {record['variant']}：BCE {record['BCE_mean']:.6f} ± {record['BCE_sd']:.6f}；AP {record['macro_AP_mean']*100:.4f}%；exact {record['f1_exact_mean']*100:.4f}%。")
    lines.extend(["","配对裁决（BCE下降为正）"])
    for name,comparison in decision["comparisons"].items():
        lines.append(f"{name}：相对下降 {comparison['relative_reduction']*100:.3f}%，逐种子收益 {comparison['paired_BCE_gain']}，95%身份bootstrap区间 {comparison['identity_bootstrap95_BCE_gain']}，通过 {comparison['passes']}。")
        lines.append(f"检查项 {comparison['checks']}")
    lines.extend(["","证据范围",decision["evidence_role"],"区间描述固定模型种子条件下的身份不确定性；种子波动由配对差标准差另报。", "筛选数据已消费，确认阶段使用全新生成身份。历史locked test保持封存。"])
    (output/"results.txt").write_text("\n".join(lines),encoding="utf-8")
    files = ["training_curves.png","bce_comparison.png","summary.csv","results.txt","decision.json","evaluation_metrics.csv","evaluation_labels.csv","evaluation_combinations.csv","freeze.json","protocol.json"]
    hashes = {name:hashlib.sha256((output/name).read_bytes()).hexdigest() for name in files}
    (output/"artifact_sha256.json").write_text(json.dumps(hashes,indent=2),encoding="utf-8")
    print(json.dumps({"summarized":str(output),"decision":decision}),flush=True)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    summarize(args.output)
