"""Package completed measured curves; never reads evaluation predictions."""
import argparse, csv, hashlib, json, shutil
from pathlib import Path
import numpy as np


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args();out=args.output;dest=Path(__file__).resolve().parent
    status=json.loads((out/'status.json').read_text(encoding='utf-8'));assert status['status']=='completed'
    selection=json.loads((out/'selection.json').read_text(encoding='utf-8'));n=selection['selected_max_epochs']
    runs=[json.loads((out/f'A_{s}/history.json').read_text(encoding='utf-8')) for s in (2050,2051,2052)]
    receipts=[json.loads((out/f'A_{s}/receipt.json').read_text(encoding='utf-8')) for s in (2050,2051,2052)]
    assert all(len(r)==150 and len({v['epoch'] for v in r})==150 for r in runs)
    ap=np.array([[r['early_stop_macro_AP'] for r in h] for h in runs]);best=np.maximum.accumulate(ap,axis=1)
    vb=np.array([[r['early_stop_BCE'] for r in h] for h in runs]);min_mean_bce_epoch=int(np.argmin(vb.mean(0)))+1
    checkpoints=[]
    for epoch in sorted({8,10,20,30,50,100,150,n}):
        fields={}
        for key in ('train_BCE','early_stop_BCE','early_stop_macro_AP'):
            v=np.array([h[epoch-1][key] for h in runs]);fields[key+'_mean']=float(v.mean());fields[key+'_std']=float(v.std(ddof=1))
        fields['best_so_far_AP_mean']=float(best[:,epoch-1].mean());checkpoints.append(dict(epoch=epoch,**fields))
    with (out/'summary_checkpoints.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(checkpoints[0]));w.writeheader();w.writerows(checkpoints)
    gap=best[:,-1]-best[:,n-1]
    lines=[f'# 10–150 轮训练诊断：建议基线训练预算 {n} 轮','',f'**结论：当前固定学习率 1e-4 下，建议 A 基线的最大训练轮数采用 {n}，训练过程中仍按 early-stop macro-AP 保存最佳 checkpoint。** 该预算由三个种子的完整 150 轮曲线和事先固定的容差规则选出。','',
        '本轮为训练预算诊断，三种子为 2050、2051、2052。每次 fit 8000、early-stop 1008，150 epoch × 250 updates = 37500 updates；三次合计 112500 updates。模型仍为 dfms_iq / width16 / dropout0.2，F2 共享权重初始化后全模型微调；AdamW lr=1e-4、weight_decay=1e-4、batch32、clip5、FP32、无学习率调度。','',
        '![三种子曲线](loss_curves.png)','','![相同 eval 模式下的 loss 对比](eval_loss_comparison.png)','','## 关键轮次实测值','',
        '下表为三种子均值；AP 列用百分数，完整逐种子值与标准差见 CSV。训练 BCE 为 Dropout 开启时的 minibatch loss，early-stop BCE 为 eval 模式；二者的同模式比较见第二张图。','',
        '| 轮次 | 训练 BCE | early-stop BCE | 当轮 early-stop macro-AP (%) | 截至当轮最佳 AP 均值 (%) |','|---|---:|---:|---:|---:|']
    for r in checkpoints:lines.append(f'| {r["epoch"]} | {r["train_BCE_mean"]:.5f} | {r["early_stop_BCE_mean"]:.5f} | {100*r["early_stop_macro_AP_mean"]:.4f} | {100*r["best_so_far_AP_mean"]:.4f} |')
    lines.extend(['','## 如何确定轮数','',
        '在开训前固定候选预算为 10、20、……、150。对每个种子取该预算以内的最佳 early-stop macro-AP；选择相对完整 150 轮最佳值，平均差不超过 0.001 AP（0.1 个百分点）、且任一种子差不超过 0.002 AP 的最早候选预算。这是工程容差，不是统计等效检验。','',
        f'选择 {n} 轮时，相对 150 轮的三种子平均最佳 AP 缺口为 **{100*gap.mean():.4f} 个百分点**，最大单种子缺口为 **{100*gap.max():.4f} 个百分点**。三种子平均 early-stop BCE 最低点在 **第 {min_mean_bce_epoch} 轮**。','',
        '| 种子 | 建议预算内最佳 AP 轮次 | 150 轮内最佳 AP 轮次 | early-stop BCE 最低轮次 |','|---|---:|---:|---:|'])
    for r in selection['per_seed']:lines.append(f'| {r["seed"]} | {r["selected_budget_best_epoch"]} | {r["peak_AP_epoch_150"]} | {r["min_early_BCE_epoch"]} |')
    lines.extend(['','训练 loss 继续下降而 early-stop loss 上升，说明继续拟合训练样本并不能保证泛化改善。训练预算与最后采用的 checkpoint 轮次是两件事，应保留验证最优的 checkpoint，而非直接使用最后一轮。','',
        '## 证据边界与下一步','',
        '- 本轮只训练 A 基线，结论适用于当前数据、模型和固定学习率；尚未证明 B/C 的收敛轮次相同。','- calibration / development / locked-test 的波形均未作为模型输入或用于本次选轮；仅核对源容器文件的整体 SHA-256。','- 原 8 轮 A/B/C 的结果保持独立。要更新方法比较，下一轮需冻结共同的新训练预算并重跑 A/B/C，不能将本轮 A 与旧 8 轮 B/C 混比。','- 新训练复用了原初始化和前三种子的前 8 轮身份排列，收据已验证；GPU 缓存输入及 CUDA 数值路径不承诺与旧运行逐位相同。','',
        f'外部完整产物：`{out}`。三次训练耗时合计 {sum(r["elapsed_seconds"] for r in receipts):.1f} 秒（不含数据读取、作图与汇总）。','',
        '原始逐轮曲线：[curves.csv](curves.csv)；10–150 轮逐 10 轮记录：[checkpoints_10_150.csv](checkpoints_10_150.csv)；预算选择：[horizon_selection.csv](horizon_selection.csv)；完整选择收据：[selection.json](selection.json)。'])
    if n==150:lines.extend(['','**边界：规则仅接受了 150 轮上限；本次不能证明已经在更早轮次进入稳定平台。**'])
    (dest/'results.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    names=['preflight.json','curves.csv','checkpoints_10_150.csv','horizon_selection.csv','summary_checkpoints.csv','selection.json','loss_curves.png','eval_loss_comparison.png']
    for name in names:shutil.copy2(out/name,dest/name)
    hashes={name:digest(dest/name) for name in names};(dest/'artifact_sha256.json').write_text(json.dumps(hashes,indent=2),encoding='utf-8')
    print(json.dumps(dict(selected_max_epochs=n,min_mean_early_BCE_epoch=min_mean_bce_epoch,mean_gap_pp=100*gap.mean(),max_seed_gap_pp=100*gap.max(),checkpoints=checkpoints),indent=2))


if __name__=='__main__':main()
