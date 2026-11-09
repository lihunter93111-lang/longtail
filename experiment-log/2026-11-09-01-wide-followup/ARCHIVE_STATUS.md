# 11月9日续做实验归档状态

北京时间2026年11月9日13时50分的停止收据记录 `status=stopped`、`goal_completed=false`。12时52分的RESULTS.txt、continuation_state.json和监督器状态保留了停止前的计划与运行状态。当前完成范围以停止收据和逐阶段结果为准。[停止收据](execution_stop.json)

## 已完成阶段

| 阶段 | 数值与判断 | 证据 |
|---|---|---|
| 分类头复验 | 两套划分共40次头拟合。标准学习率头raw BCE退化，低学习率硬头全部选回epoch0，均衡头未达2%收益门槛 | [头诊断](results_at_stop/head_review.json) |
| 40/80/160轮预算 | 三种子开发BCE分别为0.101195、0.106160、0.108286，按原规则选40轮 | [预算决策](results_at_stop/budget_decision.json) |
| 组合混合采样 | 五种子参考BCE 0.102849599、候选0.101100202，下降1.7009%，passes=false。逐类仿射校准BCE另存 | [采样比较](results_at_stop/sampling_primary_comparison.json)与[校准比较](results_at_stop/sampling_probability_review.json) |
| 数据轴试验 | 三种子参考BCE 0.101195453、候选0.108661520，增加7.3779%，三种子均退化。标记为开发诊断，eligible_for_final_confirmation=false | [试验比较](results_at_stop/axis_pilot_review.json) |
| 采样加正则组合分支 | 保存了首种子部分训练历史，停止时处于combined_primary阶段，完整组评估尚缺 | [部分历史](results_at_stop/combined_primary/M_2100/history.json)与[协议](results_at_stop/combined_primary/protocol.json) |

外部目录未出现本轮新的最终确认文件。11月8日确认结果仍使用其原始实验归属。

## 快照和核验范围

`results`与原delivery_manifest.json配套，413个原交付文件的哈希此次全部通过。`results_at_stop`补齐了外部目录后续生成的轻量文件，其来源与哈希记录在仓库发布清单。原verification.json记录518个标量、17次完整训练和40次头拟合的当时复核范围，包含一条等待组评估的记录；数据轴后续完成情况另查axis_pilot_review.json。

所有原始报告、协议、失败分支和中断记录保留，后续阅读以本页状态和具体收据时间为准。完整数据与预测目录为 `E:/LJQ_code/LJQ_code_experiment_storage/daytime_v1/20261109_01`。
