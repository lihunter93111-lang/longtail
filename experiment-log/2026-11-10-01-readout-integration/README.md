# 宽骨干专家与局部读出整合实验

2026年11月10日完成40次完整骨干训练、180次残差头训练，18组候选在两套新确认身份上完成五种子评价，9组方法完成物理配对诊断。

## 阅读顺序

1. [结果与局限](RESULTS.txt)
2. [阶段探索和阴性结果](STAGE_ANALYSIS.txt)
3. [目录作用 数据角色和调用链](DELIVERY_GUIDE.txt)
4. [长期方案](LONG_TERM_PLAN.txt)与[阶段Goal](long_term_goals)
5. [双方分工和两份Word](../../docs/research/collaboration/2026-11-11/README.md)

## 新确认结果

Exact为六标签整向量完全正确率，表中为每划分五种子均值。

| 方法 | 第一划分 | 第二划分 |
|---|---:|---:|
| W32单模型 | 82.68% | 81.74% |
| 普通双模型融合 | 84.55% | 83.77% |
| 组合加权专家融合 | 85.46% | 84.18% |
| 组合加权加保护式分段读出 | 85.54% | 84.47% |

联合方法相对普通双模型两套各5/5种子Exact改善。局部读出第一划分的增量区间跨零，物理召回收益形式随划分变化。均值保护通过，逐种子全部保护通过为各1/5。第二划分SinFM弱召回下降约1.10个百分点，完整逐标签代价见结果文件。

每划分另使用4032个组合均衡监督身份拟合融合，普通双模型与各仿射控制共享该池。主要方案延迟约为直接W32的2.11倍，较普通双模型增加约3.9%。四条件双长尾配额完成设计，新的配额训练尚待执行。

## 目录与来源

根目录保存协议、Goal和运行入口，long_term_goals保存后续七阶段提示词，results保存发布时核验过的轻量结果、源码演变收据、图表与数据角色清单。delivery_manifest.json绑定原745个交付文件，本README作为本次协作导航新增。

实验依赖源码固定在提交a8fb5da9c342503c84c3eab63e20b13a20aa7636，仓库历史包含该版本。复现实验可单独检出该提交作为依赖源码，将本目录作为编排与评价入口。common.py及protocol.json记录原机路径，迁移时创建新的协议和输出目录，按DELIVERY_GUIDE配置依赖、数据、已有基线与存储路径。

## 外部数据与模型

本轮原存储路径为 `E:/LJQ_code/LJQ_code_experiment_storage/readout_integration_v1/20261110_01`，既有W32基线位于 `E:/LJQ_code/LJQ_code_experiment_storage/w32_validation_v1/20261109_01`，依赖源码快照位于 `E:/LJQ_code/longtail_20261110`。

发布时E盘未挂载。本次核验本地交付副本与记录哈希，外部大文件本次可用性待挂载后核验。模型及阈值路径与哈希见results中的final_freeze.json，数据角色与身份预算见data_usage_receipt.json及各角色manifest。按这些记录向实验负责人取得大文件，复制后核对哈希并更新本机路径。仓库既有ModelScope数据入口与本轮新增大文件的发布范围分别以各自清单为准。

## 文件校验

在仓库根目录运行以下命令。它检查本轮745个发布副本，运行训练还需要完整外部数据与模型。

```powershell
python tools/verify_experiment_delivery.py --manifest experiment-log/2026-11-10-01-readout-integration/delivery_manifest.json --working-tree
```

最终确认前已冻结科学执行代码、模型和阈值，独立复算主指标4160个、物理指标14400个。早期评价脚本有评分后记录哈希的留痕缺口，详见RESULTS。发布过程保留原始科学执行文件字节。
