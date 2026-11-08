# 长尾基线与分段读出

CPU 方案统一归入路线 A 的 `cjlt`，复用原有 `dfms_iq` 骨干。路线 B、W 弱视图覆盖方案及既有实验记录保持各自入口。当前实现面向 AMA 六类：`[FMB, AMA, SawFM, SinFM, MFT, ISRJ]`；不能与含 VFT 的基础六类直接交换权重或标签列。

## 实现位置

以下 Python 路径相对于 `projects/CJ_Imbalance_0716`。

| 位置 | 职责 |
|---|---|
| `cjlt/models/segment_readout.py` | 时间轴 1/2/4 段均值、标准差；零输出初始化残差头 |
| `cjlt/models/classwise_fusion.py` | 原骨干、FMB 专家、历史线性/局部/配对头与新增残差的融合 |
| `cjlt/data/ama_hdf5.py`、`label_sampling.py` | 显式标签契约、身份隔离、自然/标签/组合/混合采样 |
| `cjlt/engine/feature_transform.py`、`model_bundle.py` | 仅训练集拟合的标准化/PCA；完整单文件权重与旧模型导入 |
| `cjlt/engine/readout_trainer.py`、`longtail_workflow.py` | 冻结读出训练、基线训练、按种子 CPU 多进程运行 |
| `cjlt/engine/longtail_evaluation.py` | F1 校准、Exact、AP、误报、弱分量与功率差分层指标 |
| `experiments/train_longtail.py`、`train_readout.py`、`evaluate_readout.py` | 三个统一命令入口 |
| `configs/longtail/` | 基线、SawFM/MFT 分段读出、ISRJ 读出三份模板 |

物理仿真放在 `cjlt/data/signal_simulator.py`，仓库根目录的 `data_generation/python_longtail/generate.py` 提供数据生成命令。保留两个函数式测试文件，覆盖数值、融合和端到端训练流程。

## 方法与融合顺序

自然采样无放回；label-aware 按各标签正例频率的倒数累加权重；combo-aware 使已出现的组合等概率；mixed-combo 将自然和组合采样各占一半。三种损失为普通 BCE、正类权重 `(1-p)/p` 的 weighted BCE、训练时加 `logit(p)` 的 Logit Adjustment。先验仅从 fit 统计。

分段模块提取骨干池化前的时间特征。默认宽度 16 时为 128 通道，1/2/4 段统计得到 1792 维，经自然 fit 上的标准化和 PCA 降到 128 维，再通过 `128→32→目标标签数` 的 GELU 残差头。将 `source` 改为 `global` 即可做同输出维数、同头容量、同更新预算的全局控制。

旧候选的计算顺序完整保留：骨干输出 → 五标签线性头 → 局部头修正 SawFM/MFT/ISRJ → 配对头替换 SawFM/MFT 两列 → 分段残差 → FMB 专家替换 FMB 列。新增读出仅更新指定标签；原分支、骨干和阈值冻结。配对头虽独立增益有限，仍属于当前候选权重图，因此保留其加载和推理。

## 数据和选择协议

配置必须声明 `classes`、`train`、`development`、`split`。HDF5 包含 `iq_complex[L,N]`、`label_selected[6,N]`、`receiver_noise_seed`，可提供唯一 `sample_id`。I/Q 按整条信号 RMS 归一化。带 `per-label-v1` 功率真值时，读取并核对 `relative_power_dB`、`weak_component_mask` 与 `weak_gap_db`。

`split.json` 声明 fit、calibration、check、report 的索引；也兼容旧文件 `roles` 下的 `development_test` 作为 report。校验角色之间的样本身份和噪声身份均无交集。PCA 只看自然 fit；基线按 check 选择模型，在 calibration 拟合逐标签 F1 阈值。读出固定参考阈值，在 check 选择 epoch，未通过保护条件则保留零残差的 epoch 0。

保护条件为各标签 FPR 上升不超过 0.5 pp、目标弱召回下降不超过 1 pp、目标平均弱 AP 下降不超过 0.2 pp、AMA/SinFM 平均 F1 下降不超过 0.3 pp。它们是单轮选择约束，不能代替五种子正式收益判断。

训练完成写出 `config.json`、`history.json`、`model.pt`、`frozen.json`。权重包含所有分支、PCA 和阈值，加载时无需旧日期目录。显式 report 评估要求冻结记录，并核对模型 SHA、数据文件 SHA 和各角色身份顺序；训练不会自动读取 report 信号。评估同时输出冻结阈值和统一 0.5 阈值的指标，便于区分采样、损失和校准收益。新训练会重新拟合 fit PCA，不承诺重现旧多阶段训练的逐步轨迹。

小比例配对训练要求额外 fit HDF5：具有唯一 `sample_id`，相邻视图 `pair_side=0/1`，共享噪声，标签仅增加 `pair_target` 指定的一位；`target_gap_bin=0/1/2` 覆盖 SawFM/MFT 六个分层。配对噪声不能进入任何评估角色。`pair_views: 2` 表示每批 30 条自然样本和一对视图；自然对照设为 0。

## 运行

从仓库根目录执行，环境依赖使用根目录 `requirements.txt`。输出建议放到仓库外；每个输出目录必须尚不存在。默认五种子 2050–2054，24 epoch、每 epoch 250 次更新、batch 32；正式对照应复制模板，只改变待研究的采样或损失设置，固定同一身份池与预算。

```bash
export AMA_DATA_ROOT=/path/to/ama_pool
export AMA_REFERENCE_MODEL=/path/to/reference/seed{seed}.pt
export AMA_PAIR_FIT=/path/to/paired_fit.h5
export RUN_ROOT=/path/to/outputs

python projects/CJ_Imbalance_0716/experiments/train_longtail.py \
  --config projects/CJ_Imbalance_0716/configs/longtail/baseline.yaml \
  --output "$RUN_ROOT/bce" --jobs 5 --threads 4

python projects/CJ_Imbalance_0716/experiments/train_readout.py \
  --config projects/CJ_Imbalance_0716/configs/longtail/segment_readout.yaml \
  --output "$RUN_ROOT/sawmft" --jobs 5 --threads 4

python projects/CJ_Imbalance_0716/experiments/evaluate_readout.py \
  --model "$RUN_ROOT/sawmft/seed2050/model.pt" \
  --config projects/CJ_Imbalance_0716/configs/longtail/segment_readout.yaml \
  --role report --frozen "$RUN_ROOT/sawmft/seed2050/frozen.json" \
  --output "$RUN_ROOT/sawmft_report/seed2050"
```

`--jobs` 是并行种子数，`--threads` 是每个进程的骨干计算线程数；轻量读出默认使用 1 个线程。分配预算时考虑两者乘积与特征缓存内存。ISRJ 模板以已选 SawFM/MFT 模型为 reference，继续训练自然 ISRJ 残差。参考模型不能已有同名读出；若从完整旧候选重做消融，配置 `remove_readouts: [sawmft, isrj]` 或 `[isrj]`，并保持原阈值。

旧五种子最终候选可一次性转为单文件，以下操作仅作导入，不训练也不重选阈值：

```bash
python projects/CJ_Imbalance_0716/experiments/evaluate_readout.py \
  --model /path/to/accepted/selected_models.json --seed 2050 \
  --export-model /path/to/portable/seed2050.pt
```

导入器核对旧选择记录中的组件 SHA，要求接受标志和完整最终分支图；不泛化猜测其他历史格式。原始六类骨干可通过配置 `reference_format: backbone`、`backbone` 和 `reference_thresholds` 显式导入，使用者必须确认训练标签确为 AMA 顺序。

生成新的默认物理池时，命令沿用参考 manifest 中的 63 组合训练配额；`base-seed` 必须选用未使用过的身份域，生成后仍须与拟比较的历史池审计重叠。该生成器采用原物理方程、每分量独立 10–30 dB、Python PCG64，不宣称 MATLAB 随机流逐位一致。

```bash
python data_generation/python_longtail/generate.py \
  --reference-manifest /path/to/reference/dataset.json \
  --output /path/to/new_pool --base-seed 3900000000
```

## 效果、边界与比较方式

下表是既有五种子实验各自独立报告池上的均值，仅能在同一行内比较。

| 单轮比较 | 冻结参考 Exact | 同预算 global128 | 分段读出 | 相对全局控制 |
|---|---:|---:|---:|---:|
| SawFM/MFT | 74.68% | 74.82% | 75.72% | +0.91 pp |
| ISRJ | 75.04% | 75.07% | 76.72% | +1.65 pp |

最终候选在随后一轮 8064 条独立身份报告池上为 77.18%，五个模型分别测量后取均值。SawFM 更宽头虽有小幅点估计提高，未通过原实用收益门槛；可训练投影仅完成两种子筛选，因此均未并入默认架构。分段统计的收益仍需通过到达时间扰动和机制消融判断稳健性。

此次整理核对五种子各 8064 条历史缓存判定完全一致，逐种子另抽 64 条原始 I/Q 前向核对；最大 logit 差小于 `5e-6`，单文件重载逐值一致。另验证缩小预算的 CPU 训练、两进程种子并行和冻结报告评估。这验证实现迁移，没有重新证明完整训练收益。

与 W 等其他路线比较时，先确认 AMA 标签契约和骨干接口，再统一身份池、RMS、校准角色、阈值、训练预算及五种子。先固定同一参考权重，比较 W、分段读出、二者结合的增量，最后评估组合；不将不同报告池的分数直接排序。

旧 CPU 独立工程及其批次报告已移出源码树，仍可从 Git 历史追溯。既有 `experiment-log` 与原路线研究记录保留。`tools/verify_experiment_delivery.py` 默认在旧 manifest 自身的 Git 提交上验证历史交付；`--working-tree` 显式比较当前文件，正常目录调整也可能使历史哈希不匹配。当前实现的检查命令为：

```bash
cd projects/CJ_Imbalance_0716
python -m pytest tests/test_segment_readout.py tests/test_readout_workflow.py -q
```
