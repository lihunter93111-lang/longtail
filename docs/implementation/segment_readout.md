# 长尾基线、分段读出与冻结融合

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
| `cjlt/engine/selective_fusion.py` | 冻结 R/W 的逐标签仿射融合、选择性残差门控与独立阈值拟合 |
| `experiments/train_longtail.py`、`train_readout.py`、`evaluate_readout.py` | 三个统一命令入口 |
| `experiments/compare_longtail.py` | A/B/W 与 2×2 读出对照、四组冻结融合的统一编排和配对统计 |
| `configs/longtail/` | 基线、分段读出、弱视图与融合模板 |

物理仿真放在 `cjlt/data/signal_simulator.py`，仓库根目录的 `data_generation/python_longtail/generate.py` 提供数据生成命令。保留两个函数式测试文件，覆盖数值、融合和端到端训练流程。

`configs/longtail/weak_view.yaml` 补充主干 W 的 CPU 方法复现：40 轮端点余弦、hard BCE、按 check raw BCE 选择，以及与原实现相同的 181 点 F1 网格校准。A 使用固定学习率，B 使用相同余弦但无弱视图。三组使用同一种子初始化与身份顺序；每个可干预身份交替使用自然/弱视图，身份数、标签配额和更新次数不变。此配置默认随机初始化；缺少原 MATLAB 池和 F2 权重时，不能称为历史数值的严格复现。

生成命令增加 `--weak-views` 后，会在重建自然波形并逐值核对后，只将一个可行的 SawFM/MFT/ISRJ 分量降至相对其他最强分量 −10 dB，保存 `weak_fit.h5` 及单目标干预收据。弱视图仅绑定 fit 身份，不作为独立新样本。`data.cache: memory` 可缓存归一化后的训练、检查和校准信号；不会提前缓存报告信号。

fit 身份在每个组合内按固定种子 20261116 重排，弱视图的位置引用最终 `split.fit`，并在加载时核对。这样可避免连续物理 seed 的目标选择与原/弱视图切换的奇偶位置耦合；组合配额及每身份的呈现预算保持不变。

基线每轮写出进度和可恢复状态。同输出目录、同配置、同输入文件可继续未完成的基线训练，恢复模型、优化器、随机状态及身份顺序；输入或配置变化会拒绝续跑。已冻结的运行只核验后返回。轻量读出中断后会从该阶段重新训练。

## 方法与融合顺序

自然采样无放回；label-aware 按各标签正例频率的倒数累加权重；combo-aware 使已出现的组合等概率；mixed-combo 将自然和组合采样各占一半。三种损失为普通 BCE、正类权重 `(1-p)/p` 的 weighted BCE、训练时加 `logit(p)` 的 Logit Adjustment。先验仅从 fit 统计。

分段模块提取骨干池化前的时间特征。默认宽度 16 时为 128 通道，1/2/4 段统计得到 1792 维，经自然 fit 上的标准化和 PCA 降到 128 维，再通过 `128→32→目标标签数` 的 GELU 残差头。将 `source` 改为 `global` 即可做同输出维数、同头容量、同更新预算的全局控制。

旧候选的计算顺序完整保留：骨干输出 → 五标签线性头 → 局部头修正 SawFM/MFT/ISRJ → 配对头替换 SawFM/MFT 两列 → 分段残差 → FMB 专家替换 FMB 列。新增读出仅更新指定标签；原分支、骨干和阈值冻结。配对头虽独立增益有限，仍属于当前候选权重图，因此保留其加载和推理。

融合阶段冻结完整 R（历史双骨干）和 W+分段（单骨干）。静态融合为 `z = α·z_R + β·z_W + b`，每标签三个无约束仿射参数；选择性融合为 `z = z_R + sigmoid(w·u+c)·(a·z_W+b-z_R)`，其中 `u` 只包含两路 logits、差值和到已有阈值的边际。门控默认只开放 SawFM/MFT/ISRJ；训练使用 BCE、门控稀疏正则及参考预测锚定。推理不使用真实功率或标签，两路模型、融合状态和最终阈值一起存入 `cjlt-logit-fusion-v1` 单文件，不再依赖历史运行目录。现有 classwise v1 权重继续兼容。

融合只在 fit 拟合参数，再在 calibration 的 181 点网格上选 F1 阈值，并限制各标签 FPR 不超过同池 R +0.25 pp。`history.json` 另记录 FMB F1 下降≤0.2 pp、三目标平均弱召回下降≤0.5 pp 的保护检查。后两项是诊断，不会触发自动拒绝或回退；不能把运行完成或 Exact 提高解释为全部保护通过。

## 数据和选择协议

配置必须声明 `classes`、`train`、`development`、`split`。HDF5 包含 `iq_complex[L,N]`、`label_selected[6,N]`、`receiver_noise_seed`，可提供唯一 `sample_id`。I/Q 按整条信号 RMS 归一化。带 `per-label-v1` 功率真值时，读取并核对 `relative_power_dB`、`weak_component_mask` 与 `weak_gap_db`。

`split.json` 声明 fit、calibration、check、report 的索引；也兼容旧文件 `roles` 下的 `development_test` 作为 report。校验角色之间的样本身份和噪声身份均无交集。PCA 只看自然 fit；基线按 check 选择模型，在 calibration 拟合逐标签 F1 阈值。读出固定参考阈值，在 check 选择 epoch，未通过保护条件则保留零残差的 epoch 0。

保护条件为各标签 FPR 上升不超过 0.5 pp、目标弱召回下降不超过 1 pp、目标平均弱 AP 下降不超过 0.2 pp、AMA/SinFM 平均 F1 下降不超过 0.3 pp。它们是单轮选择约束，不能代替五种子正式收益判断。

训练完成写出 `config.json`、`history.json`、`model.pt`、`frozen.json`。权重包含所有分支、PCA 和阈值，加载时无需旧日期目录。显式 report 评估要求冻结记录，并核对模型 SHA、数据文件 SHA 和各角色身份顺序；训练不会自动读取 report 信号。评估同时输出冻结阈值和统一 0.5 阈值的指标，便于区分采样、损失和校准收益。新训练会重新拟合 fit PCA，不承诺重现旧多阶段训练的逐步轨迹。

小比例配对训练要求额外 fit HDF5：具有唯一 `sample_id`，相邻视图 `pair_side=0/1`，共享噪声，标签仅增加 `pair_target` 指定的一位；`target_gap_bin=0/1/2` 覆盖 SawFM/MFT 六个分层。配对噪声不能进入任何评估角色。`pair_views: 2` 表示每批 30 条自然样本和一对视图；自然对照设为 0。

## 运行

从仓库根目录执行，环境依赖使用根目录 `requirements.txt`。输出建议放到仓库外；新实验使用新目录，同配置、同输入的基线可续跑。默认五种子 2050–2054，24 epoch、每 epoch 250 次更新、batch 32；弱覆盖模板使用 40 epoch。正式对照应复制模板，只改变待研究的因素，固定同一身份池与预算。

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

### 统一实验比较

从仓库根目录使用同一个入口，无须复制每轮运行脚本。配置只声明数据和方法，运行目录承载种子、阶段、权重、预测及收据；程序不绑定服务器路径或 CPU 核编号。

```bash
# 先生成包含弱视图的新池；base-seed 使用尚未消费的身份域
python data_generation/python_longtail/generate.py \
  --reference-manifest /path/to/reference/dataset.json \
  --output /path/to/shared_pool --base-seed 4200000000 --weak-views
export AMA_DATA_ROOT=/path/to/shared_pool

# A/B/W 相同初始化、40轮预算；B/W × global/segment 两阶段自然读出
python projects/CJ_Imbalance_0716/experiments/compare_longtail.py \
  --mode shared --config projects/CJ_Imbalance_0716/configs/longtail/weak_view.yaml \
  --output "$RUN_ROOT/shared" --jobs 5 --threads 4 --report

# 另生成独立融合池，保持其与已有训练/评测身份不重叠
export AMA_FUSION_DATA_ROOT=/path/to/fresh_fusion_pool
export AMA_REFERENCE_MODEL='/path/to/historical/seed{seed}/model.pt'
export AMA_SPECIALIST_MODEL="$RUN_ROOT/shared/W_segment/isrj/seed{seed}/model.pt"
python projects/CJ_Imbalance_0716/experiments/compare_longtail.py \
  --mode fusion --config projects/CJ_Imbalance_0716/configs/longtail/fusion.yaml \
  --output "$RUN_ROOT/fusion" --jobs 5 --threads 4 --report
```

`shared` 模式比较七个同池方法，不重新训练历史 R；历史读出重训继续使用 `train_readout.py`。`fusion` 模式比较 R 独立阈值、全标签静态融合、SawFM 门控和三标签门控。两种模式均等全部候选冻结后才允许 report；不加 `--report` 只训练并冻结，随后原命令加该参数可继续评估。`plan.json` 绑定方法配置和种子，改变计划必须使用新输出目录。`--jobs` 限制总并行作业数，单作业线程由 `--threads` 指定。

只训练配置指定的一种融合时，使用现有 `train_readout.py --kind fusion --config .../fusion.yaml --output ...`；随后由 `evaluate_readout.py` 加载其 `model.pt` 和 `frozen.json`，与普通读出使用相同评估命令。

输出 `metrics.json`、`summary.json` 和逐种子的 `predictions.npz`；配对统计核对标签和身份顺序，给出 Exact/BCE 差值、五种子 t95、纠错与新增错误数。仅一个种子时不输出置信区间。`readout_training` 和 `readout` 配置块可缩小读出预算/维度作启动检查，正式协议使用默认两阶段各 24×250 步、PCA128、小头32。融合的训练步数和正则可在 `fusion` 块显式配置。

## 效果、边界与比较方式

下表是既有五种子实验各自独立报告池上的均值，仅能在同一行内比较。

| 单轮比较 | 冻结参考 Exact | 同预算 global128 | 分段读出 | 相对全局控制 |
|---|---:|---:|---:|---:|
| SawFM/MFT | 74.68% | 74.82% | 75.72% | +0.91 pp |
| ISRJ | 75.04% | 75.07% | 76.72% | +1.65 pp |

最终候选在随后一轮 8064 条独立身份报告池上为 77.18%，五个模型分别测量后取均值。SawFM 更宽头虽有小幅点估计提高，未通过原实用收益门槛；可训练投影仅完成两种子筛选，因此均未并入默认架构。分段统计的收益仍需通过到达时间扰动和机制消融判断稳健性。

此次整理核对五种子各 8064 条历史缓存判定完全一致，逐种子另抽 64 条原始 I/Q 前向核对；最大 logit 差小于 `5e-6`，单文件重载逐值一致。另验证缩小预算的 CPU 训练、两进程种子并行和冻结报告评估。这验证实现迁移，没有重新证明完整训练收益。

与 W 等其他路线比较时，先确认 AMA 标签契约和骨干接口，再统一身份池、RMS、校准角色、阈值、训练预算及五种子。先固定同一参考权重，比较 W、分段读出、二者结合的增量，最后评估组合；不将不同报告池的分数直接排序。

后续统一 Python 新身份实验中，W+分段为 70.486%，相对 B+分段 +1.620 pp，五种子 t95 为 [+0.210,+3.029] pp。另一个新融合池中，R 校准基线为 77.431%，静态融合为 78.400%（+0.970 pp，t95 [+0.870,+1.069] pp），SawFM/三标签门控分别为 78.038%/78.197%。这些是已完成实验的记录，本次整理没有重新训练全部模型。三种融合在两个种子上未通过校准弱召回保护；静态融合还有少量报告 FPR 超出校准容差。融合需要三个骨干，尚无同成本普通第三骨干控制。

旧 CPU 独立工程及其批次报告已移出源码树，仍可从 Git 历史追溯。既有 `experiment-log` 与原路线研究记录保留。`tools/verify_experiment_delivery.py` 默认在旧 manifest 自身的 Git 提交上验证历史交付；`--working-tree` 显式比较当前文件，正常目录调整也可能使历史哈希不匹配。当前实现的检查命令为：

```bash
cd projects/CJ_Imbalance_0716
python -m pytest tests/test_segment_readout.py tests/test_readout_workflow.py -q
```
