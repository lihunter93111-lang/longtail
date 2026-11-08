# 标签长尾与功率不均衡共同作用下的弱分量识别

研究输入复数 I/Q 时，训练标签长尾与混合分量功率不均衡如何共同造成弱分量漏检，以及如何在控制误报的同时提高尾类弱分量识别率。类别固定为 `[FMB, AMA, SawFM, SinFM, MFT, ISRJ]`，输出为六类存在概率；弱标记用于按真实可见度评测，无需模型在推理时获得功率真值。

保留 **DFMS-IQ（CNN）+ BCE + 普通随机采样** 作为可执行起点，同时提供纯线性分块与自注意力的 Transformer 示例；研究目标与训练、评测接口不绑定 CNN。已有 v3 仅作为标签长尾分类参考，联合任务使用重新生成且保留逐分量真值的数据。

历史实验说明保存在 [doc.md](doc.md)。其中的旧工程路径和历史成绩属于原实验；当前可执行入口、数据路径与配置以本文为准。

## 环境

使用 Python 3.10、uv 和清华 PyPI 源。依赖统一定义在 [pyproject.toml](pyproject.toml)，精确版本及下载哈希锁定于 [uv.lock](uv.lock)。

```bash
uv sync --locked
uv run pytest -q
```

本机已构建 `.venv`，PyTorch 为 2.5.1 + CUDA 12.4。CUDA 训练需要兼容驱动；`train.device` 可设置为 `auto`、`cuda` 或 `cpu`。

## 改动应放在哪里

| 需求 | 修改位置 | 影响范围 |
|---|---|---|
| RMS 归一化、去直流、裁剪、步长、幅度缩放 | JSON 的 `dataloader.preprocess` | 训练与测评共用同一变换 |
| batch、worker、pin memory、内存缓存 | JSON 的 `dataloader` | 数据加载行为；batch 为所有 GPU 的总 batch |
| 更换或新增模型 | 新建一个模型文件，修改 `model.entrypoint` 和 `model.kwargs` | 不修改训练、dataloader 或测评代码 |
| epochs、seed、学习率、优化器、损失、AMP、调度器 | JSON 的 `train` | 训练策略 |
| 阈值网格、校准方式、测评 batch | JSON 的 `evaluation` | 训练结束时的校准及后续测评 |
| 数据位置与角色拆分 | JSON 的 `data` | 独立于模型结构 |

主任务入口为 [configs/joint.json](configs/joint.json)，继承 [configs/baseline.json](configs/baseline.json) 中的公共训练设置；后者也保留旧 v3 的参考入口。基础 JSON 只保留数据入口、模型入口与本次实验的 seed、epochs、学习率、weight decay。公共运行默认值集中于 `src/longtail/config.py::DEFAULTS`；结构默认值在各模型构造函数中。已移除配置中的重复 class 数、空初始化、恒定关闭的开关及默认预处理参数；需调整时只补充相应字段。支持 `extends` 继承；运行时会展开公共默认值及模型构造参数并保存完整 JSON，测评无需原始父配置。自定义工厂建议显式声明可配置参数的默认值；若只接收 `**kwargs`，应将影响结果的参数显式填入模型 JSON。未知公共参数会报错。`run.output_dir` 支持 `{seed}`、`{signature}`，不同配置自动生成不同目录，已有运行不会被静默覆盖。

```text
src/longtail/
├── config.py                  # JSON 解析、校验和路径解析
├── data/
│   ├── manage.py              # 数据工作副本整理与核验
│   ├── dataset.py             # HDF5 I/Q 读取和原始信号缓存
│   ├── signals.py / generate_files.py  # Python 六类信号与 HDF5 生成
│   ├── generation.py / joint.py  # 生成入口、配对视图整理
│   ├── power_truth.py         # 逐标签功率契约与弱标记校验
│   ├── preprocess.py          # 纯信号预处理
│   ├── splits.py              # 按组合分层和身份隔离的数据角色
│   └── loaders.py             # DataLoader 与 worker 随机种子
├── models/
│   ├── factory.py             # 按 module:function 导入模型
│   ├── dfms.py                # DFMS-IQ 结构，257,511 参数
│   ├── transformer.py         # 线性分块与自注意力，无卷积
│   └── components/            # 复数卷积、编码器、融合、时序池化
├── engine/                    # DDP、训练计算、推理、checkpoint
├── metrics.py                 # 指标与校准
├── weak_metrics.py            # 弱分量、尾类与功率差分层指标
├── train.py                   # 训练入口
└── evaluate.py                # 独立测评入口
```

新增模型只需提供 `build_model(num_classes, **kwargs) -> torch.nn.Module`，接受 `[B, 2, L]`，输出 `[B, 6]` 原始 logits。`num_classes` 从数据 manifest 自动传入，不必在实验 JSON 重复填写。例如新增 `src/longtail/models/my_model.py` 后设置：

```json
{
  "model": {"entrypoint": "longtail.models.my_model:build_model"}
}
```

公用结构从 `models/components` 导入，不需要修改中央注册表。切换 `model.entrypoint` 时会清除旧模型的 kwargs，避免 CNN 的 width 等参数误传给 Transformer。非 CNN 示例使用 `configs/transformer.json`，联合任务使用 `configs/joint_transformer.json`。

## 训练与测评

单 GPU 或 CPU：

```bash
uv run longtail-train --config configs/joint.json
```

两张 GPU 使用 torchrun + DDP，NCCL 同步梯度，按 rank 划分训练数据：

```bash
CUDA_VISIBLE_DEVICES=0,1 uv run torchrun --standalone --nproc_per_node=2 \
  -m longtail.train --config configs/joint.json
```

全局 batch 必须能被进程数整除。训练样本数不能整除进程数时，DistributedSampler 的补齐数量记录在 `history.json`。验证由 rank 0 对完整角色执行，不通过带补齐的分布式验证采样器，避免重复统计。`train.sync_batchnorm=true` 可在多 GPU 间同步 BN；默认使用各 rank 的 BN 统计，因此单 GPU 和多 GPU 数值结果不保证相同。

[configs/smoke.json](configs/smoke.json) 是 width=4、1 epoch 的快速检查配置；CPU 使用时将 AMP 改为 `none` 或 `bfloat16`。

每个运行目录包含：

| 文件 | 用途 |
|---|---|
| `config.json` | 完整的模型、预处理、训练与测评配置 |
| `dataset.json` / `split.json` | 数据契约、源文件哈希、固定角色索引及划分哈希 |
| `last.pt` | epoch 边界恢复；模型、优化器、调度器、scaler 和各 rank 随机状态 |
| `best.pt` | 最优模型及训练结束后冻结的逐标签阈值 |
| `decision.json` | 阈值来源及绑定的数据划分 |
| `history.json` / `status.json` | 训练过程、设备与最终状态 |

断点恢复要求配置、数据和进程数保持一致：

```bash
CUDA_VISIBLE_DEVICES=0,1 uv run torchrun --standalone --nproc_per_node=2 \
  -m longtail.train --config configs/joint.json --resume RUN_DIR/last.pt
```

测评只指定 checkpoint，自动读取同目录 `config.json` 和 `dataset.json`，复用模型与预处理，使用 checkpoint 中已冻结的阈值。配置或划分不匹配会报错，测评不重新拟合阈值。

```bash
uv run longtail-evaluate --checkpoint RUN_DIR/best.pt
uv run longtail-evaluate --checkpoint RUN_DIR/best.pt --split check
# 数据搬迁后，保持内部相对路径并指定新的根目录
uv run longtail-evaluate --checkpoint RUN_DIR/best.pt --data-root /new/path/longtail
```

结果保存到 `RUN_DIR/evaluation/<split>/metrics.json` 和 `predictions.npz`，包含固定 0.5 与校准决策、逐类指标、组合指标、原始 logits、行索引和身份。报告 Macro-AUPR、Macro-F1、Hamming、样本 exact、宏组合 exact、稀有组合、组合阶数和 ECE/Brier。

## 旧 v3 参考数据

当前数据根目录为 `/ultrafast_share/xuwang_group/wxwu/datasets/longtail`：

```text
longtail/
├── dataset.json
├── backup.json
├── backup/                    # 原始基础六类、九挡功率、v1/v2/v3 全部移入此处
└── ama_v3/
    ├── train.mat              # 8,000 条长尾训练样本
    ├── development.mat        # 6,048 条均衡组合开发样本
    └── source_audit.json      # 原 v3 审计及派生来源
```

保留文件来自 [ModelScope fulanya/longtail](https://modelscope.cn/datasets/fulanya/longtail) 的 `six_class_pure_label_longtail_v3_IF3_CF8_N8000_AMA`。原始波形保持逐字节不变；目录中的旧 `test` 重命名为 `development`，明确其已消费开发证据的角色。**全部原始数据保留，未删除。** 原 `closedset_6class`、九挡功率、v1/v2 和重复的原始 v3 已通过同文件系统移动归入 `backup/`，保留原始内部目录和文件名。旧参考配置使用 `ama_v3/` 工作副本，当前联合任务使用下文的新数据。当前 manifest 见 [data/dataset.json](data/dataset.json)，原始 36 文件清单见 [data/source_snapshot.json](data/source_snapshot.json)，其相对路径现在位于 `backup/` 下。

- 每条样本为 3,000 点复数 I/Q；读取 `iq_complex` 的实、虚部，不依赖缺失的 `data_legacy`。
- 训练集覆盖全部 63 个非空组合，标签计数为 `[6729,2243,3973,3897,4069,3986]`，IF=3；每组合 41–322 条，CF≈7.85366。
- development 每组合 96 条，按固定种子 20260905 分为 calibration 16、check 16、development-test 64，即 1008 / 1008 / 4032 条。
- 训练与开发池的 receiver seed 无交集，三个开发角色互不重叠。默认在 **check 选 checkpoint，calibration 拟合 F1 阈值，development-test 报告结果**。
- v3 不含逐分量功率与参数回归真值，不能用于弱分量分层结论。经核对，`component_nominal_JNR_dB` 的 HDF5 形状为 `[1,N]`，旧构建脚本逐条执行 `.mean()`；它是标称 JNR 均值。详细证据与文件哈希见 [data/power_truth_audit.json](data/power_truth_audit.json)。

默认使用 `dataloader.cache="memory"`，一次解压原始 HDF5 I/Q，在多个角色之间共享只读缓存，预处理仍在 dataloader 中执行。内存不足可改为 `none`，数值行为一致，但当前小块 gzip 文件的随机读取会较慢。

```bash
uv run longtail-data verify --root /ultrafast_share/xuwang_group/wxwu/datasets/longtail
```

数据整理工具将原 ModelScope v3 布局复制为上述目录，校验副本但保留原始数据：`uv run longtail-data prepare --source-root SOURCE --output-root ROOT`。`backup-unused --root ROOT --inventory data/source_snapshot.json` 可按原始清单把旧版本移动到 `backup/`；目标冲突会报错，不覆盖现有数据。工具不提供数据删除命令。

## Python 数据生成与 MAT 读取

Python 可以直接读取现有 MAT，无需安装 MATLAB。当前 v3 是 v7.3/HDF5 格式，工程已通过 `h5py` 读取其中的复数 I/Q 和标签；传统 v5/v7 MAT 则可使用 `scipy.io.loadmat`。读取 v3 的例子：

```python
import h5py
with h5py.File("../datasets/longtail/ama_v3/train.mat", "r") as f:
    iq = f["iq_complex"][:, :2]        # HDF5 中为 [L, N]，实虚部字段 real/imag
    labels = f["label_selected"][:, :2].T
```

新数据由 [signals.py](src/longtail/data/signals.py) 和 [generate_files.py](src/longtail/data/generate_files.py) 纯 Python 生成，使用 NumPy/SciPy/h5py 环境，不再需要 MATLAB。沿用旧生成器的六类物理公式和参数范围，但随机流使用 NumPy PCG64，因此构成一套新的可复现数据，不声称与 MATLAB 逐点相同。MATLAB 参考源码已移入本机 `backup/generation_matlab_20261017/`。

| 类型 | 生成定义 |
|---|---|
| FMB | 实高斯噪声经 FFT 带限与峰值归一化后调频，对瞬时频率积分生成复包络 |
| AMA | 单位 RMS 的带限噪声构成 `1 + modulation_index * noise` 调幅包络 |
| SawFM | 锯齿瞬时频率，带宽 2–5 MHz、周期 20–80 μs |
| SinFM | 正弦瞬时频率，中心频率 0.5–1.5 MHz、频偏 1–3 MHz、调制频率 10–20 kHz |
| MFT | 对参考 LFM 生成 3–8 个不同整数时延与相对幅度的假目标 |
| ISRJ | 参考 LFM 经周期间歇截取，将原片段及 1–5 个副本直接转发 |

共同参数：10 MHz 采样、300 μs PRI、100 μs LFM 脉宽、5 MHz 雷达带宽；每条 3,000 点；无真实目标；分量合成后只加入一次单位功率复高斯接收机噪声。完整物理参数在 `signals.py::DEFAULT_PHYSICS`；若要调整，可在生成 JSON 中添加 `physics`，只填写需要覆盖的参数。

[生成 JSON](generation/joint.json) 保留计数来源、种子、功率差、标称 JNR、弱定义和输出位置六项。标签计数继续采用 v3 的 IF=3 分布，训练 8,000 条，开发 6,048 条；不额外生成标签均衡组。

未衰减分量 JNR=30 dB，每个多分量样本轮换一个活动类衰减 0/6/12/18/24 dB。**弱分量定义为实测干净分量功率比该样本最强分量低至少 10 dB**；单分量和等功率样本不标弱。各功率视图共享同一组干净分量、非功率参数和噪声，只调整选定分量的幅度。

新文件使用原生 HDF5 `.h5`，避免依赖 MATLAB 写文件；原有 MAT 和新 HDF5 共用同一 dataloader。连续真值按 `[6,N]` 保存，缺席类为 NaN，掩码为 false；包含逐类有效 JNR、实测功率、相对功率、弱标记、请求衰减角色、分量/噪声种子，以及带名称的物理参数。I/Q 为 complex64，功率计算与真值为 float64。旧 v3 仍只用作历史分类参考，原数据及 backup 保留。

```bash
# 生成所有配对功率视图；不需要 MATLAB
uv run python -m longtail.data.generation --execute

# 主实验：每个训练身份取一个功率视图，总训练样本数仍为 8,000
uv run longtail-data prepare-joint \
  --raw-root ../datasets/longtail/joint_ama_v1/raw \
  --output-root ../datasets/longtail/joint_ama_v1/label_if3_mixed

# 必要对照：相同标签长尾分布，仅训练输入改为等功率
uv run longtail-data prepare-joint \
  --raw-root ../datasets/longtail/joint_ama_v1/raw \
  --output-root ../datasets/longtail/joint_ama_v1/label_if3_equal --train-gaps 0
```

完整数据已由本机 Python 生成并校验：原始五档视图共 70,240 条波形，约 1.73 GB；整理后的主实验为 8,000 条训练样本与 6,048 条开发样本。训练集中六类弱正例数为 `[1252,379,692,678,700,686]`；两个条件的开发文件 SHA-256 完全一致。生成完成记录位于数据目录 `raw/generation_complete.json`，代码与验证摘要见 [data/power_truth_audit.json](data/power_truth_audit.json)。

两个条件共用完全相同的混合功率开发文件；角色仍按每组合 calibration/check/development-test = 16/16/64 划分。训练数据每个身份只出现一个视图，全部五档功率参与混合训练。check 的弱 Macro-AUPR 选 checkpoint，calibration 拟合逐类 F1 阈值，development-test 报告结果。

```text
joint_ama_v1/
├── raw/label_if3/gap_{0,6,12,18,24}/
│   ├── train.h5
│   └── development.h5
├── label_if3_mixed/            # 主实验，dataset.json + train/development.h5 + views/
└── label_if3_equal/            # 等功率训练对照，相同开发文件
```

```bash
CUDA_VISIBLE_DEVICES=0,1 uv run torchrun --standalone --nproc_per_node=2 \
  -m longtail.train --config configs/joint.json
uv run longtail-evaluate --checkpoint RUN_DIR/best.pt
# 主任务内按功率分档观察结果，沿用同一组样本身份和冻结阈值
uv run longtail-evaluate --checkpoint RUN_DIR/best.pt --view gap_24
```

对照使用 `configs/joint_equal.json`。后续换模型只改 `model.entrypoint`，当前论文起点保持 CNN；Transformer 文件仅作为接口可扩展的示例，不额外安排模型对比实验。

## 研究目标与实验安排

研究目标收敛为：**在标签长尾与功率不均衡同时存在时，提高六类复合干扰中弱分量，尤其是尾类弱分量的识别能力。** 输出仍是六类存在概率，功率与弱标记用于评测分层，推理只输入 I/Q。

| 安排 | 数据与目的 | 报告内容 |
|---|---|---|
| 主实验 | 标签 IF=3 + 混合分量功率差，使用 CNN+BCE 作为起点，并承接后续方法 | 弱 Macro-AUPR、弱 Recall、尾类弱 Recall，以及总体 Macro-F1 |
| 等功率对照 | 相同标签计数、模型、训练预算和开发样本，仅训练功率改为相等 | 与主实验直接比较，说明训练中引入功率不均衡的作用 |

逐类与功率分档结果放在同一主实验中分析，不再拆成独立实验。弱 AUPR 对每类选择“真实弱正例 + 该类缺席负例”，排除强正例；尾类由训练计数最低的三分之一确定，当前为 AMA、SinFM。固定 0.5 与 calibration 阈值结果均由已有测评代码输出。

## 旧 v3 baseline 与历史成绩的边界

默认 baseline 使用 width=16、dropout=0.2、phase lags `[1,4,16]`、BCE、AdamW、学习率与 weight decay 均为 `1e-4`、全局 batch=32、8 epochs、梯度裁剪=5，从随机初始化训练。

`doc.md` 中约 0.6518 exact 的历史结果还使用了缺失的预训练特征 checkpoint，并在 calibration 上同时选 epoch 和拟合阈值。因此当前默认 baseline 的分数不能冒充该历史复现。若补齐原权重，可使用 [configs/historical_stage_a.json](configs/historical_stage_a.json)：它按已记录 SHA-256 检查权重，只载入 `backbone.*` 特征，保留新分类头，并恢复原 calibration 选点协议。

训练权重和完整运行产物保存在本机 `runs/`，不纳入 Git。历史源码临时备份在 `/tmp/longtail_legacy_before_refactor_20261017.tar.gz`；`doc.md` 保持用户提供的内容。

## 旧 v3 可用权重

已使用两张 A800 完成 seed=2050、8 epochs 的完整 baseline 训练，并在归档旧数据后重新加载 checkpoint 测评。权重位于：

```text
runs/baseline_seed2050_9832439103/best.pt
```

```bash
uv run longtail-evaluate \
  --checkpoint runs/baseline_seed2050_9832439103/best.pt \
  --output runs/baseline_seed2050_9832439103/evaluation/recheck
```

4,032 条 development-test 样本上的本次结果：

| 决策 | Macro-F1 | Exact | Macro-AUPR |
|---|---:|---:|---:|
| 固定 0.5 | 0.872380 | 0.375992 | 0.964568 |
| calibration F1 阈值 | 0.897822 | 0.534970 | 0.964568 |

这是单种子、从头训练的开发基线。完整记录见 [data/baseline_validation.json](data/baseline_validation.json)。原有 15 项测试通过，覆盖预处理、缓存与延迟读取一致性、身份隔离、模型扩展、历史特征载入、配置绑定、断点恢复、两进程 DDP、测评重载以及备份冲突保护；另通过了真实双 GPU 的 BF16 + SyncBatchNorm 检查。原始数据的归档记录见 [data/backup.json](data/backup.json)，36 个原文件均保留。

## 当前联合任务 baseline

已在新 Python 数据上完成两张 A800、seed=2050、8 epochs 的 DFMS-IQ CNN+BCE 训练，耗时约 139 秒。权重、完整 JSON、划分和冻结阈值位于：

```text
runs/joint_dfms_seed2050_78d6f2949c/best.pt
```

4,032 条 development-test 样本、2,088 个真实弱正例上的结果：

| 指标 | 结果 |
|---|---:|
| 总体 Macro-F1（calibration 阈值） | 0.895560 |
| 弱分量 Macro-AUPR | 0.303164 |
| 弱分量 Macro-Recall | 0.140322 |
| 尾类弱分量 Recall | 0.300885 |

结果文件为 `runs/joint_dfms_seed2050_78d6f2949c/evaluation/development_test/metrics.json`。这是当前任务的起点，后续方法可直接复用本数据和评测协议。

34 项测试通过，覆盖六类信号的相位与转发结构、逐分量功率、共享噪声配对、HDF5 生成与读取、CNN/Transformer DDP、checkpoint 重载及冻结阈值测评。完整生成与 baseline 记录统一保存在 [data/power_truth_audit.json](data/power_truth_audit.json)。

## 类别查询路线的 CPU 验证（2026-11-02）

已实现 [DFMSQueryIQ](src/longtail/models/dfms_query.py)：保留 DFMS 全局统计分类分支，六个类别查询通过交叉注意力读取复数特征的实部与虚部，再以可学习残差系数补充分类 logits。最深层查询读取第4层，多层版本读取第2/3/4层，输入仍仅为 I/Q。小型一维解码器借鉴 Query2Label 思路，不照搬其图像骨干和大模型参数。

本轮四组各运行两个种子（2050/2051），每组8轮、batch32、BCE、FP32；每作业绑定4个物理核，8作业并行，共用32核。所有组重新进行CPU训练，采用相同数据身份、check弱AUPR选模和calibration阈值协议。全套44项测试通过。主训练和七项冻结测评实际耗时21.0分钟，单作业训练耗时约16.1–18.8分钟；FPR补充测评和记忆删除诊断另计。

| 模型 | 参数量 | 六标签全部正确率 Exact | 弱 Macro-AUPR | Macro-F1 |
|---|---:|---:|---:|---:|
| DFMS基线 | 257,511 | 45.08% | 0.3055 | 89.57% |
| 最深层查询 | 300,211 | **46.45%** | **0.3159** | **89.97%** |
| 最深层查询容量对照 | 310,531 | 46.09% | 0.3139 | 89.84% |
| 第2/3/4层查询 | 310,707 | 45.67% | 0.3024 | 89.73% |

以上为两个种子的均值。最深层查询的Exact较基线提高1.38个百分点，但默认F1阈值下弱宏召回从15.30%降至14.32%，误报也减少；增益不能解释为全部弱分量更易检出。校准FPR目标5%时，最深层查询弱召回21.78%，基线20.09%，实际开发集FPR分别约5.36%与5.23%。多层查询低于单层和容量对照，本轮未验证额外多层证据的收益。配对24 dB视图中最深层查询Exact为14.09%，基线为15.71%，极端功率差仍是明显短板。

本轮是已消费开发数据上的双种子初验，没有新锁定测试。下一步候选为最深层查询，仍需五种子与新身份确认。完整逐种子、逐类、功率分档、误报约束、注意力与删除诊断见 [HTML报告](reports/query_route1_cpu_20261102/report.html)，机器可读结果见 [results.json](reports/query_route1_cpu_20261102/results.json)。权重、冻结配置、划分、源代码快照和全部预测保存在 `runs/query_route1_cpu_20261102/`。

补充的冻结模型诊断见 [性能瓶颈分析](reports/query_route1_bottlenecks_20261102/report.html)。本次仅在check的1,008个身份和固定fit子集上推理，没有更新权重或部署阈值。最深层查询约77%的错误样本仅包含弱正例漏检；多层模型逐头注意力归一化熵约0.998，类别注意力高度相似。弱正例虽只占诊断子集约8.7%的标签判断，却贡献约49%–59%的BCE损失，因此不能简单归结为弱正例缺少误差信号。下一轮优先验证查询读取已有幅相融合后时序特征的效果，保留相同参数量及均匀汇聚对照；更改结构的收益仍需独立训练确认。

## 24轮证据来源与读取方式对照（2026-11-02完成）

已于北京时间19:46启动 `runs/query_evidence24_cpu_20261102/`：原DFMS、最深层查询、融合后时序特征查询、均匀汇聚容量对照，分别运行种子2050/2051。三种新增分类头的模型总参数均为300,211；均匀对照使用两个实际参与训练的MLP块，没有保留不使用的Q/K参数。融合后查询与原最深层查询具有完全相同的初始state_dict，只改变查询读取的证据来源。

八个作业各绑定4个不同物理核，总共32核；从头训练24轮，统一采用BCE、AdamW、初始学习率1e-4及24轮cosine日程。该日程在启动前固定，因此旧8轮常数学习率成绩不作为本轮训练预算对照。每轮check弱AUPR选模，calibration拟合逐类F1阈值及1%/5%误报目标阈值；最终自动评估check和已消费开发报告集的混合视图与五档配对功率视图。独立测试仍需后续新身份确认。

本轮于北京时间20:47:12全部完成，训练、测评及自动汇总实际耗时61.2分钟，处于初始55–75分钟预估内。8/8作业、192/192训练轮与补充测评均完成，无失败；核验104份主测评产物、68份冻结源文件及8个权重。最终状态见 `pipeline_completed.json`，各作业日志位于 `logs/`。进度文件已更新为全部完成，保留了最后一次训练轮询的原始记录。

4,032个身份的开发报告集双种子均值如下（该报告集已消费，不是独立测试）：

| 模型 | Exact | 弱Macro-AUPR | 弱Macro-Recall | 平均FPR |
|---|---:|---:|---:|---:|
| DFMS基线 | 49.49% | 0.3737 | 19.40% | 2.67% |
| 最深层查询 | 51.08% | 0.3835 | 21.87% | 3.04% |
| 融合后特征查询 | 50.27% | 0.3853 | 17.52% | 1.75% |
| 均匀汇聚容量对照 | 49.76% | 0.3713 | 19.54% | 2.66% |

最深层查询相对基线及同容量均匀汇聚对照，两个种子的Exact增益均为正。融合后查询在check上最好，但报告集相对最深层的Exact低0.81个百分点，默认阈值下弱召回更低、误报更少；24 dB视图的Exact为14.35%，最深层为16.51%。其收益不能解释为极弱分量问题已解决。保留两种查询候选，下一步优先固定协议后进行新身份与更多种子确认。本轮没有追加调参或训练。

完整训练曲线、功率分档、误报工作点及配对统计见 [本轮结果分析](reports/query_evidence24_cpu_20261102/analysis.html)，逐次数字见 [results.csv](reports/query_evidence24_cpu_20261102/results.csv)。

新模型位于 [dfms_query_variants.py](src/longtail/models/dfms_query_variants.py)，配置为 `configs/query_evidence24_{cpu,single,fused,uniform}.json`，控制器为 [run_query_evidence24_cpu.py](experiments/run_query_evidence24_cpu.py)。51项测试通过，明确冻结构造参数后重新运行的新模型7项测试亦通过。

## v3新身份长尾基线归因（2026-11-02启动，11-03完成）

已建立ordinary BCE、label-aware sampler、combo-aware sampler、weighted BCE和训练期Logit Adjustment五组，分别运行2050–2054五种子。逐标签F1校准是每个模型的测评条件，因此是25个训练模型、固定0.5/校准两组决策。所有模型使用相同DFMS-IQ、24轮cosine、每轮8000次抽样和250步，按check Macro-AUPR选模；calibration精确搜索逐标签F1阈值，报告身份等全部训练完成后才统一测评。源代码和配置在启动前冻结。

新池为 `../datasets/longtail/v3_fresh_identity_20261102/`，包含8000训练身份和6048开发身份，检查52份旧数据文件无身份交集。保留v3标签计数、物理公式及逐分量独立10–30 dB JNR；本机没有可用MATLAB/Octave，执行采用已有Python物理移植与PCG64，**不声称MATLAB随机流逐点复现**。原MATLAB源码哈希和Python生成记录均保留。该池与固定30 dB后衰减的joint任务不同。

label-aware定义为均匀标签混合采样，仍受共现影响：FMB期望曝光约84.14%、AMA38.69%；combo-aware对63组合均匀采样，六类期望曝光均约50.79%。有放回采样每轮仍为8000次，额外记录独立身份覆盖和真实标签/组合曝光。weighted BCE采用fit计数的 `pos_weight=(N-n_pos)/n_pos`；LA使用 `BCE(z+logit(pi),y)`、tau=1，推理使用raw z，不二次调整。采样和损失不叠加。

59项测试及五方法真实I/Q批次检查通过。采用8个作业各4个物理核，25次训练于北京时间11月2日21:39:52启动，11月3日00:55:41全部训练和报告测评完成，耗时3小时15分49秒。运行目录为 `runs/longtail_v3_controls_20261102_exact/`，完成标记为 `pipeline_completed.json`。25个模型均完成24轮、6000步；复核80份冻结文件、数据哈希和100组测评条件通过，报告测评发生在全部训练完成之后。

新身份报告池上的五种子均值如下（百分数）：

| 方法 | 固定0.5 Exact | F1校准 Exact | F1校准 Macro-F1 | Macro-AUPR |
| --- | ---: | ---: | ---: | ---: |
| BCE | 57.698 | 63.373 | 92.305 | 97.759 |
| 标签采样 | 56.776 | 62.168 | 92.108 | 97.603 |
| 组合采样 | 65.258 | 63.800 | 92.543 | 97.918 |
| weighted BCE | 57.277 | 56.627 | 90.656 | 96.873 |
| Logit Adjustment | 63.785 | 63.497 | 92.309 | 97.771 |

BCE校准带来稳定的Exact +5.675个百分点，主要缓解FMB误报。LA在同样校准后与BCE基本持平；当前weighted BCE造成FMB排序退化。组合采样+0.5的整体Exact最高，相对BCE+校准增加1.885个百分点，但尾类F1下降1.643个百分点；双方都校准后整体Exact差仅+0.427个百分点，五种子描述性区间跨零，稀缺组合Exact仍改善。逐类F1优化与六标签Exact目标不一致，组合采样校准后平均误报增加，Exact反而下降。

复核还发现有放回采样采用 `seed+epoch`，相邻种子会在偏移轮复用抽样流：每种采样的120轮仅有28种组合曝光记录。模型初始化种子仍不同，但不能把五种子区间视为五套独立采样流的严格显著性证据。下一轮应修正种子派生、预定义Exact与误报/召回目标，并使用新的报告身份确认。当前报告池已经消费，不再用于调参。

完整逐类权衡、采样覆盖、阈值目标和可信度边界见 [结果分析](reports/longtail_v3_controls_20261102/analysis.html)，原始 [CSV](reports/longtail_v3_controls_20261102/results.csv) 与 [完成审计](reports/longtail_v3_controls_20261102/completion_audit.json) 可复核。冻结方案和原时间估计见 [方案与可行性](reports/longtail_v3_controls_20261102/plan.html)。只复核结果可运行 `experiments/analyze_longtail_v3_controls.py --suite runs/longtail_v3_controls_20261102_exact`。

此前查询路线的复现命令（与上述v3长尾对照是两套实验；CPU编号对应本机，换机器须先调整 `SLOTS`）：

```bash
.venv/bin/python experiments/validate_query_route1_cpu.py --suite runs/query_route1_cpu_repeat --prepare
.venv/bin/python experiments/validate_query_route1_cpu.py --suite runs/query_route1_cpu_repeat --benchmark
.venv/bin/python experiments/validate_query_route1_cpu.py --suite runs/query_route1_cpu_repeat --run
.venv/bin/python experiments/evaluate_query_route1_fpr.py --suite runs/query_route1_cpu_repeat
.venv/bin/python experiments/probe_query_route1_memory.py --suite runs/query_route1_cpu_repeat
```

报告汇总脚本为 `experiments/summarize_query_route1.py --suite RUN_ROOT --output REPORT_DIR`，使用含 NumPy、Matplotlib 的Python环境运行。主训练依赖不需增加Matplotlib。按照相同硬件并发和负载重跑，建议预留25–35分钟，实际取决于机器竞争和数据缓存状态。

## CPU优化验证（2026-11-03）

北京时间11:40:09启动、13:31:02全部完成 `runs/combo_optimization_cpu_20261103/`：自然采样、纯组合采样、50/50混合采样各五种子，共15个模型、360轮、90,000优化器步。统一DFMS-IQ、普通BCE、24轮cosine、每轮8000次抽样；8个作业各4个物理CPU核并发，实际耗时1小时50分53秒。

采样随机流改为对 `(seed, epoch, stream)` 分域哈希派生，避免原来的偏移轮复用；50/50混合采用4000自然身份无放回抽样、4000次组合有放回抽样，合并后打乱。67项测试和三方法真实I/Q优化器步检查已通过。代码、配置和决策规则在启动前冻结到运行目录的 `source_snapshot/`。

保留此前8000训练、1008 calibration、1008 check身份及波形；使用 `../datasets/longtail/v3_optimization_report_20261103/` 的4032个新报告身份，检查54份既有数据文件无交集。数据仍采用原v3物理公式、逐分量独立10–30 dB JNR的Python PCG64移植，非MATLAB逐点复现。

每个模型比较固定0.5、逐类F1和受约束Exact校准：Exact只使用calibration，逐类召回相对F1最多下降3个百分点、误报最多上升2个百分点，固定网格和双起点坐标搜索，平局偏好0.5。checkpoint仍按check Macro-AUPR选取。另按预定规则验证同种子融合：FMB使用纯组合模型0.5决策，其他五类使用BCE模型F1决策。全部15个模型及三种决策的哈希冻结后，才统一测评新报告身份。

新报告身份上的五种子均值如下（百分数）：

| 方案 | Exact | Macro-F1 | Macro-AUPR | 尾类F1 |
| --- | ---: | ---: | ---: | ---: |
| 自然BCE＋F1 | 62.996 | 92.230 | 97.741 | 97.053 |
| 纯组合＋0.5 | 65.218 | 92.575 | 97.908 | 95.929 |
| 混合＋0.5 | 66.190 | 92.855 | 98.148 | 96.288 |
| 混合＋F1 | 66.156 | 93.085 | 98.148 | 96.742 |
| 混合＋Exact | 66.215 | 93.072 | 98.148 | 96.707 |
| 预定义双模型按类融合 | 68.423 | 93.577 | 98.266 | 97.053 |

融合相对BCE＋F1的Exact增加5.427个百分点，条件配对95%区间[4.875, 5.979]；相对纯组合＋0.5增加3.204个百分点，五种子均提高。融合保留其他五类的BCE决策，同时改善FMB召回和误报，代价是两模型推理。混合采样的Macro-AUPR相对纯组合提高0.240个百分点，五种子均为正，尾类与MFT/ISRJ表现有所恢复；固定0.5的Exact差值区间跨零，不能宣称所有指标稳定更优。

Exact校准相对F1只增加0.060–0.283个百分点Exact，各方法条件区间均跨零。经验约束在calibration均满足，但新报告中BCE种子2054的ISRJ召回下降3.662个百分点，超过拟合阶段3个百分点约束，不能作为新数据保证。每种采样的120轮抽样顺序和派生种子均不同，原随机流复用问题已修复。

完整分析见 [新一轮结果分析](reports/combo_optimization_cpu_20261103/analysis.html)，原始 [汇总](reports/combo_optimization_cpu_20261103/report.html)、[逐种子CSV](reports/combo_optimization_cpu_20261103/results.csv)、[完成审计](reports/combo_optimization_cpu_20261103/completion_audit.json) 以及 [冻结方案](reports/combo_optimization_cpu_20261103/plan.html) 可复核。运行目录保留 `progress.json`、`pipeline.json`、`controller.log` 和最终 `pipeline_completed.json`。本轮新报告已经消费，后续调参需用新的报告身份确认；模拟结果不代表实测或MATLAB逐点复现。

## 弱分量识别验证（2026-11-03完成，11-04复核）

北京时间16:52:35启动 `runs/weak_component_validation_cpu_20261103/`，验证数据多样性、弱分量难例和弱正例损失能否进一步提高六标签全部正确率。固定复用上一轮同种子的组合采样FMB专家及0.5阈值，其余五类使用下表的BCE分支。种子为2050–2054，新增B/C/D三组共15个模型，A基线与FMB共10个模型复用冻结权重。

| 组 | 训练身份池 | 训练变化 |
| --- | ---: | --- |
| A | 8000 | 复用已有自然BCE，逐类F1校准 |
| B | 32000 | 保留原8000条波形，按原组合计数增加24000个身份；自然独立10–30 dB JNR |
| C | 32000 | 与B共用标签、物理参数和噪声；13718条配对难例的目标SawFM/MFT/ISRJ分量相对最强分量≤−10 dB |
| D | 与C完全相同 | 仅将SawFM/MFT/ISRJ弱正例损失乘1.5，并按fit的逐类平均权重归一化 |

B/C/D均为DFMS-IQ、普通自然采样、AdamW、24轮cosine；每轮从32000个身份中无放回抽取8000条，共6000优化器步，与A预算相同。B/C/D同种子使用相同身份抽样序列。8个作业各绑定4个物理CPU核，共32核，原预计训练与自动测评预留2–3小时，实际于北京时间18:37:45全部完成，耗时1小时45分10秒。

新数据位于 `../datasets/longtail/v3_weak_validation_20261103/`，原8000训练波形逐项保留，calibration/check各1008身份及波形保持一致；另生成4032个报告身份，核对56份旧数据文件无身份交集。新报告仍采用原独立10–30 dB分布，条件弱难例只用于训练。B/C弱正例占比分别为22.73%/33.29%。物理生成采用Python PCG64移植，不声称MATLAB逐点复现，执行时的生成源码另存并核验哈希。

checkpoint按check Macro-AUPR选取。calibration拟合逐标签F1阈值，并另拟合匹配A逐类经验误报预算的辅助阈值，FMB在所有组中固定为同一专家0.5决策。仅使用check从B/C/D选择一个主候选；全部15次训练完成、25个模型权重及决策和候选选择冻结后，才开始新报告推理。主验收要求同池Exact平均提高至少1个百分点、至少4/5种子正向、配对种子95%区间下限为正、尾类F1下降不超过0.3个百分点、最大逐类平均FPR上升不超过1个百分点。若check没有符合条件的候选，则本轮没有主候选；不根据新报告改选。

75项测试、三组真实I/Q优化器步以及合成数据240组自动汇总/候选选择/哈希防篡改检查均通过。启动与最终进度分别见 [launch.json](runs/weak_component_validation_cpu_20261103/launch.json)、[progress.json](runs/weak_component_validation_cpu_20261103/progress.json)，方案见 [冻结方案](reports/weak_component_validation_cpu_20261103/plan.html)。控制器使用 `source_snapshot/` 中的冻结源码运行，最终已写入 `pipeline_completed.json` 及 `reports/weak_component_validation_cpu_20261103/report.html`、`results.json`、`results.csv`。训练数据的功率真值仅用于损失和测评，推理输入仍为I/Q信号。

11月4日复算240组单模型/融合测评条件、25份校准决策、360轮采样与15个模型的6000步，核验93份冻结源码以及数据哈希和56份旧文件的报告身份无交集。模型、阈值与check候选于18:35:43冻结，首个新报告作业18:35:44启动，隔离顺序正确。各种子24轮覆盖31959–31970个训练身份；跨五种子和24轮为120个不同抽样流，同种子的B/C/D共用对应流。

同一4032个新报告身份上的五种子融合均值如下（百分数，逐类F1主决策）：

| 组 | Exact | Macro-F1 | Macro-AUPR | 尾类F1 | SawFM/MFT/ISRJ平均弱召回 | 平均FPR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A 原融合 | 68.953 | 93.756 | 98.394 | 97.213 | 56.818 | 4.913 |
| B 自然32k | 71.205 | 94.301 | 98.620 | 97.591 | 61.062 | 4.587 |
| C 弱难例32k | 70.149 | 94.016 | 98.500 | 97.468 | 58.348 | 4.320 |
| D 弱难例＋加权 | 69.717 | 93.899 | 98.414 | 97.560 | 57.966 | 4.372 |

B−A的Exact提高2.252个百分点，条件配对95%区间[0.821, 3.683]，五种子均正向；但SawFM FPR从6.915%升至8.488%，check阶段也超过预设逐类误报增量1个百分点，因此B未被选为正式主候选。check唯一合格候选为C；新报告C−A的Exact提高1.195个百分点，4/5种子正向，但条件95%区间[−0.565, 2.956]跨零，**本轮主验收未通过**。不能在看过新报告后改选B为本轮通过方案。

收益主要来自自然数据多样性：C−B的Exact下降1.057个百分点，目标弱召回下降2.714个百分点，Macro-AUPR下降0.119个百分点；当前难例比例不支持进一步提升。D−C的Exact下降0.432个百分点，Macro-AUPR五种子均下降，额外1.5倍弱正例加权没有显示收益。训练弱正例比例提高并不自动转化为自然分布上的弱分量识别改善。

预定义辅助决策“B＋calibration经验误报预算”的Exact为70.813%，相对同辅助决策的A（68.859%）提高1.954个百分点，条件95%区间[0.938, 2.971]，五种子均正向；六类平均FPR均未增加，SawFM降至6.169%。相对A主决策提高1.860个百分点，但B的SawFM弱召回从其F1决策的57.143%降至49.814%。该辅助方案是下一轮优先确认候选，仍属于探索证据，不改变C未通过主验收的结论。之后优先单独验证B的6000/12000步预算，难例增强则先降低比例；本轮没有证明这些后续方案有效。

完整分析与图表见 [结果检查与分析](reports/weak_component_validation_cpu_20261103/analysis.html)，[完成审计](reports/weak_component_validation_cpu_20261103/completion_audit.json) 与 [逐类分析JSON](reports/weak_component_validation_cpu_20261103/analysis.json) 可复核。当前新报告已经消费，下一轮应固定候选及验收规则后使用新身份确认；calibration/check多轮复用及模拟到实测的泛化仍是限制。

## 决策、冻结分类头与one-bit局部头顺序验证（2026-11-04）

北京时间14:02:44启动 `runs/staged_boundary_cpu_20261104/`。复用自然32k BCE模型及同种子的FMB组合专家，按下列顺序自动执行；后阶段仅采用前阶段check选择，报告分数不参与后续选型。每阶段分别生成新的1008 calibration、1008 check和4032 report身份，核对62份旧文件无背景噪声身份交集。自然32k训练池及原模型保持固定。

| 阶段 | 对照 | 新增训练 |
| --- | --- | ---: |
| 1 冻结logits决策 | F1、误报预算、误报约束Exact、弱召回约束Exact | 0 |
| 2 冻结128维特征分类头 | 原头、自然采样重训、50/50温和组合采样重训 | 2组×5种子 |
| 3 one-bit局部头 | 原模型、自然局部头、配对BCE局部头、配对margin局部头 | 门控通过后3组×5种子 |

每个新增模型均为24轮、每轮250步、batch32，共6000步/192000个视图曝光，AdamW lr=0.001、weight decay=0.0001及cosine日程。分类头仅645个可训练参数；局部头为128→32→3的4227参数零初始化残差，仅作用SawFM/MFT/ISRJ。特征提取时原backbone为eval且冻结；FMB始终由原组合专家0.5决策。五种子并行，各推理作业绑定4物理核；缓存上的小分类头训练使用单线程。

严格one-bit数据区分唯一sample_id、共享pair_id和receiver_noise_seed；训练/检查/报告分别为2432/768/1536对，各侧共享背景和噪声，仅增删一个成分，原始I/Q差分等于目标分量。四个诊断目标各有31种非空背景边，按19/6/6种边隔离；FMB配对仅作诊断。该留出针对配对训练边，原自然backbone已经见过63种标签组合，不能称完全未见组合泛化。配对差分与干净成分只用于生成审计，不作为模型或可观测性探针输入。

第三阶段先对单条信号的物理特征Logistic探针及冻结模型作配对边留出检查；目标的二者最佳check AUROC至少0.60才启用相应残差，均未通过则跳过局部训练。未通过不等于证明信号在信息论上不可观测。配对训练维持75%自然视图，batch为24自然视图+4对；margin对照仅额外加入0.1×relu(0.5−目标正负logit差)。

各阶段只用calibration拟合决策、check选唯一候选，全部权重和阈值及候选哈希冻结后才测评本阶段report。约束不可行时显式记录，后阶段采用已预定的误报预算回退。正式验收要求同池Exact增益至少0.5个百分点、至少4/5种子正向、条件配对95%区间下限为正，并满足尾类F1、逐类FPR及SawFM弱召回门槛。更改决策与重训头的收益分别汇报，不跨不同报告池直接相加。

84项测试、真实I/Q分类头与三种局部头优化器步、三阶段合成数据选优/冻结/汇总流程检查均通过，101份训练源码及配置冻结。数据位于 `../datasets/longtail/v3_staged_validation_20261104/`，运行入口为 [run_staged_boundary_cpu.py](experiments/run_staged_boundary_cpu.py)。[冻结方案](reports/staged_boundary_cpu_20261104/plan.html)、[最终进度](runs/staged_boundary_cpu_20261104/progress.json) 与 `controller.log` 保留执行状态；已保存三阶段HTML/JSON结果及 `pipeline_completed.json`。

全部流程于北京时间14:15:19完成，耗时12分35秒。25个新增模型完成600轮、150000个实际优化器步；核验15个复用权重、101份冻结源文件、254组测评条件及50份calibration决策，身份隔离与报告冻结顺序均通过。三个目标的单条信号可观测性检查均通过，但不等于证明极弱分量已经解决。

| 阶段 | check预选方案 | 同池参考Exact | 候选Exact | 差值 pp | 正式验收 |
| --- | --- | ---: | ---: | ---: | --- |
| 1 | 逐类F1 | 71.503% | 71.855% | +0.352 | 未通过 |
| 2 | 自然重训线性头 | 71.558% | 71.667% | +0.109 | 未通过 |
| 3 | one-bit配对BCE局部头 | 72.133% | 73.462% | +1.329 | **通过** |

各阶段使用不同的新报告池，增益不能跨池相加。第三阶段预选候选相对本阶段线性头基线的Exact条件配对95%区间为[+0.938,+1.720] pp，五种子均提高；Macro-F1 94.643%、Macro-AUPR 98.761%，平均FPR从3.789%降至3.434%。相对同池未经分类头重训的原B融合（71.597%），完整候选提高1.865 pp，区间[+1.348,+2.382] pp。

收益归因：自然非线性局部头已经提高Exact 0.957 pp，区间[+0.487,+1.428]；加入one-bit配对BCE再提高0.372 pp，五种子方向一致，但区间[−0.047,+0.791]跨零，不能把全部收益归因于配对数据。额外margin相对配对BCE为−0.109 pp，没有显示额外Exact收益。最终候选SawFM弱召回均值从50.560%升到54.784%，但该增益及目标三类平均弱召回的区间均跨零；目前支持整体边界改善，尚不能宣称弱分量召回稳定全面提高。

完整图表与归因见 [三阶段结果分析](reports/staged_boundary_cpu_20261104/analysis.html)，权重与逐种子阈值见 [候选模型清单](reports/staged_boundary_cpu_20261104/selected_models.json)，[完成审计](reports/staged_boundary_cpu_20261104/completion_audit.json) 可复核。五种子为五个分别测评的模型，不是五模型预测集成。复核入口为 `experiments/analyze_staged_boundary_cpu.py --suite runs/staged_boundary_cpu_20261104`；报告已消费，后续调参需另用新身份确认。

## 配对复现、时序局部头及新身份重训（2026-11-04）

北京时间14:42:26启动 `runs/replication_temporal_cpu_20261104/`，CPU五种子并行、每作业绑定4物理核。完整流水线从冻结源码快照运行，按以下顺序自动训练、选型、冻结、测评和审计：

| 阶段 | 目的及对照 | 新增模型数 |
| --- | --- | ---: |
| 配对复现 | 两份独立配对身份池，各比较普通global32局部头与25%配对视图BCE局部头 | 20 |
| 时序读取 | global32（4227参数）、参数量相近global33（4359）、时序标签注意力（4323）；采样策略由上一阶段check决定 | 15 |
| 新身份重训 | 新32k自然池随机初始化重训B backbone，再训练线性头及普通/胜出局部头；固定旧FMB专家 | 15～20 |

每个模型均24轮×250步、batch32，共6000步/192000视图；全轮50～55模型、300000～330000实际优化器步。backbone学习率0.0001、头学习率0.001，均AdamW、weight decay0.0001及24轮cosine。90项全套测试及真实I/Q三种局部头前向、反向检查通过；另补充了校准决策在独立报告测评中保持不变的集成测试。预池化张量为128×375，缓存为float32；提取路径与原模型全局128维输出逐元素一致。注意力头读取已有上下文时间位置，不含新增位置编码。

四份独立自然development池分别提供1008 calibration、1008 check、4032 report；两份配对复现池及新重训配对池各含2432/768/1536训练/检查/报告对。配对复现保持原19/6/6背景边划分，只重生成身份；普通训练轨迹在相同模型种子下共享，分别在新check选最佳轮次。统计时先在同一模型种子内平均两份数据，不能把相同backbone当成10个独立种子。

后续训练只采用上阶段check选择，report不影响后续选型。正式成功仍要求同池Exact至少提高0.5pp、4/5种子正向、条件95%区间下限为正，并限制尾类F1下降0.3pp、各类FPR上升1pp，以及SawFM和ISRJ弱召回下降3pp。辅助决策只匹配calibration误报预算，报告上的实际FPR另行测量。第三阶段还测评独立0～20dB及10～40dB分量JNR池，沿用主校准阈值；这些功率偏移检查不能替代真实数据验证。

数据见 `../datasets/longtail/v3_replication_temporal_20261104/`，所有新增背景身份与历史文件核对无重叠。[冻结方案](reports/replication_temporal_cpu_20261104/plan.html)、[结果页面](reports/replication_temporal_cpu_20261104/analysis.html)、[实时进度](runs/replication_temporal_cpu_20261104/progress.json) 和 `coordinator.log` 保留状态。完成后自动生成阶段结果、候选清单、`completion_audit.json` 与 `pipeline_completed.json`；以这些文件和报告门槛判定结果，启动不表示已经通过。

如执行被外部中断，可使用原冻结快照继续同一流水线：

```bash
PYTHONPATH=runs/replication_temporal_cpu_20261104/source_snapshot/src CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 .venv/bin/python -u runs/replication_temporal_cpu_20261104/source_snapshot/experiments/run_replication_temporal_cpu.py --suite runs/replication_temporal_cpu_20261104
```

已有完成作业跳过；训练中的头和backbone均可从末轮检查点继续。协调器用文件锁防止同时重复启动。固定旧FMB分支、相同Python物理仿真器及次要比较未校正多重检验，仍属于本轮结论的适用限制。

全流程于北京时间16:07:07完成，耗时84分41秒；实际50个新增模型、1200轮、300000优化器步，无失败。710组主/辅助指标及20份校准决策重放通过，预测标签、身份、功率真值及报告冻结顺序均通过审计。完成后新增的统计复核脚本再次检查冻结权重、源码和数据哈希，不修改训练、阈值或check选择。

| 验证 | 同池结果 | 结论 |
| --- | --- | --- |
| repA配对BCE−普通局部头 | Exact +0.293pp，条件95%区间[−0.267,+0.852] | 配对额外收益未确认 |
| repB配对BCE−普通局部头 | Exact +0.010pp，区间[−0.604,+0.623] | 配对额外收益未确认 |
| 时序注意力−普通global32 | Exact −0.615pp，区间[−0.824,−0.406]，五种子均下降 | 当前时序读出方案不采用 |
| 新身份重训的普通局部头−自身线性基线 | Exact +1.349pp，区间[+0.910,+1.789]，五种子均提高 | 通过预设门槛 |

两份配对复现先在同一模型种子内平均后，额外收益只有+0.151pp，区间[−0.351,+0.653]。check阶段已选择普通采样：repA配对Exact下降；repB的SawFM弱召回下降5.283pp，MFT FPR增加1.008pp。报告不用于改选。配对视图本身没有提高弱分量密度：自然池弱正例22.73%，两份配对池的分量弱正例约21.52%和21.73%；保证单成分增删不等于已构造足够多的弱分量边界样本。

最终同一4032身份报告上，新B原分类器融合Exact73.036%，重训线性头73.433%，普通局部头74.782%，上一轮固定权重候选按本轮相同校准规则测评为74.296%。新局部头Macro-F1为94.874%、Macro-AUPR98.864%、平均FPR3.039%。相对旧候选Exact仅+0.486pp，区间[−0.678,+1.650]，4/5种子正向，**尚未证明稳定超过旧候选**。不能与上一轮不同报告池的73.462%直接相减。

本轮局部头的主要收益为减少误报和改善ISRJ决策：相对自己的线性基线，SawFM FPR4.839%→3.367%，但弱召回49.868%→47.895%；MFT弱召回68.319%→67.672%；ISRJ弱召回59.062%→62.388%。三个目标的弱AP均提高约2.5pp，排序改善未全部转化为逐类F1决策下的弱召回提升。校准误报预算辅助规则并不保证报告池FPR相同，不能据此宣称严格等误报下的召回优势。

沿用原阈值时，新方案在0～20dB低JNR报告上的Exact66.285%，10～40dB宽功率报告53.408%；旧候选分别65.759%和53.264%。原分布中15～20dB功率差的分量召回46.220%；宽功率报告中20～30dB差距的1369个正分量召回仅9.890%。原训练分布不包含20dB以上功率差，这一外推缺口尚未解决。当前时序头的失败只适用于本次无新增位置编码的标签注意力读出，不能据此否定所有时序信息。

详细复核见 [完成统计与分析](reports/replication_temporal_cpu_20261104/review.html)、[统计JSON](reports/replication_temporal_cpu_20261104/review.json) 和 [完成审计](reports/replication_temporal_cpu_20261104/completion_audit.json)。[汇总图](reports/replication_temporal_cpu_20261104/outcome_overview.png) 包含条件区间、弱召回、功率偏移和功率差分层结果。复核入口：`experiments/analyze_replication_temporal_cpu.py --suite runs/replication_temporal_cpu_20261104`。保留普通非线性局部头作为结构基线，并保留旧候选作参照；下一轮先固定logits检查SawFM/MFT的校准与功率分层边界，再决定是否采用小比例定向配对；不继续复用已消费的本轮报告身份调参。

## 固定候选的大校准池与小比例定向配对（2026-11-04）

北京时间19:29:28启动 `runs/calibration_targeted_cpu_20261104/`，CPU五作业并行，各绑定4物理核。新旧参考候选的B、线性头、原局部头和FMB权重均以SHA绑定并保持不变；第二阶段只训练新局部头副本。

第一阶段不训练权重，每种子比较新旧固定候选上的七种决策：原固定阈值、1008条小校准池F1、8064条大校准池F1、3%/5%/7%校准误报预算，以及弱召回/FPR约束下的双标签联合Exact搜索。只允许SawFM/MFT阈值改变，其他四类阈值固定。大校准池可拆为8份互不重叠、各组合16条的小池；只预先指定block0为1008条控制，不挑选得分最高小池。check选择一个新旧候选共用的决策规则，报告不参与后阶段选型。

两阶段分别生成8064 calibration、2016 check、8064 report，阶段间身份互斥；另生成4032条10～40dB功率偏移报告。核对91份历史身份文件无交集。主报告均保持原独立10～30dB分量JNR分布。

第二阶段固定原B、线性头和四个非目标标签输出，从新候选局部头隐藏层及SawFM/MFT输出权重初始化4194参数的副本。训练后只替换SawFM/MFT对应残差；FMB/AMA/SinFM/ISRJ输出沿用原参考候选。三个对照均同初始化、相同自然池和预算：

| 方法 | 每batch32视图组成 | 配对视图比例 | 新模型 |
| --- | --- | ---: | ---: |
| natural继续训练 | 32自然视图 | 0% | 5 |
| pair1 | 30自然视图＋1对 | 6.25% | 5 |
| pair2 | 28自然视图＋2对 | 12.5% | 5 |

每模型24轮×250步，共6000步/192000视图；全轮15模型、90000实际优化器步。普通两目标BCE、AdamW lr0.0003、weight decay0.0001、cosine24轮；相同check SawFM/MFT平均弱AUPR选择最佳轮次。天然0%继续训练控制用于区分额外训练预算与定向配对收益。

配对数据只针对SawFM/MFT，按6～10、10～15、15～20dB功率差分层；在原独立10～30dB生成流程上筛选身份，不额外缩放分量。2736个fit对、576个check对、1152个report对，各目标19/6/6背景边隔离；训练时六个目标×功率差分层在24轮曝光中均衡。共享噪声、单成分增删、唯一sample_id和原始差分等于目标成分均验证；模型只接收单条I/Q，功率真值用于fit采样及评测，不进入模型。原训练不覆盖20dB以上功率差，宽功率报告只作外推诊断。

第一阶段只按check选共享规则；第二阶段在冻结新候选、natural、pair1、pair2之间按check Exact及逐类FPR/弱召回约束选候选。共同决策在新校准池若对固定参考不可行，则整个比较显式回退大池F1；单个训练方法不可行则排除，不逐种子静默放宽。正式Exact成功要求同池至少+0.5pp、4/5种子正向、条件95%区间下限为正，并限制两目标各自弱召回下降1pp、逐类FPR上升1pp、尾类F1下降0.3pp。辅助弱召回权衡门槛另行报告，不能冒充整体Exact通过。校准误报预算不保证测试池FPR相等。

98项测试及真实I/Q局部头副本优化器步通过。数据位于 `../datasets/longtail/v3_calibration_targeted_20261104/`。[冻结方案](reports/calibration_targeted_cpu_20261104/plan.html)、[结果页](reports/calibration_targeted_cpu_20261104/analysis.html)、[实时进度](runs/calibration_targeted_cpu_20261104/progress.json) 与 `coordinator.log` 可复核。流水线完成后自动保存阶段结果、候选清单和完成审计；以 `pipeline_status.json`、`pipeline_completed.json` 和报告验收判定状态。

外部中断后可从同一冻结快照继续：

```bash
PYTHONPATH=runs/calibration_targeted_cpu_20261104/source_snapshot/src CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 .venv/bin/python -u runs/calibration_targeted_cpu_20261104/source_snapshot/experiments/run_calibration_targeted_cpu.py --suite runs/calibration_targeted_cpu_20261104
```

已完成作业跳过，头副本从末轮检查点继续，文件锁防止重复协调器。五种子是固定数据池下的条件统计，次要比较未校正多重检验；同一Python物理仿真器、固定FMB和缺少实测验证仍限制结论范围。

两阶段已于北京时间2026-11-04 19:45:19完成，耗时15分51秒。15个新增头副本完成360轮、90000实际优化器步；全部作业成功。1068组指标、10份校准决策、18个独立重生成的原始配对波形均通过审计，原参考权重和其他四类logits及阈值保持不变。完成复核再次检查冻结权重、114份训练源文件及全部数据哈希，并重放85组主/补充报告固定阈值指标；未修改训练、阈值或check选择。

第一阶段check仍选择原固定阈值locked。8064条大池F1相对预先指定1008条小池，新固定候选Exact仅+0.027pp，条件95%区间[−0.511,+0.565]；两目标平均弱召回+4.833pp，同时平均FPR+1.127pp。大校准池改变了误报—漏检权衡，没有提高固定logits的排序能力。3%预算损失弱召回，5%/7%预算及大池F1在check违反逐类FPR保护；旧候选weak_guard在4/5种子不可行，未参与共同选型。没有在看到报告后改选阈值。

第二阶段check预选6.25%配对（pair1）及locked决策。同一8064条独立报告上的五种子结果：

| 方法 | Exact | SawFM弱召回 | MFT弱召回 | 两目标平均FPR | 两目标平均弱AP |
| --- | ---: | ---: | ---: | ---: | ---: |
| 固定新候选 | 74.506% | 47.319% | 68.751% | 3.707% | 78.140% |
| 0%自然继续训练 | 74.608% | 47.653% | 69.465% | 3.674% | 78.520% |
| 6.25%分层配对 | 74.688% | 49.499% | 70.976% | 4.083% | 78.697% |
| 12.5%分层配对 | 74.665% | 50.812% | 71.626% | 4.357% | 78.671% |
| 固定旧候选 | 74.040% | 52.191% | 71.836% | 4.947% | 77.095% |

6.25%相对固定新候选：Exact +0.181pp，区间[−0.035,+0.397]，4/5种子正向，**整体Exact门槛未通过**。SawFM/MFT弱召回分别+2.180/+2.225pp，五种子均提高；两者平均+2.202pp，区间[+1.474,+2.931]。逐类FPR分别增加0.459/0.292pp，按五种子均值**通过预设弱召回权衡门槛**。该门槛允许小幅误报增加，不等于在相同测试误报率下改善。

收益归因：相对相同额外训练预算的0%控制，6.25%的额外Exact收益只有+0.079pp，区间[−0.100,+0.258]；平均弱召回+1.679pp，区间[+0.713,+2.645]；弱AP+0.177pp，区间[+0.087,+0.266]。弱排序有小幅改善，但不能把相对固定权重的全部收益归因于配对，且相对0%控制的SawFM FPR增加0.549pp，未通过较严格辅助门槛。12.5%虽然平均弱召回+3.184pp，但SawFM FPR增加0.781pp，超过0.5pp辅助上限；Exact也未优于6.25%，因此不扩大比例。

大池F1辅助决策下6.25%/12.5%的Exact均约74.983%，属于未被选定的辅助结果，不替代主验收。预选6.25%相对固定旧候选的主决策Exact +0.647pp，区间[−0.007,+1.302]仍跨零，且SawFM弱召回较旧候选低2.692pp，不能宣称全面优于旧候选。

定向配对把目标正侧中至少10dB的弱分量比例提高至2/3，前次未分层配对约22%。但跨轮同时变化了初始化、学习率、训练标签范围和配对筛选，不能把结果只归因于比例降低；本轮0%、6.25%、12.5%对照才用于归因。最难边界仍未突破：原分布15～20dB功率差的两目标召回32.38%→35.25%，宽功率20～30dB差距仅4.63%→5.37%；宽功率Exact53.24%→53.53%。配对边留出Exact56.50%→56.57%，15～20dB差距SawFM/MFT AUROC仅约74.46%→74.60%和78.31%→78.52%。

完整复核见 [完成统计与收益归因](reports/calibration_targeted_cpu_20261104/review.html)、[详细统计](reports/calibration_targeted_cpu_20261104/review.json)、[汇总图](reports/calibration_targeted_cpu_20261104/outcome_overview.png) 与 [完成审计](reports/calibration_targeted_cpu_20261104/completion_audit.json)。复核入口：`experiments/analyze_calibration_targeted_cpu.py --suite runs/calibration_targeted_cpu_20261104`。保留6.25%为弱分量候选，固定其权重、决策和比例后另用新身份复现；整体Exact仍未显著提高。本轮报告已经消费，后续不得继续用它调参；更困难样本的原始I/Q与128维特征可分性检查优先于盲目加大比例或延长头训练。

## 冻结主干的全局压缩与分段统计验证（2026-11-05）

北京时间12:14:49启动 `runs/segment_statistics_cpu_20261105/`。先用2050/2051两种子筛选，再对满足check条件的特征方案扩展到2050～2054五种子；相同预算的global128控制始终完成五种子。全程CPU，最多五作业，各绑定4物理核，代码从冻结快照运行。107项测试及真实I/Q前向、统计池化和优化器步检查通过。

固定上一轮check选出的6.25%配对候选全部原始权重及阈值，以初始输出为零的4194参数小头，只向SawFM/MFT logits增加残差；其余四类输出、阈值逐元素保持参考值。三个输入为当前投影后global128、投影前均值/标准差global256、以及128×375时序特征按1/2/4段统计得到的1792维segments124。各组只用相同自然32k训练身份拟合标准化和PCA128，无白化，再使用完全相同的128→32 GELU→2小头。PCA冻结系数数量不同，并非完整内存容量匹配；保留方差和变换成本单独报告。负结果只否定该压缩实现，不能否定所有局部信息。

训练保持原32k自然池及2736个按功率差分层的训练配对，普通两目标BCE，每batch30自然视图＋1对，24轮×250步、AdamW lr0.0003、weight decay0.0001、cosine24轮；相同模型种子/轮次的自然与配对抽样序列完全一致。每组均按check固定阈值Exact及相同弱召回、FPR和弱AP约束选择轮次，保留epoch0作为无改善回退，避免把分段特征与额外训练、阈值调优混淆。

全新8064 calibration、2016 check、8064 report，另有4032条10～40dB报告及1152个新身份配对边留出对；核对99份历史身份文件无重叠。两个特征方案只有两种子check Exact相对冻结候选至少+0.1pp、不低于同预算global128且满足保护条件才进入五种子。全部选型、权重及决策SHA冻结后才提取最终报告；原固定阈值为主指标，大池逐目标F1和匹配校准FP数仅作辅助。报告再对预指定的每组合16条、合计1008身份做0/1/3/7采样点循环平移诊断，不参与选型。

正式验收要求同池Exact至少+0.5pp、4/5种子正向、配对条件95%区间下限为正；SawFM/MFT各自弱召回下降不超过1pp、各类FPR上升不超过0.5pp、两目标平均弱AP下降不超过0.2pp、尾类F1下降不超过0.3pp。确认特征收益须同时超过冻结候选和同预算global128；未进入五种子的方案只报告两种子check筛选，不冒充独立报告结论。没有方案通过时停止本路线，不自动扩大网络或配对比例。

入口为 [冻结方案](reports/segment_statistics_cpu_20261105/plan.html)、[结果页](reports/segment_statistics_cpu_20261105/analysis.html)、[实时进度](runs/segment_statistics_cpu_20261105/progress.json)；完成后保存 `summary.json`、`selected_models.json`、`completion_audit.json` 和 `pipeline_completed.json`。同一Python物理仿真器、固定数据池条件区间及缺少实测验证仍限制结论范围。

中断后可从原冻结快照继续，已完成作业跳过、训练可从末轮检查点恢复：

```bash
PYTHONPATH=runs/segment_statistics_cpu_20261105/source_snapshot/src CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 .venv/bin/python -u runs/segment_statistics_cpu_20261105/source_snapshot/experiments/run_segment_statistics_cpu.py --suite runs/segment_statistics_cpu_20261105
```

本轮于北京时间12:31:07完成，耗时16分18秒。两个特征方案均通过两种子check筛选，实际15个小头、360轮、90000优化器步全部完成，无失败。525组主/辅助指标重放、119份冻结源码、数据及原参考权重SHA、其他四类输出与相同抽样序列均通过审计。完成后独立复核再次验证全部原源码、数据和权重哈希，重放五种子的自然训练特征与历史缓存一致，核对全部15份PCA的训练身份绑定、均值、标准差和正交性；未修改模型、阈值或check选型。

同一8064条新身份报告、原固定阈值、五种子分别评测的结果如下，不能与历史不同报告池直接相减：

| 方法 | Exact | SawFM弱召回 | MFT弱召回 | 两目标平均弱AP | 两目标平均FPR |
| --- | ---: | ---: | ---: | ---: | ---: |
| 固定6.25%配对候选 | 74.683% | 51.582% | 70.138% | 77.911% | 4.322% |
| 相同预算global128控制 | 74.816% | 51.785% | 70.117% | 78.059% | 4.209% |
| 压缩前global256 | 74.965% | 52.009% | 69.904% | 78.485% | 4.095% |
| **1/2/4段统计segments124** | **75.724%** | **52.615%** | **76.791%** | **80.903%** | **3.742%** |

check预选segments124在独立报告上，相对固定候选Exact **+1.042pp**，五种子全部提高，固定池条件配对95%区间[+0.739,+1.345]pp；相对同预算global128为**+0.908pp**，区间[+0.665,+1.151]pp，同样五种子正向。两项比较均通过预先冻结的正式门槛。global128和global256相对固定候选分别+0.134/+0.283pp，虽各自条件区间为正，但未达到0.5pp的实用增益门槛；global256相对同预算global128的+0.149pp区间跨零，不能确认仅绕过投影层带来可靠额外收益。

收益集中在MFT：弱召回+6.652pp，区间[+2.733,+10.572]pp；MFT FPR下降0.938pp。SawFM弱召回+1.033pp，区间[−0.420,+2.485]pp，尚未确认稳定提高；SawFM FPR下降0.222pp。两目标平均弱AP+2.992pp，区间[+2.264,+3.720]pp，表明不只是固定阈值下的召回换误报。其他四类预测和阈值不变，尾类F1因此不变。每种子平均修正151.2条原错误，同时引入67.2条新错误，净增84条完全正确样本。

主报告15～20dB功率差两目标召回31.224%→36.309%；其中MFT召回35.539%→43.866%，弱边界AP34.355%→45.610%、AUROC84.499%→89.258%。新身份配对边留出Exact58.359%→59.366%，但其SawFM弱召回45.407%→45.148%，SawFM的困难边界仍未突破。10～40dB宽功率Exact53.527%→54.335%；20～30dB差距两目标召回6.382%→6.427%，几乎没有改善，不能宣称解决训练覆盖外的极弱分量。

1008条预指定报告身份在0/1/3/7点循环平移下，分段方案相对参考Exact分别+1.032/+0.893/+1.091/+0.734pp，均保持正向；这只是轻微平移诊断，不替代宽范围到达时间或实际观测协议验证。辅助大池目标F1下segments124为75.707% Exact；匹配校准FP数下75.560%，均不替换主固定阈值75.724%的验收结果，报告FPR并非严格相等。

综合支持采用分段统计作为新的内部验证候选，收益主要来自对已有局部时序统计的读取；保留其当前固定配置，后续先做独立复现及到达时间/物理参数变化，再针对SawFM验证物理辅助证据，不凭本轮结果自动修改主干。详细见 [完成复核](reports/segment_statistics_cpu_20261105/review.html)、[复核JSON](reports/segment_statistics_cpu_20261105/review.json)、[汇总图](reports/segment_statistics_cpu_20261105/outcome_overview.png)、[五种子模型清单](runs/segment_statistics_cpu_20261105/selected_models.json) 和 [完成审计](runs/segment_statistics_cpu_20261105/completion_audit.json)。复核入口为 `experiments/analyze_segment_statistics_cpu.py --suite runs/segment_statistics_cpu_20261105`；报告身份已消费，后续调参使用新的独立身份池。

## ISRJ分段统计头启动验证（2026-11-05）

`runs/isrj_segment_cpu_20261105/` 对当前固定segments124候选新增ISRJ头，比较相同4161参数的global128与segments124两组；其他五类logits与全部原阈值保持固定。小头输出零初始化，原B、FMB、线性头、局部头、配对头、分段头和训练集拟合的标准化/PCA变换均SHA绑定。继续使用相同自然32k训练池，新增头均自然采样，不加入新的定向配对，以隔离特征收益。每模型24轮×250步、batch32，共6000步/192000视图；AdamW lr0.0003、weight decay0.0001、cosine24轮、普通ISRJ BCE。相同种子/轮次的两组自然采样序列一致。

先用2050/2051两个种子筛选；segments124只有check Exact相对固定候选至少+0.1pp、不低于同预算global128且满足ISRJ弱召回/FPR/弱AP保护条件，才扩展至2050～2054五种子。global128始终完成五种子，实际新增7～10个模型、42000～60000优化器步。CPU最多五作业，各绑定4物理核。按check固定阈值Exact及相同保护条件选最佳轮次，保留epoch0回退；选型、权重和决策冻结后才提取独立报告。

新身份提供8064 calibration、2016 check、8064 report及4032条10～40dB功率偏移报告，与103份历史身份文件核对无重叠。最终报告还对预指定的1008身份做0/1/3/7采样点循环平移诊断。主指标沿用原固定阈值，ISRJ单标签F1与匹配校准FP数只作辅助。正式成功要求相对同池参考Exact至少+0.5pp、4/5种子正向、条件配对95%区间下限为正，ISRJ弱召回下降不超过1pp、各类FPR上升不超过0.5pp、ISRJ弱AP下降不超过0.2pp及尾类F1下降不超过0.3pp。确认分段特征收益须同时超过冻结候选和同预算global128。

真实I/Q前向与历史特征重放、两组优化器步及指标计算、零初始预测和其他五类输出不变检查均通过。测评集成测试实际执行calibration/check与冻结后的report序列，并验证缺少冻结或修改决策后禁止读取报告；这些测试使用合成记录，不消费正式报告。启动不代表已经获得性能收益。

[冻结方案](reports/isrj_segment_cpu_20261105/plan.html)、[自动结果页](reports/isrj_segment_cpu_20261105/analysis.html)、[实时进度](runs/isrj_segment_cpu_20261105/progress.json) 和 `coordinator.log` 保留状态。完成后自动生成 `summary.json`、`selected_models.json`、`completion_audit.json`、`pipeline_completed.json`。同一Python仿真器、固定数据池条件区间及缺少实测验证仍限制结论。当前授权只启动ISRJ特征读取验证，不自动增加定向配对、可训练投影或微调主干。

外部中断后可从相同冻结快照恢复，已完成作业跳过，头训练从末轮检查点继续：

```bash
PYTHONPATH=runs/isrj_segment_cpu_20261105/source_snapshot/src CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 .venv/bin/python -u runs/isrj_segment_cpu_20261105/source_snapshot/experiments/run_isrj_segment_cpu.py --suite runs/isrj_segment_cpu_20261105
```

本轮于北京时间13:44:07全部完成，从13:36:13起耗时7分54秒。两个特征组均进入五种子确认，10个新增小头、240轮、60000优化器步完成，无失败。270组主/辅助指标重放、原权重/变换/数据哈希、其他五类输出不变和相同自然采样序列均通过审计。完成后独立复核124份冻结源文件、全部原权重和数据SHA，重放10份PCA的自然训练身份绑定及均值/标准差；重新计算主报告15组模型预测并重放45组决策指标，未修改权重、阈值或check选型。

同一8064条新身份报告、原固定阈值、五种子分别评测：

| 方法 | Exact | MacroF1 | ISRJ弱召回 | ISRJ弱AP | ISRJ FPR |
| --- | ---: | ---: | ---: | ---: | ---: |
| 冻结当前候选reference | 75.045% | 95.004% | 60.675% | 74.259% | 6.512% |
| 同预算global128头 | 75.069% | 95.012% | 60.900% | 74.536% | 6.532% |
| **ISRJ分段统计头segments124** | **76.721%** | **95.385%** | **69.202%** | **82.568%** | **5.071%** |

check预选segments124相对冻结当前候选Exact **+1.677pp**，固定池条件配对95%区间[+1.378,+1.975]pp；相对同预算global128为**+1.652pp**，区间[+1.402,+1.902]pp，两项比较均五种子提高并通过预设正式门槛。global128相对参考仅+0.025pp，区间[−0.103,+0.152]pp，3/5种子正向，没有可靠整体收益。本轮reference与上一轮segments124是相同固定权重，但在新身份池为75.045%；不可直接用76.721%减上一轮不同报告池的75.724%，也不可跨池叠加两轮增益。

ISRJ弱召回+8.528pp，区间[+4.692,+12.363]pp；弱AP+8.309pp，区间[+6.630,+9.988]pp；FPR下降1.442pp，区间[−2.074,−0.809]pp，五种子方向一致。普通BCE、相同自然采样和原固定阈值下，分段统计相对同容量全局头有额外收益，支持局部信息读取的价值。采样、损失或阈值没有在组间变化。结合ISRJ生成结构，可以推测局部片段及重复结构的统计有用，但尚不能据此证明网络已学会物理机理或排除生成器位置线索。

每种子平均修正215.0条原本整组错误的样本，同时引入79.8条新整组错误，净增135.2条完全正确样本。ISRJ FP数258.4→201.2，FN数455.2→334.2，弱FN数384.6→301.2。只有ISRJ改变，SawFM/MFT及另外三类预测和阈值逐元素不变。新增4161参数的小头可以共用已有分段统计和PCA向量，不需要新增I/Q主干前向。

主报告ISRJ在10～15dB功率差下召回71.111%→79.700%，15～20dB下38.397%→46.795%；后者AP30.674%→43.667%、AUROC81.084%→86.586%。10～40dB宽功率报告Exact54.965%→56.027%，ISRJ弱召回41.581%→47.721%。但宽功率20～30dB差距下ISRJ召回9.640%→9.189%，仍仅约9%；AUROC虽59.372%→66.057%，排序改善没有在固定阈值下转化为召回提升，训练覆盖外的极弱分量尚未解决。

1008条预指定报告身份在0/1/3/7点循环平移下，分段方案相对参考Exact分别+1.825/+1.786/+1.706/+1.667pp，均保持正向。最长7点仅0.7微秒，不能替代大范围到达时间、采样片宽或复制次数变化测试，亦不能用子集约78%的Exact替换主报告76.721%。辅助ISRJ F1校准得到76.758% Exact，不替代主验收；匹配校准FP数后76.466%，ISRJ弱召回72.781%对参考60.818%，但实际报告FPR6.668%对6.563%，不能称为严格相同测试误报率。

当前剩余“仅该标签判错”的样本数均值为SawFM505.0、ISRJ380.6、MFT238.6、SinFM197.8、FMB180.4、AMA68.6。这是错误归因，不是可实现提升的预测；其他标签的单错误归因数会随ISRJ共错被纠正而变化，不能视为其预测发生回退。下一轮优先固定本轮ISRJ/MFT候选，验证SawFM可训练分段投影；同时以新身份做物理参数偏移及独立复现。ISRJ小头最佳轮次为11～19，末六轮弱AP变化很小，暂无盲目延长训练的依据。

详细见 [完成复核](reports/isrj_segment_cpu_20261105/review.html)、[复核JSON](reports/isrj_segment_cpu_20261105/review.json)、[汇总图](reports/isrj_segment_cpu_20261105/outcome_overview.png)、[候选验收](reports/isrj_segment_cpu_20261105/candidate_assessment.json)、[五种子模型清单](runs/isrj_segment_cpu_20261105/selected_models.json) 与 [完成审计](runs/isrj_segment_cpu_20261105/completion_audit.json)。复核入口为 `experiments/analyze_isrj_segment_cpu.py --suite runs/isrj_segment_cpu_20261105`；本轮报告身份已经消费，后续调参使用新身份。该内部v3仿真验证不能替代实测数据验证。

## SawFM可训练分段投影与容量控制（2026-11-07）

北京时间18:02:57启动 `runs/saw_projection_cpu_20261107/`。固定上一轮通过验收的ISRJ+SawFM/MFT分段候选，只允许SawFM logits改变；MFT、ISRJ及其他三类输出、全部原阈值保持固定。沿用自然32k训练池及只用训练身份拟合的标准化/PCA，不新增配对、不微调主干。

| 方法 | 输入与结构 | 可训练参数 | 用途 |
| --- | --- | ---: | --- |
| reference | 当前候选固定 | 0 | 同池基线 |
| pca_small | 冻结分段PCA128→32→1，复制当前SawFM权重 | 4161 | 额外训练控制 |
| pca_wide | 相同PCA128→442→1，前32单元复制原头、额外输出权重置零 | 57461 | 参数量控制 |
| learned_projection | 标准化1792维统计→32→1，将当前PCA与隐藏层折叠后初始化 | 57409 | 允许学习被固定PCA丢弃的方向 |

宽头与投影仅相差52参数，但形状和优化几何仍不同。三组均使用 `reference + f_current(x) - f_initial(x)`，初始整组预测逐元素一致；初始模型保持冻结并随检查点保存，需计入推理和存储成本，不能只报告可训练参数。折叠后的函数与原Saw分支经数值等价检查，更新后其他五类输出不变。

每模型24轮×250步、batch32、6000步/192000视图，普通SawFM BCE、AdamW、隐藏层lr0.00003、输出层lr0.0003、weight decay0.0001、cosine24轮、梯度范数上限5；三组相同种子/轮次使用完全相同自然抽样序列。先2050/2051两种子筛选，两种PCA控制始终完成2050～2054五种子；learned_projection只有check Exact相对reference至少+0.1pp、不低于两个控制并满足保护条件才扩展。实际12～15个新增模型、72000～90000优化器步。CPU最多五作业，各绑定4物理核。

新身份提供8064 calibration、2016 check、8064 report，以及各4032条宽功率、到达时间变化、SawFM周期变化报告，与106份历史身份文件核对无重叠。宽功率分量JNR10～40dB；到达时间50～90微秒，相对训练10～50微秒；SawFM扫频周期80～120微秒，相对训练20～80微秒。两个物理变化池仅用于冻结后的诊断，不能参与轮次、模型或阈值选择；到达时间仍使用既有仿真器np.roll约定，不代表实测传播信道。另保留1008条预指定报告身份的0/1/3/7采样点循环平移检查。

只用check固定阈值Exact及同一保护条件选择轮次，保留epoch0回退；按五种子check选模型，再冻结选型、全部权重和校准决策SHA，才读取报告。正式Exact验收仍为同池至少+0.5pp、4/5种子正向、条件配对95%区间下界为正，同时SawFM弱召回下降不超过1pp、各类FPR上升不超过0.5pp、SawFM弱AP下降不超过0.2pp、尾类F1下降不超过0.3pp。宣称可训练投影有额外特征收益须同时通过相对reference、pca_small、pca_wide的比较。Saw单标签F1和匹配校准FP数仅作辅助，不替代主固定阈值结果。

124项测试通过，真实I/Q前向与历史缓存重放、三组真实特征优化器步与指标、初始预测一致及其他五类固定检查通过。实现测试确认投影可学习PCA子空间外方向；测评集成测试覆盖calibration/check及八种报告条件，并验证缺少冻结或修改决策后禁止读取报告。

[冻结方案](reports/saw_projection_cpu_20261107/plan.html)、[结果页面](reports/saw_projection_cpu_20261107/analysis.html)、[实时进度](runs/saw_projection_cpu_20261107/progress.json) 和 `coordinator.log` 可复核。完成后自动生成 `summary.json`、`selected_models.json`、`completion_audit.json`、`pipeline_completed.json`。五种子分别评测，非集成；同一仿真器、固定身份池条件区间及次要比较未作多重检验校正仍限制结论，启动不代表已经取得增益。

外部中断后可从相同冻结快照恢复，完成作业跳过，头训练从末轮检查点继续：

```bash
PYTHONPATH=runs/saw_projection_cpu_20261107/source_snapshot/src CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 .venv/bin/python -u runs/saw_projection_cpu_20261107/source_snapshot/experiments/run_saw_projection_cpu.py --suite runs/saw_projection_cpu_20261107
```
