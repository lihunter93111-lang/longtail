# 三轴长尾场景识别

雷达复合干扰多标签识别研究代码，涵盖标签频率、组合频率和功率可见性三轴不均衡，以及复数 I/Q 建模、物理参数辅助任务、冻结决策诊断和配对功率分析。

## 目录

| 路径 | 内容 |
|---|---|
| [`projects/CJ_Imbalance_0716`](projects/CJ_Imbalance_0716) | 路线 A：`cjlt`、实验入口、配置、测试、研究报告和轻量冻结收据 |
| [`projects/CJ_Inbalance_0724`](projects/CJ_Inbalance_0724) | 路线 B：`cj0724`、I/Q 接口对照、多任务与专家实验、配置和测试 |
| [`data_generation/python_longtail`](data_generation/python_longtail) | AMA 六类 v3 物理方程的 Python 生成入口，显式区分训练、校准、选择和报告角色 |
| [`data_generation/0716`](data_generation/0716) | E 盘 0716 原工程对应的 MATLAB 生成程序 |
| [`data_generation/paper`](data_generation/paper) | 论文数据生成分支，含 AMA 和 `longtail6/solve_pure_label_longtail.py` |
| [`visualize_0716_closedset_train_test.py`](visualize_0716_closedset_train_test.py) | I/Q、频谱、匹配滤波和 STFT 可视化入口 |
| [`docs/research`](docs/research) | 项目导航、路线审计、功率配对说明和文献分析 |
| [`data/catalog`](data/catalog) | 数据版本审计、功率清单和哈希索引 |
| [`docs/release`](docs/release) | 源码来源、逐文件 SHA-256、整理范围和验证结果 |

两个 Python 工程保持原来的内部结构，分别从各自根目录运行。两份 MATLAB 生成程序分别留存；它们存在科学语义和标签版本差异，不能直接相互覆盖。

## 研究与合作入口

- [长尾基线与分段读出](docs/implementation/segment_readout.md)：CPU 方法统一在路线 A 的 `cjlt` 内，包含训练、评估、旧权重导入及与弱视图覆盖方案的对照方式。
- [仓库目录与复现导航](docs/research/仓库目录与复现导航.md)：逐层解释文件夹、代码调用链、数据角色与接手顺序。
- [2026-11-07 阶段复盘与协作计划](docs/research/2026-11-07_阶段复盘与协作计划.md)：01—05 实验结论、逐类代价和下一步计划。
- [实验复现入口](experiment-log/REPRODUCING.md)：外部数据交付、路径配置及分阶段命令。
- [按日期查看实验方案与服务器结果](experiment-log/README.md)
- [Codex 协作思考入口](docs/research/CODEX协作思考入口.md)：研究背景、证据边界和后续问题。
- [实验流程与阶段性结果](docs/research/当前最佳与最新试验结果说明.md)：数据预处理、模型、训练、评测及对应文件。
- [数据集版本与使用说明](docs/research/数据集版本与使用说明.md)：基础六类、九挡功率、长尾 v1/v2/v3 和 AMA 锁定测试的区别。
- [项目交接说明](docs/research/项目交接说明_长尾复合干扰识别.md)（[Word](docs/research/word/项目交接说明_长尾复合干扰识别.docx)）：已有工作、合作分工和候选创新方向。
- [长尾方法综述与试验建议](docs/research/长尾方法综述与试验建议.md)（[Word](docs/research/word/长尾方法综述与试验建议.docx)）：可开展的实验及公平对照要求。
- [文献库说明](docs/research/文献库说明.md)：本地文献目录索引；论文 PDF 与全文抽取文本保留在外部文献库。

以上研究说明按各自整理日期理解，发布时未重跑训练。Word 保留原稿，仓库内的可点击路径以 Markdown 为准。

## 环境与入口

建议使用 Python 3.10；本机验证使用已有的 `PY10` 环境。按机器选择合适的 PyTorch CPU/CUDA 版本，再安装其余依赖。

```powershell
conda activate PY10
python -m pip install -r requirements.txt

# 路线 A：先查看参数；正式运行前编辑配置中的数据路径
Set-Location projects/CJ_Imbalance_0716
python experiments/run_tapc.py --help

# 路线 B
Set-Location ../CJ_Inbalance_0724
python -m cj0724.train --help
```

原工程 README、配置和历史报告保留原始路径与实验状态。克隆后应根据本机位置调整配置或传入 CLI 路径参数；部分历史脚本依赖未发布的外部数据、旧模型实现或冻结预测，需先查阅[移植说明](docs/release/PORTABILITY.md)。

## 数据

数据仓库：[ModelScope · fulanya/longtail](https://www.modelscope.cn/datasets/fulanya/longtail)。如仓库需要授权，使用有访问权限的账号登录。

基础闭集、九挡功率和 v1/v2/v3 长尾版本的组织见[数据说明](data/README.md)。MAT 数据和模型权重不纳入 Git；下载后保持各版本独立目录，核对版本审计中的 SHA-256，并在运行配置中明确数据位置。

基础六类顺序为 `[FMB, SawFM, SinFM, MFT, VFT, ISRJ]`；AMA 长尾版本使用自己的标签契约，以对应 `audit_summary.json` 为准。九挡功率是共享样本身份的配对视图。无效参数为 NaN，参数损失只计算活动标签、有效 mask 且真值有限的位置。

## 证据与复现范围

本次发布整理已有源码和研究说明，不重跑训练或读取锁定测试。历史报告中的 pilot、smoke、development、diagnostic 和 locked-test 角色以各自收据为准；某一分支的分数不能替代其他分支的确认结果。

校准和阈值只在约定角色的数据上拟合；保留组合覆盖、种子、样本身份和实际可行 IF/CF。训练前冻结配置与 manifest，训练后保存成功、失败和停止收据。

源文件来源和内容哈希见 [`source_inventory.csv`](docs/release/source_inventory.csv)。本次验证范围和结果见 [`VALIDATION.md`](docs/release/VALIDATION.md)。
