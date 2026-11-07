# 实验复现入口

更新于 2026-11-07。以下命令从仓库根运行，示例路径需替换为本机实际位置。

## 环境与文件校验

Python 3.10、PyTorch CUDA、NumPy、SciPy、scikit-learn、h5py 和 matplotlib 依赖见根 `requirements.txt`。历史实测版本见 [验证记录](../docs/release/validation.json)。训练入口要求 CUDA。MATLAB 用于生成与物理配对。

```sh
python tools/verify_source_snapshot.py
python tools/verify_experiment_delivery.py
python experiment-log/2026-11-06-01-lt-abc/run_lt_abc.py --help
python experiment-log/2026-11-06-05-weak-component-goal/run_weak_goal.py --help
```

## 外部交付的必需内容

| 内容 | 从哪里定位 |
|---|---|
| 原始九类两份 MAT 和生成 config | 01 的 `local_20261106_01/manifest.json` 中历史来源 |
| 初始 F2 权重 | 01 的 `preflight.json` 中 `init_path` 与 `init_sha256` |
| 01 完整数据准备结果 | manifest、preflight、身份清单、fit_pool/dev_pool 的 MAT、generation_config 和 receipt |
| 150/160 轮恢复状态 | 相应外部运行目录中的 checkpoint |
| 05 配对训练数据 | 新池、角色 manifest、身份清单、配对 MAT 与 pairing receipt |

Windows 原运行根为 `E:/LJQ_code/LJQ_code_experiment_storage/`，其下有 `lt_abc_v1`、`validation_bce_v1`、`weak_component_v1`。哈希清单登记实际产物。Linux 接收数据后，在新的工作副本中映射 manifest 的 `fit_path`、`dev_path`、`historical_sources[].path` 与 preflight 的 `init_path`，保留原始收据，并记录路径迁移。波形和权重哈希用于确认内容一致。

## 01 的生成与训练

先在 MATLAB 执行，路径由调用者传入。

```matlab
repoRoot = '/path/to/longtail';
outputRoot = '/path/to/runs/lt_abc_reproduction';
sourcePath = '/path/to/original/closedset_9class_train.mat';
addpath(fullfile(repoRoot,'experiment-log','2026-11-06-01-lt-abc'));
generate_lt_abc_candidates(repoRoot, outputRoot, sourcePath);
```

然后运行 Python，两个 `--historical-pool` 都应提供。

```sh
python experiment-log/2026-11-06-01-lt-abc/run_lt_abc.py --stage pipeline --output /path/to/runs/lt_abc_reproduction --init /path/to/F2/best_model.pth --historical-pool /path/to/original/closedset_9class_train.mat --historical-pool /path/to/original/closedset_9class_test.mat
```

Windows 的 `start_local.ps1` 支持 `-OutputRoot`、`-Init`、`-HistoricalRoot`、`-CondaExecutable` 和 `-CondaEnvironment`。候选池的生成收据齐备后再启动训练。

## 02—04 的入口

```sh
python experiment-log/2026-11-06-02-epoch-diagnostic/run_epoch_diagnostic.py --source /path/to/01_full_run --output /path/to/new_150_run
python experiment-log/2026-11-06-03-epoch160/run_epoch160.py --source /path/to/01_full_run --previous /path/to/150_full_run --output /path/to/new_160_run
python experiment-log/2026-11-06-04-validation-bce-goal/run_resilient.py --source /path/to/01_full_run --output /path/to/new_bce_screen
```

04 的 `continue_goal.py` 根据 `decision.json` 决定后续动作。历史筛选结果为 `confirmation_required=false`。该入口现支持 `--matlab-executable` 和 `--source-config`，在新协议确实触发确认时使用实际环境路径。

## 05 的生成、配对与训练

以下顺序复现已冻结的历史方案。生成器和准备脚本内的身份种子、角色划分种子均固定。另立科学实验时需在新方案中分配新的身份域和输出目录。

1. MATLAB 调用 `generate_fresh_candidates(repoRoot, dataRoot, sourcePath, promptPath)`。四个参数依次为仓库根、新数据根、原始九类 train MAT、05 的 `GOAL_PROMPT.txt`。
2. Python 完成身份验收与角色划分。

```sh
python experiment-log/2026-11-06-05-weak-component-goal/prepare_fresh.py --prompt experiment-log/2026-11-06-05-weak-component-goal/GOAL_PROMPT.txt --original /path/to/01_full_run --output /path/to/weak_data
```

3. MATLAB 在加入 05 目录后，调用 `generate_paired_views(repoRoot, dataRoot)`。它重建原始信号并核对目标分量干预。
4. Python 先验收，再执行预定训练和分析。

```sh
python experiment-log/2026-11-06-05-weak-component-goal/run_weak_goal.py --source /path/to/weak_data --output /path/to/new_weak_run --preflight-only
python experiment-log/2026-11-06-05-weak-component-goal/run_weak_goal.py --source /path/to/weak_data --output /path/to/new_weak_run
```

`--preflight-only` 会建立本轮输出及验收记录。正式命令使用相同 source/output 与配置。默认执行 A/B/W、种子 2060—2064，随后通过 `analyze_goal.py` 计算配对指标并生成报告。分析器会将轻量结果复制回其同层 `results/`，因此复现应使用独立 checkout 保存该次输出。

## 科学复现与历史记录

本次归档的原始结果保持各自执行时的代码哈希。2026-11-07 的入口改动涉及 01 的 Conda 可配置参数、输出目录创建，以及 04 的 MATLAB 参数、跨平台进程标志和恢复包装器参数解析。恢复包装器现支持 `--help` 及 `--output=路径`。新运行会记录当前源文件哈希。训练核心及原指标收据按历史版本归档。

05 的已评价确认池用于查阅已有结论，后续选参应在新协议分配的开发数据上完成。各轮的 `decision.json` 与实际角色记录共同决定证据范围。
