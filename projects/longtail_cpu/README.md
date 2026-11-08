# CPU 长尾基线与分段统计读出

这是服务器端 `longtail` 工程迁入的独立子工程。它与仓库原有 Route A/B、`experiment-log/` 并存，使用独立的 `src/longtail` 包、配置和测试。迁入日期为 2026-11-08，保留原运行代码与科学结果；本次不重新训练，也不把不同路线的分数直接比较。

## 从哪里开始

- [方法改动与效果交流总结](reports/项目改动与效果交流总结_2026-11-07.md)：按问题、改动、结果和结论介绍方法。
- [完整技术明细](reports/项目阶段结果与方法创新总结_2026-11-07.md)：历史对照、数值、方法边界与代码位置。
- [两条路线如何比较](COMPARISON.md)：训练覆盖与分段读取的差别、后续公平对照条件。
- [原服务器工程说明](README.server.md)：保留原实验命令与历史环境信息。
- [迁入源文件清单](delivery/source_import_manifest.json)、[历史运行收据](delivery/receipts/)和[外部产物索引](delivery/artifact_manifest.json)：用于复核来源和哈希。

## 当前方法与结果

输入为单条复数 I/Q（2×3000），标签顺序固定为 `[FMB, AMA, SawFM, SinFM, MFT, ISRJ]`。DFMS-IQ 自然训练分支提供全局和时序特征，FMB 使用组合采样专家；1/2/4 段均值/标准差经训练集 PCA128 后，分别为 SawFM/MFT 与 ISRJ 提供小型残差。

| 同轮独立池对比 | Exact 增益 | 结论 |
| --- | ---: | --- |
| SawFM/MFT 分段头相对同预算全局头 | +0.908 pp | 五种子通过原验收，主要改善 MFT |
| ISRJ 分段头相对同预算全局头 | +1.652 pp | 五种子通过原验收，弱召回/弱 AP/误报同时改善 |
| 最新 SawFM 宽头相对固定候选 | +0.141 pp | 低于 +0.5 pp 实用门槛，不采用 |

保留模型是 ISRJ 分段轮的候选。它在最新 8064 身份报告池上的 Exact 为 77.178%，是五种子各自测评的均值，不是集成。最新宽头虽然被 check 预选，但正式报告验收为 false。不同池分数不能相减，报告身份均已消费；后续调参需新身份。

## 环境与快速检查

所有命令在本子工程目录执行。Python 版本为 3.10–3.12，锁定依赖包括 PyTorch 2.5.1、NumPy 1.26.4、SciPy 1.14.1、h5py 3.12.1 和 scikit-learn 1.5.2。

```bash
cd projects/longtail_cpu
uv sync --frozen --group dev
uv run python tools/verify_delivery.py
uv run pytest -q
uv run python -m longtail.train --help
uv run python -m longtail.evaluate --help
uv run python experiments/run_saw_projection_cpu.py --help
```

本次先复用服务器依赖环境检查迁入，再用锁文件在本子工程独立 `.venv` 中离线安装37个包；独立环境下124项测试通过，8个命令入口正常，实际导入路径为本子工程的 `src/longtail`。这属于本机 Linux 验证，不替代其他平台安装测试。历史编排使用 Linux `fcntl` 和 CPU affinity，换机器须调整核绑定；`--help` 与单元测试通过不代表历史实验能脱离外部产物直接重跑。

## 数据、权重和本机工作方式

Git 保存源码、配置、测试、图表、轻量指标、原始收据及外部产物 SHA-256。数据波形、模型权重、PCA、预测和缓存仍是外部产物；其定位见 `delivery/artifact_manifest.json`，不宣称已经上传至 GitHub。冻结收据内保留原服务器路径，以保持原始字节和审计含义；路径映射应另记，不改写历史哈希。

在原服务器上，可显式挂接旧工程的历史产物：

```bash
python tools/attach_server_artifacts.py \
  --source /ultrafast_share/xuwang_group/wxwu/longtail \
  --datasets /ultrafast_share/xuwang_group/wxwu/datasets
python tools/verify_delivery.py \
  --check-source /ultrafast_share/xuwang_group/wxwu/longtail \
  --check-artifacts
```

挂接不会复制几十 GB 的缓存。`runs/` 是本子工程自己的目录，历史 suite 通过本地链接用于读取，新试验必须使用新的 suite 名称和输出目录。`projects/datasets` 的本机映射仅写入 `.git/info/exclude`，不会进入提交。不要在原历史 suite 上启动恢复或重训；它们保存的是已完成证据。

换服务器时，应先取得索引对应的数据、权重、PCA和预测，并记录迁移映射；当前索引中的原服务器路径不是公开下载地址。基础旧数据导航见仓库根的数据说明，新增 Python 身份池须使用其自身 manifest，不能与 MATLAB 数据混称为同一池。

## 主要实现

| 功能 | 文件 |
| --- | --- |
| DFMS-IQ | [src/longtail/models/dfms.py](src/longtail/models/dfms.py) |
| 分段统计、训练集 PCA、双标签残差 | [src/longtail/segment_statistics.py](src/longtail/segment_statistics.py) |
| ISRJ 独立残差 | [src/longtail/isrj_segment.py](src/longtail/isrj_segment.py) |
| SawFM 投影与容量对照 | [src/longtail/saw_projection.py](src/longtail/saw_projection.py) |
| 分层 one-bit 配对与目标头 | [src/longtail/targeted_validation.py](src/longtail/targeted_validation.py) |
| 固定参考与报告冻结流程 | [experiments/](experiments/) |

源文件导入前后的 SHA 分开登记。算法 `.py`、配置、锁文件及科学 JSON 保持字节一致；Markdown 的本机绝对链接转换为仓库相对链接，并清理两份文档的末尾空行，相关变化在迁入清单登记。历史CSV保留原始CRLF字节。补充的交付工具与 README 属于迁移适配，不改变原训练方法和结果。
