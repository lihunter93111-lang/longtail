# 2026-11-07 实验归档发布

目标仓库为 `lihunter93111-lang/longtail`，起点为 `552f762232e050d028be38e913104d5263ebf5ab`。

## 交付范围

- 01—05 的源码、方案、审计与轻量结果。
- 从外部运行根补齐 01 与 160 轮诊断共 73 份文件，逐文件来源见 [导入清单](2026-11-07-imports.json)。
- [逐层目录导航](../research/仓库目录与复现导航.md)、[复现入口](../../experiment-log/REPRODUCING.md)、[阶段复盘](../research/2026-11-07_阶段复盘与协作计划.md)及 [师弟材料评审](../research/collaboration/README.md)。
- 实验交付哈希清单及对应校验工具。

现有源码快照已经建立。本次采用增量归档，初始 `prepare_source_snapshot.py` 的目标目录保护适用于首次复制。初始源码清单继续通过 `verify_source_snapshot.py --check-originals` 核对。

## 代码改动

01 启动脚本支持 Conda 可执行文件和环境参数，运行前创建输出目录。04 的确认编排支持 MATLAB 可执行文件和原始配置路径参数，按系统选择进程窗口标志。04 的恢复包装器修复了帮助入口直接读取 `--output` 导致异常的问题，并支持标准等号参数写法。

原始训练和科学结果的哈希按运行时版本保留。上述入口改动由新提交追踪，实际运行时生成当前代码哈希。

## 验证

实际检查记录见 [结构化结果](2026-11-07-validation.json)。包括初始源码哈希、导入字节核对、历史交付清单、Python/JSON 解析、CLI 帮助、文档链接及现有 Route A/B 基础测试。实验清单可通过以下命令重新检查。

```sh
python tools/verify_source_snapshot.py
python tools/verify_experiment_delivery.py
```

本次验证范围为归档完整性和轻量入口兼容性。训练、MATLAB 生成及完整科学复验应按复现说明另行执行。师弟总结中的服务器实现与预测尚待交付。

原始 MAT、模型权重及完整预测继续存于外部运行根，并由历史产物清单记录。工作区现有 `modelscope_verify_final/` 和根目录两份 Word 文件保留本地，已有历史 Word 交接稿位于 `docs/research/word/`。
