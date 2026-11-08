# CPU 子工程增量迁入

迁入日期为2026-11-08，基于新仓库 `1ecfd499a6bc914d5f7dbd2e7334efe45707ff26`，分支为 `feat/cpu-segment-readout`。

## 交付内容

- 新增 [projects/longtail_cpu](../../projects/longtail_cpu/README.md)：独立包、源码、配置、测试、生成入口、结果报告与历史轻量收据。
- [313份导入文件清单](../../projects/longtail_cpu/delivery/source_import_manifest.json)分别记录源/目标哈希。算法Python、配置、锁文件和科学JSON保持原字节；文档本机绝对链接、本地忽略规则及两份Markdown末尾空行作适配，历史CSV保持原CRLF字节。
- [外部产物清单](../../projects/longtail_cpu/delivery/artifact_manifest.json)核对92份模型/PCA/主报告预测，共62325082字节；另保留16份数据文件的冻结哈希引用。本次没有通过Git交付这些二进制文件，也没有重新计算大数据文件哈希。
- [两条路线比较入口](../../projects/longtail_cpu/COMPARISON.md)，以及根目录和协作页的新工程导航。

本机历史产物使用忽略的本地链接。新实验须使用新suite名称，不能在历史suite上恢复训练。原服务器工作区、权重、阈值和科学选型未修改；原Route A/B和01—05实验源码均保留。

## 验证范围

独立环境按锁文件离线安装，124项测试通过；8个CLI帮助入口通过；导入文件与原服务器逐文件核对；92份外部产物哈希核对；关键Markdown链接63处可解析；原2690份源码快照校验通过。本次不启动新训练，不重新评价科学增益，不读取新的锁定测试。

原2026-11-07交付清单保持字节不变。根README和协作页新增导航，因此当前树的校验使用[2026-11-08增量清单](2026-11-08-cpu-integration-manifest.json)，校验工具默认指向该新清单。旧清单中的原内容可在基点提交 `1ecfd49` 核验，不把导航调整当成历史实验被改写。

```bash
python tools/verify_source_snapshot.py
python tools/verify_experiment_delivery.py
cd projects/longtail_cpu
.venv/bin/python tools/verify_delivery.py
.venv/bin/python -m pytest -q
```

结构化验证记录见 [2026-11-08-cpu-validation.json](2026-11-08-cpu-validation.json)。
