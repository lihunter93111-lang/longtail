# E盘项目中间数据清理建议

统计日期：2026-11-08。以下依据实际文件清单、转换收据、项目停止报告及 11 月 6 日至 7 日最新实验归档整理；未删除、移动或改写 E 盘数据。本次已在本清单旁备份 302 份轻量 CSV/JSON 元数据，逐文件 SHA-256 一致。容量为逻辑文件大小，实际释放量以文件系统为准。

## 最适合先处理的文件

| 顺序 | 路径或数据 | 可处理的 MAT 大小 | 判断与条件 |
|---|---|---:|---|
| A | `E:/LJQ_code/CJ0724_rechunked_data/` 下 330 个缓存 MAT | **15.95 GiB** | 优先候选；330/330 均有转换收据且对应原始文件仍在，原始大小与收据一致。保留原始数据与 JSON 收据后可重建 |
| B1 | `E:/LJQ_code/CJ_Inbalance_0724/generated_data/p8b_axis_stratified_longtail_seed1310000000/` | **21.31 GiB**，9 个 MAT | P8b 三种子开发门失败，已停止；建议原始数据备份后清理 |
| B2 | `E:/LJQ_code/CJ_Inbalance_0724/generated_data/p8c_vft_hard_negative_logit_cap_seed1325000000/` | **21.31 GiB**，9 个 MAT | P8c 三种子开发门失败，已停止；建议原始数据备份后清理 |
| C | `E:/LJQ_code/CJ0724_generated_data/` 下历史生成的闭集训练 MAT | **64.49 GiB** | P3o-L至U历史原始数据，可迁到其他盘冷归档；不列为无备份直接删除项 |

A 是可重建缓存，B 是已停止实验的原始数据，两者性质不同。A+B 合计 **58.57 GiB**，只有在接受 B 的复现代价或完成数据备份后，才可作为组合清理额度。C 与 A 是源数据—缓存关系：不能同时删除两者后仍声称保留了可重建来源。

## A 缓存为什么可优先处理

`CJ0724_rechunked_data/` 是为旧路线 B 按样本读取生成的 float32、sample-chunked I/Q 缓存，转换代码为 [rechunk_independent_power_data.py](../../../projects/CJ_Inbalance_0724/cj0724/rechunk_independent_power_data.py) 与 [rechunk_single_mat.py](../../../projects/CJ_Inbalance_0724/cj0724/rechunk_single_mat.py)。原始来源目前位于 `E:/LJQ_code/CJ0724_generated_data/`。

逐份转换收据已映射 330 个缓存，原始缺失数 0、原始大小不符数 0、未被收据覆盖的缓存数 0。这里只核对存在性与收据字节数，未重新通读波形或验证全部字段。历史收据保留 D 盘路径，已在 [E盘缓存源映射.json](./E盘缓存源映射.json) 中登记当前 E 盘实际对应路径；重新运行时需要显式指定当前源路径和目标路径。缓存转换不保留完整 MATLAB config；原始 MAT 仍有复现和诊断价值。

清理目标限于 CSV 中 A 类 MAT，**保留 JSON 转换收据和完整原始 MAT**。不应直接删完整目录。已复制的转换收据见 `E盘元数据备份/CJ0724_rechunked_data/`，原始 CSV 身份和功率清单也已备份。缓存从原始数据重建仍需要 I/O 时间；旧脚本直接读取缓存的复现流程会暂停，直到缓存重建。

11 月 6 日 01—05 的归档记录与 11 月 7 日导航中未发现这些旧缓存根目录的直接引用；当前相关原始数据、最新试验目录和模型初始化位置另列保留。未对所有旧脚本做动态依赖分析。

## B 停止试验的依据

- P8b：[三种子开发止损记录](../../../projects/CJ_Inbalance_0724/docs/p8b_axis_stratified_longtail_development_stop_20260802.md)，明确 `stop_no_confirmation_data`，不进入追加确认。
- P8c：[三种子开发停止报告](../../../projects/CJ_Inbalance_0724/docs/p8c_vft_hard_negative_logit_cap_development_stop_20260803.md)，明确八项必要门失败，停止对应精确机制。

这两套数据是各自独立的九功率开发/训练身份池，并非当前 11 月 AMA 主线的数据。可先保留生成器与配置哈希、随机种子、身份和功率清单、审计报告、输出预测、阈值和结果，再将大 MAT 备份到其他磁盘。元数据备份不是原始数据备份；停止状态不等于数据失去全部科学价值。

## C 原始生成数据的分支依据

`CJ0724_generated_data/` 涵盖 P3o-L/M/N/O/P/Q/R/S/T/U。L 为组件条件残差，M—R 为峰、拒识、相干、支持曲线与分离诊断，S 为 MFT oracle 去除诊断，T/U 为残差路线。对应停止/冻结报告位于 `projects/CJ_Inbalance_0724/docs/p3o*.md`。

L/M/N/O/P/Q/R/T/U 未通过各自门槛；S 虽未晋级，却有明确的因果诊断信息，尤其不应把 S 原始数据简单标成无价值文件。保留完整原始数据的一份冷备份，有利于将来复核负结果和 oracle 证据。若只想立即腾空间，先处理 A，不必同时处理 C。

## 暂不建议清理

| 目录 | 已统计大小 | 保留理由 |
|---|---:|---|
| `E:/LJQ_code/LJQ_archive/` | 170.10 GiB | 对当前 Route-B 目录按相对路径比较，归档中的 3451 文件没有同相对路径的现存副本；不能认作重复备份。可能存在改名副本，未作全盘哈希去重 |
| `E:/LJQ_code/CJ0724_frozen_generation_artifacts/20260802/` | 2.29 GiB | P3o-J/K 空间不足事故的冻结工件，包含中断生成留下的 partial MAT；历史冻结记录要求保留审计。可备份到其他介质，不列入本次优先清理 |
| `E:/LJQ_code/CJ_Inbalance_0724/generated_data/` 中除 P8b/P8c 外的部分 | 根目录总共188.14 GiB，包含上述 B | 含原始与重分块、开发与确认等多种角色，不应整目录删除。P8b/P8c 已计入根目录大小，不重复累计 |
| `E:/LJQ_code/CJ_Imbalance_0716/generated_data/route_a_locked_test_ama_seed4050000000/` | 1.16 GiB | 独立锁定测试证据，保留 |
| `E:/LJQ_code/LJQ_code_experiment_storage/lt_abc_v1/`、`validation_bce_v1/`、`weak_component_v1/` | 分别约1.67 GiB、67.97 MiB、2.03 GiB | 11 月最新实验与确认来源，保留 |
| `E:/LJQ_code/CJ_Imbalance_0716/frozen/` | 未纳入本轮清理额度 | 其中 seed2039 的 backbone 仍被最新训练引用，保留 |
| `E:/论文相关/干扰数据/复合干扰数据0716/九类/full_power_10_30dB_v1/` | 两个大原始 MAT | 最新预检仍引用，保留 |

`E:/LJQ_code/LJQ_code_experiment_storage/p3oj_component_conditioned_20260802/` 约 6.14 GiB，是历史 P3o-J 准备数据；相关报告分为执行契约异常和生成失败两条记录，先逐文件确认与冻存工件关系，不据目录名直接清空。

## 明细与证据

- [E盘中间数据候选.csv](./E盘中间数据候选.csv)：逐个 MAT 的绝对路径、容量、优先级和保留条件。
- [E盘缓存源映射.json](./E盘缓存源映射.json)：330 个缓存与当前原始来源的映射。
- [E盘元数据备份清单.json](./E盘元数据备份清单.json)：本地元数据备份的原路径与 SHA-256。
- [E盘中间数据扫描.json](./E盘中间数据扫描.json)：本次相关目录的文件和容量统计。

前次清单中两套基础六类与九挡功率数据的约47.36 GiB去重额度是另一组文件，尚须完整哈希比较；不与本次已核实可重建缓存混为一谈。上述总量不包含无条件清空整个历史输出目录。
