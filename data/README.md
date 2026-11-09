# 数据入口与版本

MAT 数据保存在 [ModelScope · fulanya/longtail](https://www.modelscope.cn/datasets/fulanya/longtail)，本 GitHub 仓库保留数据契约、审计与哈希索引。

| ModelScope 子目录 | 规模与角色 | 本地说明 |
|---|---|---|
| `closedset_6class/` | train/test 各 25,200；六类 63 组合 | [`catalog/closedset_6class`](catalog/closedset_6class) |
| `power_sweep_9level_v1/` | 10–50 dB 九挡，每挡 train/test 各 25,200；共享身份的配对视图 | [`catalog/power_sweep_9level_v1`](catalog/power_sweep_9level_v1) |
| `six_class_pure_label_longtail_v1_IF3_CF8_N8000/` | 8,000 / 8,000，旧版纯标签长尾 | [v1 审计](catalog/six_class_pure_label_longtail_v1_IF3_CF8_N8000/audit_summary.json) |
| `six_class_pure_label_longtail_v2_IF3_CF8_N8000_AMA/` | 8,000 / 8,000，AMA 标签版本 | [v2 审计](catalog/six_class_pure_label_longtail_v2_IF3_CF8_N8000_AMA/audit_summary.json) |
| `six_class_pure_label_longtail_v3_IF3_CF8_N8000_AMA/` | 8,000 / 6,048，AMA；诊断测试均衡覆盖 63 组合 | [v3 审计](catalog/six_class_pure_label_longtail_v3_IF3_CF8_N8000_AMA/audit_summary.json) |

此表描述数据目录与本地审计，不作为远端上传完成证明；九挡大文件上传由原数据上传任务处理，本次代码发布没有重新验证其远端完整性。

基础六类顺序为 `[FMB, SawFM, SinFM, MFT, VFT, ISRJ]`，v2/v3 为 `[FMB, AMA, SawFM, SinFM, MFT, ISRJ]`。v1 的审计不直接声明标签名，使用时还需核对 MAT 元数据及对应生成配置。各版本不能只按相同列数混用。

长尾训练的审计记录 IF=3、CF≈7.85366、每组合 41–322 条。v3 诊断测试为每组合 96 条，train/test 的 `receiver_noise_seed` 交集为零。这些是已有数据收据的记录，本次仅复制并验证收据来源，没有重读 MAT 或训练模型。

下载后核对各目录收据中的 `files` 哈希或功率目录的 `power_sweep_file_inventory_sha256.csv`。保留目录名和原审计，不修改原始文件。历史脚本的绝对路径需要按本机情况显式配置，见[移植说明](../docs/release/PORTABILITY.md)。

## 11月新增实验数据

11月8日整夜实验及11月9日续做的完整文件位置、数据角色、逐文件大小清单和轻量审计入口见[新增实验数据清单](catalog/experiments_2026-11-09/README.md)。
