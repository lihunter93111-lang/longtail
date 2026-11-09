# 11月新增实验数据清单

## 完整文件存储

| 实验 | 本地根目录 | 文件数 | 总字节数 | 文件索引 |
|---|---|---:|---:|---|
| 11月8日整夜实验 | `E:/LJQ_code/LJQ_code_experiment_storage/overnight_v1/20261108_01` | 1989 | 6313317152 | [逐文件大小](overnight_external_files.csv) |
| 11月9日续做 | `E:/LJQ_code/LJQ_code_experiment_storage/daytime_v1/20261109_01` | 1116 | 1472377024 | [逐文件大小](daytime_external_files.csv) |

清单覆盖各根目录内实际存在的文件，包括数据、权重、预测、日志和轻量收据。字节统计包含各运行副本。CSV用于定位和容量盘点，完整性哈希采用实验审计和冻结记录中已有的值。本次归档对发布副本另行计算SHA-256。

## 数据角色

11月8日包含主数据 `data`、近均衡数据 `balanced_data`、独立训练划分 `replication_data` 和最终确认 `final_confirmation_data`。主划分与复验各使用8000个fit身份，early_stop/calibration/development分别为1008/1008/4032。最终确认另有4032个新身份。六标签顺序为FMB、AMA、SawFM、SinFM、MFT、ISRJ。具体审计及模型冻结记录见[整夜结果目录](../../../experiment-log/2026-11-08-01-overnight/results/)。

11月9日复用上述两套训练与开发角色，新增 `axis_data` 的配额与身份映射用于数据轴开发试验。同身份功率视图的共享关系、实际训练呈现次数与独立身份数由各协议和收据记录。[停止时结果目录](../../../experiment-log/2026-11-09-01-wide-followup/results_at_stop/)

基础数据的ModelScope入口见[数据总导航](../../README.md)。这两轮新增MAT、模型和完整预测保存在上列本地目录，其远端可下载状态尚未核实。GitHub此次发布CSV/JSON指标、分布、审计、冻结哈希、图像及代码。
