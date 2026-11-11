2026-11-11 标签正负边际匹配与组合加权控制

阅读入口
RESULTS.txt：本轮实测结论、强对照与代价。
RESULTS.html：带图的本地阅读版。
NEXT_ROUND_PLAN.txt：两人后续分工、阶段Goal与3—5天会合。
COLLABORATOR_GUIDE.txt：每层目录作用、调用链、模型及数据接口。
REPRODUCING.txt：原机复核、跨机推理和重新训练的边界。

实际完成
新增B标签边际匹配的2划分×5种子完整训练；复用匹配A/C和独立主模型。100决策冻结后，两套各4032身份的新确认已评价并独立复算。未来四条件双长尾数据索引已验收，矩阵训练尚未执行。

状态与证据
results/completion_audit.json为逐项要求核验；results/archive_provenance.json为外部证据的复制来源；delivery_manifest.json为本版本文件哈希。
当前确认身份已消费，不能继续选择新结构或阈值。
外部模型和数据包见results/handoff_bundle_receipt.json，约632MB；Git仅保存源码和轻量证据。
