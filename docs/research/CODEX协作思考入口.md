# 三轴长尾场景识别 Codex 协作思考入口

更新时间：2026-11-03

## 给接续工作的 Codex

请把本文作为项目研究上下文。你的任务不是重复罗列代码，而是基于现有证据帮助设计下一阶段可证伪、可复现的长尾实验。先阅读本文列出的入口文件，再回答用户的问题。若远程仓库只包含源码、轻量清单和收据，而没有 MAT、checkpoint 或 outputs，请明确说明证据边界，不要假设外部文件已经存在。

建议输出顺序：

1. 复述当前研究问题和数据角色；
2. 说明已有结果能支持什么、不能支持什么；
3. 给出一个最小可执行试验和对照组；
4. 指定数据身份、随机种子、指标和停止门槛；
5. 说明结果如何区分“新表征收益”“采样收益”“阈值收益”和“观测不可辨识”。

## 1. 项目目标

项目研究雷达复合干扰的六类多标签识别。输入为长度 3000 的复数 I/Q 脉冲，标签为：

```text
[FMB, AMA, SawFM, SinFM, MFT, ISRJ]
```

六类允许 63 个非空组合。项目关注三个长尾轴：标签频次、组合频次和功率可见性。弱分量可能被强分量遮蔽，因此低召回可能来自模型边界，也可能来自信号本身缺少可辨识证据。九挡功率数据是共享身份的配对视图，不能直接当作九倍独立样本。

当前优先级是先把闭集长尾多标签识别做成可信、可复现的实验主线，再推进开集未知成分发现和持续学习。

## 2. 代码和数据入口

本仓库：

- 项目主页：[lihunter93111-lang/longtail](https://github.com/lihunter93111-lang/longtail)；历史版本：[fulanya55/longtail](https://github.com/fulanya55/longtail)
- 代码入口：[projects/CJ_Imbalance_0716](../../projects/CJ_Imbalance_0716)
- 路线 B：[projects/CJ_Inbalance_0724](../../projects/CJ_Inbalance_0724)
- 数据说明：[data/README.md](../../data/README.md)
- 数据审计：[data/catalog](../../data/catalog)
- 迁移限制：[docs/release/PORTABILITY.md](../release/PORTABILITY.md)

主要实现：

```text
projects/CJ_Imbalance_0716/cjlt/models/phase_multiscale.py
projects/CJ_Imbalance_0716/cjlt/models/backbone_adapter.py
projects/CJ_Imbalance_0716/cjlt/experiments/run_route_a_ama_direct_sixclass_v3_2x2_20260905.py
projects/CJ_Imbalance_0716/experiments/analyze_route_a_ama_direct_sixclass_v3_2x2_20260905.py
```

大体量 MAT、checkpoint 和 outputs 保留在原机器的外部数据根目录，没有随 Git 发布。跨电脑时应优先复核仓库中的 manifest、audit 和 source inventory，再决定需要用户提供哪些外部文件。

## 3. 当前数据和协议

主要版本：

- 基础六类闭集：63 组合，每组合 400 条，是平衡根数据；标签顺序为 `[FMB, SawFM, SinFM, MFT, VFT, ISRJ]`。
- 长尾 v1：早期派生版本，标签语义需要依据自身审计确认。
- 长尾 v2 AMA：用于 BCE exact、冻结 logits 和错误归因。
- 长尾 v3 AMA：当前 Route-A Stage-A 主对照，标签顺序为 `[FMB, AMA, SawFM, SinFM, MFT, ISRJ]`。

预处理：MAT/HDF5 读取后转换为 `[B,2,3000]` 实部/虚部，并做逐样本 RMS 归一化。参数估计只在有效 `parameter_mask` 和有限真值位置计算，不能把无效 NaN 当作零。

历史 v3 角色：fit 8000、calibration 1008、check 1008、development-test 4032。历史 v3 同时使用 calibration 选择 checkpoint 和拟合阈值；新实验应将 early-stop 和 calibration 拆开。当前 development-test 已被开发流程消费，不能重新标为 locked-test。

## 4. 当前模型和最佳结果

主模型是 F2/DFMS 复数 I/Q 网络：复数多尺度编码（16→32→48→64）→ Cartesian/magnitude/phase delayed fusion → kernel 3/7/15、dilation 1/2/4 时序残差 → mean/std 池化 → 128 维投影 → 6 个 logits。约 257,511 个参数。训练基线是 AdamW、lr `1e-4`、weight decay `1e-4`、batch 32、8 epochs。

v3 主对照内当前观测均值最好：

```text
BCE + ordinary sampling + F1 calibration
```

五种子 development-test：

- Macro-F1：`0.925639 ± 0.001556`
- 六标签 combination exact：`0.651835 ± 0.005762`
- Macro-AUPR：`0.977776`
- rare combination accuracy：`0.548077`

这是开发证据，不是独立最终锁定测试冠军。anti-alias 的一次性九功率 locked-test exact 为 `0.382099` 对 `0.376455`，数据、预算和集成方式不同，不能与 v3 直接排名。

结果入口：[当前最佳与最新试验结果说明.md](./当前最佳与最新试验结果说明.md)。

## 5. 已有结果给出的限制

- BCE、Logit Adjustment、ordinary、balanced-cycle 和阈值校准已做过部分系统比较；Logit Adjustment 没有形成稳定整体优势。
- balanced-cycle 每轮预算 8000 只使用约 4890 个不同 fit 身份，重复率约 0.38875，不能称为严格无放回均衡。
- Base-DB/ASL、Focal/DB 冻结头、功率课程、visibility auxiliary、若干 matched-filter/形态融合、IQ 分组、Physical FiLM、Temporal Query 和多种 MFT→ISRJ 路线已有负证据或停止边界。
- 冻结 logits 的四因素 Shapley 联合 oracle 只解释约 47.8% 的 exact 损失，主要指定错误包括 ISRJ/SawFM/MFT 漏检和 FMB 误报；oracle 是诊断上界，不是模型实测增益。
- 许多低功率改进未通过严格对照，不能把不可观测问题误写成网络问题。

历史总审计：[复合干扰路线 A/B 总审计](./复合干扰路线A_B全部尝试专业Prompt与总审计报告_20260806.md)。

## 6. 文献库得到的长尾方法框架

本地文献库共 44 篇已校验 PDF，主体覆盖开集、GCD、增量 GCD、开放世界持续学习和复合干扰。可迁移到当前任务的长尾方法分为：

1. **数据重采样**：label-aware、combo-aware、power-aware、balanced batch、两阶段 sampler 和真正无放回 Anchor。
2. **损失重加权**：weighted BCE、Focal、Asymmetric Loss、class-balanced、Distribution-Balanced、Logit Adjustment、Balanced Softmax。多标签任务必须单独处理正/负代价，不能直接套单标签公式。
3. **分类头解耦与决策校准**：冻结 backbone 重训 head、balanced classifier retraining、logit bias、temperature、逐标签 F1 阈值和组合 exact 阈值。
4. **表征与原型**：supervised/self-supervised contrastive、prototype、center loss、跨功率一致性和尾类局部特征；必须防止把每个组合硬编码成 63 类。
5. **困难样本与伪标签**：one-bit hard negatives、低功率边界、一致性训练、置信度伪标签和 teacher-student；必须记录伪标签覆盖率与精度。
6. **GCD/开集/持续学习**：先识别已知成分，再在已知成分条件化 residual 上发现未知成分；未知簇必须有稳定性、纯度、误晋升率、重复建类和回滚指标。

详细建议：[长尾方法综述与试验建议.md](./长尾方法综述与试验建议.md)。

## 7. 推荐的下一阶段实验

### 第一阶段：可信长尾基线

用同一 v3 新身份池、五种子、相同初始化和相同训练步数比较：

```text
ordinary BCE
label-aware sampler
combo-aware sampler
weighted BCE
Logit Adjustment
F1 calibration
```

先回答收益来自采样、损失还是阈值，而不是先增加网络模块。

### 第二阶段：拆分表征和决策

固定同一 backbone logits，比对固定 0.5、逐标签 F1、组合阈值、temperature/bias 和 classifier retraining；再将最佳决策规则带回重新训练模型。报告头类 FPR 和尾类 recall，防止只优化 Macro-F1。

### 第三阶段：只差一个标签的困难组合

以 ISRJ、SawFM、MFT 和 FMB 为重点，构造身份和背景元数据完整的 one-bit hard negatives。先做数据可辨识性审计，再比较 Focal/Asymmetric、标签局部头、contrastive 和组合 margin。若观测本身没有充分差异，应停止网络堆叠并改变观测协议。

### 第四阶段：跨功率和组合泛化

使用配对功率视图，报告每个功率层、每个标签和每个组合；增加“留出组合但不留出成分”的测试，检验未见已知组合是否被错误拒识。

### 第五阶段：成分级未知与持续学习

闭集主线稳定后，再研究“已知成分条件化 residual → 未知成分聚类 → 风险门晋升”。当前六类只能支持代理未知实验，不能单独证明真实未知机理识别。

## 8. 给其他 Codex 的具体问题

请基于本文和仓库文件，优先帮助回答以下问题：

1. 如何在多标签任务中定义同时控制标签频次、组合频次和功率可见性的长尾 Anchor？
2. 哪种 sampler/loss 组合最适合先做最小五种子实验，如何保持身份和训练预算公平？
3. 如何设计 one-bit hard-negative，使它既有物理可辨识性又不引入标签或身份泄漏？
4. 如何区分阈值校准收益、分类头收益和 backbone 表征收益？
5. 对 ISRJ/SawFM/MFT 漏检，哪些失败证据说明应改观测协议而不是继续改网络？
6. 如果推进开集，怎样证明 residual 聚类发现的是未知成分，而不是已知背景组合？
7. 哪些创新主张已被文献覆盖，哪些主张需要新的完整对照才能成立？

## 9. 证据规则

不要把 development、diagnostic、pilot 和 locked-test 混成一个分数。不要把代理未知、未见组合和真实未知物理成分混写。不要把 oracle、Shapley、后验修正或 JNR-only 结果写成完整分类模型已经通过。所有新结论都应附数据版本、身份规则、随机种子、阈值来源、checkpoint、指标和失败分支。

文献正式引用以 PDF 原文页码为准。文献入口：[文献库说明](./文献库说明.md)，论文 PDF 和抽取全文保留在外部本地文献库；分析入口：[文献深度分析](./按Prompt输出_文献库论文深度分析_20260804.md)，创新审计：[文献综述与创新审计](./开集复合干扰文献综述与创新审计_20260801.md)。
