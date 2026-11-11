"""Readable results from audited JSON; draft until confirmation and delivery pass."""
import argparse,html
from common import *

def pct(x):return f'{100*x:.3f}'
def ci(x):return f'{100*x[0]:+.3f} [{100*x[1]:+.3f}, {100*x[2]:+.3f}] pp'
def mci(x):
    text=f"{100*x['mean']:+.3f} pp (n={x['n_seeds']}; CI="+('未计算' if x['ci95'] is None else ', '.join(f'{100*v:+.3f}' for v in x['ci95']))+')'
    if x['ci95'] is not None and x['ci95'][0]==x['ci95'][1]:text+=' [观测种子差相同，区间退化；不表示总体无不确定性]'
    return text

def build():
    final=(ROOT/'confirmation_core_verification.json').exists() and read(ROOT/'confirmation_core_verification.json')['passed']
    lines=['2026-11-12 宽骨干、专家与局部读出整合结果',f'生成时间UTC：{stamp()}',
           '状态：确认已评分，交付状态另见completion_audit.json' if final else '状态：开发阶段草稿，确认尚未完成，不能作为最终交付',
           '', '一、问题与比较口径',
           'W32提供共享强表征；A自然专家、B标签正负边际匹配专家、C组合加权专家与none/global/segment读出构成完整对照。',
           '共同fusion_fit和同预算三分数仿射融合；global/segment保护版采用相同六标签约束。每划分19种决策×5种子=95，共190。',
           'B匹配C的正负目标边际，不等于标签完全均衡；C-B检验在匹配这些边际之后仍存在的目标分配差异。',
           '分段与全局小头均PCA128、隐藏32；复用既定检查点，含epoch0回退，不因本轮开发结果重新挑选历史头。',
           '两者匹配PCA输出维数和MLP容量；PCA输入分别为3584和512维，因此投影存储及计算不同，计入缓冲与推理成本。',
           'Exact要求六个标签全部正确；pp为百分点。所有种子t区间均未作多重比较校正，并且条件于各自固定数据划分。',
           '少量种子恰好得到相同计数差时，t区间可退化为一个点；这不表示真实训练方差或评价身份不确定性为0，不能据此宣布稳定显著。',
           '五种子共享评价身份；物理四视图共享原始身份，不能扩大独立样本数。','']
    lines += ['适用范围：六标签[FMB,AMA,SawFM,SinFM,MFT,ISRJ]，63个非空组合，合成I/Q长度3000，原始分量JNR为10—30dB。',
              '开发和确认均按63组合均衡取样（每组合64身份），不是按训练长尾频率抽样；训练频率加权Exact仅作为同一评价池的附加估计。','']
    primary={tuple(p) for p in CONFIG['primary_comparisons']}
    for role,title in [('development','二、开发结果'),('confirmation','三、新身份冻结确认')]:
        lines += [title]
        for split in CONFIG['splits']:
            path=ROOT/'analysis'/f'{split}_{role}.json'
            if not path.exists():lines.append(split+'：未完成');continue
            r=read(path);lines += [f"{split}：{r['identities']}身份，种子{r['seeds']}",'方法 | Exact% | 固定0.5 Exact% | raw BCE | macro AP% | ECE']
            for method,values in r['means'].items():lines.append(f"{method} | {pct(values['f1_exact'])} | {pct(values['fixed05_exact'])} | {values['BCE']:.6f} | {pct(values['macro_AP'])} | {values['ECE']:.6f}")
            chain=['reference','natural_none','marginal_none','combo_none','combo_segment']
            lines.append('固定展示顺序下的均值增量（不作为独立因果分解）：'+ '；'.join(f"{b}→{a} {100*(r['means'][a]['f1_exact']-r['means'][b]['f1_exact']):+.3f}pp" for b,a in zip(chain,chain[1:])))
            lines.append('其中reference→natural_none同时增加第二骨干和fusion_fit监督；新增单模型仿射控制只检验简单后处理，充分更新骨干与同总计算控制仍待完成。')
            lines += ['主要成对比较：']
            for row in r['comparisons']:
                if (row['candidate'],row['reference']) in primary or (row['candidate'] in ['combo_segment_guarded','marginal_segment_guarded'] and row['reference'].endswith('_none')):
                    lines.append(f"{row['candidate']} − {row['reference']}：{ci(row['exact_difference'])}；正向{row['positive_seeds']}/5；六标签平均保护{row['mean_guards_all_six']}，逐种子通过{row['per_seed_guard_passes']}/5；平均纠正{row['corrected_mean']:.1f}/新增错误{row['new_errors_mean']:.1f}。")
                    lines.append('  FPR差pp：'+', '.join(f'{l}:{100*v:+.3f}' for l,v in zip(LABELS,row['mean_fpr_difference'])))
                    lines.append('  弱召回差pp：'+', '.join(f'{l}:{100*v:+.3f}' for l,v in zip(LABELS,row['mean_weak_recall_difference'])))
            for row in r['interactions']:lines.append(row['contrast']+'：'+ci(row['exact_difference']))
            lines.append('')
    lines += ['四、同身份物理诊断','每个可行身份选择一个目标，固定原始波形背景/噪声，构造0/10/15 dB及移除目标。该子集受可行功率约束，不代表全部自然身份。',
              '正常推理对每个视图分别做复数RMS归一化，因此移除目标也会改变归一化尺度；该诊断衡量整个系统的响应，不单独证明某一特征只利用了目标形态。']
    for role in ['development','confirmation']:
        for split in CONFIG['splits']:
            path=ROOT/split/('physical_'+role+'_results')/'summary.json'
            if not path.exists():continue
            r=read(path);lines.append(f"{split}/{role}：{r['shared_identities']}配对身份；目标数{r['target_counts']}")
            for row in r['comparisons']:
                lines.append(f"{row['candidate']} − {row['reference']}：gap15召回{ci(row['gap15_TPR_difference'])}；移除目标误报{ci(row['absent_FPR_difference'])}。")
    diagnostic=ROOT/'analysis/readout_diagnostics_development.json'
    if diagnostic.exists():
        r=read(diagnostic);lines.append(f"共{r['total_stages']}个冻结读出阶段中，{r['selected_epoch_zero_stages']}个选择epoch0。零残差阶段保留，不按表现删除种子。")
    lines += ['受控配对AUC/AP不是自然单次推理性能，局部统计读出加物理诊断也不等于物理约束损失。',
              '当前小头实际更新SawFM/MFT/ISRJ三个目标；六标签诊断和保护不等于六标签均训练了局部残差。',
              '', '附加诊断：读出收益所在的组合背景',
              '按实际历史fit组合计数分尾部≤50、中部51—249、头部≥250。保留全部63组合和六标签的其他五标签背景切片；这些是观察性分组，不等于同身份物理干预。']
    for role in ['development','confirmation']:
        path=ROOT/'analysis'/f'background_diagnostics_{role}.json'
        if not path.exists():continue
        for split in read(path)['splits']:
            for row in split['comparisons']:
                if (row['candidate'],row['reference']) not in [('combo_segment','combo_none'),('combo_segment','combo_global'),('combo_none','marginal_none')]:continue
                parts=[]
                for name,value in row['groups'].items():
                    if not value['n']:parts.append(name+'：无样本');continue
                    v=value['difference'];parts.append(f"{name} n={value['n']}，Exact差={100*v['mean']:+.3f}pp，未校正种子CI=[{100*v['ci95'][0]:+.3f},{100*v['ci95'][1]:+.3f}]")
                lines.append(f"{role}/{split['split']} {row['candidate']}-{row['reference']}："+'；'.join(parts))
    lines += ['若头部和尾部均获益，不能仅据全体Exact提升宣称局部读出专门解决尾部组合；完整背景结果用于下一阶段提出假设，不按小切片显著性选择模型。','', '五、双长尾矩阵',
              '本轮矩阵训练并比较单骨干A/B/C专家目标，没有为每个新配额重训独立主W32和局部读出；不能据此声称读出增益已在整个IF/CF矩阵得到验证。']
    lines.append('权重ESS描述目标权重的集中程度；实际每轮完整访问8000个fit身份一次，共40轮，未采用有放回重采样。')
    for role in ['development','confirmation']:
        path=ROOT/'analysis'/f'matrix_{role}.json'
        if not path.exists():continue
        r=read(path);lines.append(f"{role}完整模型数：{r['complete_models']}；只统计已验证的A/B/C三臂。")
        for cell in r['cells']:
            lines.append(f"{cell['split']}/{cell['cell']}：种子{cell['seeds']}，CF={cell['combination_IF']:.4f}，IF={cell['label_IF']:.4f}；ESS={cell['objective_combo_ESS']:.1f}，有效组合数={cell['effective_combination_count']:.2f}。")
            lines.append('  自然/边际/组合 Exact%：'+', '.join(f"{e}={pct(cell['means'][e]['f1_exact']['mean'])}" for e in ['natural','marginal','combo']))
        for row in r['comparisons']:
            lines.append(f"{row['split']}/{row['cell']} {row['candidate']}-{row['reference']} Exact：{mci(row['differences']['f1_exact'])}；当前配额尾部Exact：{mci(row['differences']['quota_tail_exact'])}；当前配额头部Exact：{mci(row['differences']['quota_head_exact'])}；按训练频率重加权Exact：{mci(row['differences']['fit_prior_weighted_exact'])}。")
            lines.append(f"  排序/概率辅助指标：macro AP差={mci(row['differences']['macro_AP'])}；raw BCE均差={row['differences']['BCE']['mean']:+.6f}；ECE均差={row['differences']['ECE']['mean']:+.6f}。")
        lines.append('边界：'+'；'.join(r['boundaries']))
    lines += ['既定CF变化在近IF1时会重排很多头尾组合，不能把跨条件差异全部归因于CF标量。',
              '额外整数优化已找到保持六标签边际和基数的最小L1新配额，CF16、总变差0.029625；该新配额未训练。详见QUOTA_REFINEMENT_RESULT.txt。','',
              '附加诊断：相同校准误报目标',
              '每标签仅用calibration负例选择经验FPR≤5%的最低logit阈值，全部A/B/C使用同一规则；不修改主F1阈值。评价实际FPR可能偏离5%，不声称测试FPR完全匹配。']
    for role in ['development','confirmation']:
        path=ROOT/'analysis'/f'matrix_operating_points_{role}.json'
        if not path.exists():continue
        op=read(path)
        for row in op['comparisons']:
            if row['candidate']!='combo' or row['reference']!='marginal':continue
            lines.append(f"{role}/{row['split']}/{row['cell']} C-B附加Exact：{mci(row['exact_difference'])}")
            for label in row['labels']:
                levels={e:[r['labels'][LABELS.index(label['label'])] for r in op['records'] if r['split']==row['split'] and r['cell']==row['cell'] and r['expert']==e] for e in ['marginal','combo']}
                level_text='；'.join(f"{e}实际FPR={100*np.mean([r['FPR'] for r in values]):.3f}%,弱召回={100*np.mean([r['weak_recall'] for r in values]):.3f}%" for e,values in levels.items())
                lines.append('  '+label['label']+'：'+level_text+'；弱AP差='+mci(label['weak_vs_negative_AP']))
    path=ROOT/'analysis/near_unity_development.json'
    if path.exists():
        unity=read(path);lines += ['', '附加诊断：近单位权重与训练轨迹',
            '全部已完成近IF1配对；检查初始化、顺序、弱视图呈现、预算及BCE选轮规则。没有新增精确全1的A/A训练，不能分离微小目标扰动与数值运行差异。']
        for row in unity['records']:
            lines.append(f"{row['split']}/{row['cell']}/{row['seed']} 最大|权重-1|={row['max_absolute_weight_minus_one']:.8f}；BCE选中轮={row['best_BCE_epochs']}；B-A开发Exact={100*row['development_B_minus_A']['f1_exact']:+.3f}pp；固定0.5={100*row['development_B_minus_A']['fixed05_exact']:+.3f}pp。")
        lines.append('同一选中轮也可能出现决策差，因此不同选轮不能独自解释全部差异；后续按04G执行精确全1重复、近单位B及共同末轮控制。')
    lines += ['', '附加对照：同一冻结单模型使用相同fusion_fit监督',
              'reference_affine只用主模型logit，每标签一个尺度和偏置，重用4032个fusion_fit身份、300轮及calibration F1规则；两划分五种子共10项，原190项核心决策不变。',
              '该对照控制额外监督访问和简单仿射后处理，不控制完整系统计算，也不代表单骨干已经充分优化。']
    for role in ['development','confirmation']:
        path=ROOT/'analysis'/f'supervision_control_{role}.json'
        if not path.exists():continue
        for split in read(path)['splits']:
            metrics=split['mean_metrics'];positive=all(v>0 for values in split['scales_by_seed'].values() for v in values)
            lines.append(f"{role}/{split['split']} reference_affine：Exact={pct(metrics['f1_exact'])}%，raw BCE={metrics['BCE']:.6f}，ECE={metrics['ECE']:.6f}；所有拟合尺度为正={positive}。")
            for row in split['comparisons']:
                v=row['differences']['f1_exact'];lines.append(f"  {row['candidate']}-{row['reference']} Exact：{ci([v['mean'],*v['ci95']])}；BCE均差={row['differences']['BCE']['mean']:+.6f}。")
                if row['candidate']=='combo_segment':
                    lines.append('  六标签FPR均差pp：'+', '.join(f"{r['label']}:{100*r['FPR']['mean']:+.3f}" for r in row['labels']))
                    lines.append('  六标签弱召回均差pp：'+', '.join(f"{r['label']}:{100*r['weak_recall']['mean']:+.3f}" for r in row['labels']))
    lines += ['', '六、监督、训练与计算成本',
              '主整合每个历史骨干8000 fit、40轮、10000次更新、累计320000身份呈现；两个骨干复用相同fit身份，模型训练计算相加。',
              '小头每阶段24轮、6000次更新、192000呈现；两阶段目标分别为SawFM/MFT及ISRJ。PCA仅在fit拟合。',
              '融合额外使用4032个有标签fusion_fit身份，每标签300次全批更新；阈值使用1008 calibration。early1008用于选择。',
              'M训练覆盖SawFM/MFT/ISRJ的10/15 dB视图。矩阵每个A/B/C三臂的初始化、顺序和增强呈现逐项匹配。',
              '当前整合增量与同样使用fusion_fit的对照比较，并新增单模型同fusion_fit仿射控制；仍未完成允许充分更新骨干的同监督强单模型与同总计算比较。']
    path=ROOT/'budget_audit.json'
    if path.exists():
        budget=read(path);models=[model for s in budget['matrix'] for c in s['cells'] for model in c['models']]
        lines.append(f"纳入完整三臂对照的新增骨干{budget['new_complete_backbones']}个，对应更新{budget['new_backbone_updates']}，累计身份呈现{budget['new_backbone_identity_presentations']}。")
        lines.append(f"所有已结束训练骨干{budget['all_finished_backbones']}个，尚无完成收据的训练目录{len(budget['unfinished_training_directories'])}个；按完成收据和完整epoch历史观察到的训练更新下界={budget['observed_training_updates_lower_bound']}。目录状态不代替进程存活检查。")
        lines.append(f"新增骨干中best BCE出现在36—40轮的有{sum(m['best_BCE_epoch']>=36 for m in models)}/{len(models)}；40轮是共同预算，不构成所有条件已充分收敛的证明。")
        control=budget['supervision_control'];lines.append(f"额外监督控制：{control['decisions']}个单骨干仿射决策、{control['optimizer_steps']}次小模型更新，未新增骨干；每决策12个有效参数、24个分配参数，不新增相对主整合的监督身份并集。")
        for s in budget['matrix']:lines.append(f"{s['split']}四个设计配额fit身份并集={s['fit_identity_union_across_all_four_designed_cells']}，不能将四次8000或种子重复简单计作独立身份。")
    path=ROOT/'runtime_benchmark.json'
    if path.exists():
        r=read(path);lines.append('计时设备：'+r['device']+'；'+r['scope'])
        for row in r['records']:lines.append(f"{row['method']} batch={row['batch_size']}：神经网络参数{row['parameters']}，另有仿射系数{row['fitted_affine_coefficients']}，缓冲元素{row['buffer_elements']}（含仿射/阈值/PCA），中位批延迟{row['median_batch_ms']:.3f}ms，P95={row['p95_batch_ms']:.3f}ms，PyTorch峰值allocated={row['peak_allocated_bytes']/1024**2:.2f}MiB。")
    else:lines.append('独占GPU推理计时尚未完成。')
    lines += ['', '七、复核与交付状态']
    lines.append('数值复算覆盖全部评价身份；普通前向和迁移验证每划分使用64个覆盖63组合的身份，全部冻结决策做GPU复放、7个代表做CPU复放。交接ZIP附这两套角色的复放输入和预期logit，完整训练池外置。')
    path=ROOT/'identity_registry/registry.json'
    if path.exists():
        registry=read(path);lines.append(f"实际保存的分量种子审计：{registry['pool_count']}个已知原始池、逐池合计{registry['identity_rows_scanned']}行、{registry['stored_components_checked']}个分量；全部与记录的生成规则一致。逐池行数不等于唯一身份数，此检查不构成波形等价验证。旧九类原始池只参与种子隔离审计，不纳入本轮六标签模型训练或评分。")
    path=ROOT/'confirmation_stored_identity_verification.json'
    if path.exists():
        stored=read(path);lines.append(f"新确认实际种子隔离：passed={stored['passed']}；分别对{stored['old_pool_count']}个已登记池及另一套新确认检查sample/noise/同物理类型component交集。")
    for role in ['development','confirmation']:
        for kind in ['core','physical','portable','supervision']:
            p=ROOT/(role+'_'+kind+'_verification.json')
            if p.exists():r=read(p);lines.append(f"{role}/{kind}：passed={r['passed']}，证据={p.name}")
    lines += ['最终冻结清单、交付manifest及远端发布收据分别说明：哪些候选固定、哪些文件已交付、哪些文件已上传。',
              'MAT/PTH/NPZ保存在外部目录，Git内保存源码、协议、轻量收据和分析，不把缺失的大文件包装成仓库自带数据。','',
              '八、长期路线与双方分工',
              '本侧负责统一数据、宽骨干、匹配训练、确认和交付；师弟优先确认读出实现、审阅物理失败样本及解释局部统计，分工为待双方汇总的建议。',
              '下一步顺序：先补足矩阵匹配种子并采用最小改动CF配额，再轮换头尾标签，随后做同监督/同计算强单模型控制，最后扩展功率长尾。',
              '阶段Goal均已保存，详见GOAL_PROMPT.txt、01—07以及03A/03B/03C/04A/04B/04C/04D/04E/05A。具体晋级决定以最终人工阅读分析为准，不由分数排名自动选方法。']
    output='\n'.join(lines)+'\n';(HERE/'RESULTS.txt').write_text(output,encoding='utf-8')
    page='<!doctype html><meta charset="utf-8"><title>2026-11-12 实验结果</title><style>body{max-width:1100px;margin:32px auto;padding:0 24px;font:16px/1.7 system-ui;color:#172331;background:#f7f9fc}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:inherit;background:white;padding:24px;border:1px solid #dce3ed;border-radius:8px}</style><pre>'+html.escape(output)+'</pre>'
    (HERE/'RESULTS.html').write_text(page,encoding='utf-8');print('REPORT_BUILT','confirmation' if final else 'development_draft',flush=True)

if __name__=='__main__':build()
