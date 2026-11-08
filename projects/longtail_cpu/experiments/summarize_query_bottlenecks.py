"""Summarize frozen check diagnostics without selecting training settings."""
import argparse
import base64
from datetime import datetime
from html import escape
import json
from pathlib import Path
from statistics import mean
from zoneinfo import ZoneInfo
import numpy as np

ARMS=['baseline','single','capacity','multiscale']
NAMES={'baseline':'DFMS基线','single':'最深层查询','capacity':'最深层容量对照','multiscale':'多层查询'}
CLASSES=['FMB','AMA','SawFM','SinFM','MFT','ISRJ']


def read(path):return json.loads(Path(path).read_text())


def threshold_bootstrap(z,replicates=1000):
    """Resample calibration IDs within each combination; never use check labels."""
    y=z['targets'].astype(bool);p=1/(1+np.exp(-np.clip(z['logits'].astype(float),-40,40)))
    code=y.astype(np.int64)@(1<<np.arange(6));rng=np.random.default_rng(20261104)
    index=np.concatenate([rng.choice(g,(replicates,len(g)),replace=True) for c in np.unique(code)
                          for g in [np.flatnonzero(code==c)]],axis=1)
    grid=np.linspace(.05,.95,181);chosen=[]
    for c in range(6):
        bins=np.searchsorted(grid,p[:,c],side='right')
        encoded=bins[index]+182*np.arange(replicates)[:,None]
        hist=np.bincount(encoded.ravel(),minlength=182*replicates).reshape(replicates,182)
        positive=np.bincount(encoded.ravel(),weights=y[index,c].ravel(),minlength=182*replicates).reshape(replicates,182)
        predicted=np.cumsum(hist[:,::-1],axis=1)[:,::-1][:,1:]
        tp=np.cumsum(positive[:,::-1],axis=1)[:,::-1][:,1:]
        denom=positive.sum(1)[:,None]+predicted
        f1=np.divide(2*tp,denom,out=np.zeros_like(tp),where=denom>0)
        chosen.append(grid[f1.argmax(1)])
    chosen=np.asarray(chosen).T
    return {'role':'calibration','replicates':replicates,'thresholds_refitted_for_deployment':False,
            'threshold_2p5_median_97p5':np.quantile(chosen,[.025,.5,.975],axis=0).T.tolist()}


def analyze(suite,diagnostics,out):
    import h5py
    config=read(suite/'baseline_seed2050/config.json');root=Path(config['data']['manifest']).parent
    with h5py.File(root/'train.h5','r') as f:
        y=np.asarray(f['label_selected']).T.astype(bool);weak=np.asarray(f['weak_component_mask']).T.astype(bool)
        gap=-np.asarray(f['relative_power_dB']).T;jnr=np.asarray(f['component_effective_JNR_dB']).T
    code=y.astype(np.int64)@(1<<np.arange(6));coverage=[]
    for c,label in enumerate(CLASSES):
        counts=[int((weak[:,c]&(code==k)&np.isclose(gap[:,c],g)).sum())
                for k in range(1,64) if k&(1<<c) and int(k).bit_count()>1 for g in (12,18,24)]
        coverage.append({'label':label,'cells':len(counts),'empty':int((np.asarray(counts)==0).sum()),
                         'minimum':min(counts),'median':float(np.median(counts)),
                         'mean':float(np.mean(counts)),'maximum':max(counts)})
    training={'samples':len(y),'positive_fraction':float(y.mean()),'weak_decision_fraction':float(weak.mean()),
              'label_counts':y.sum(0).tolist(),'weak_counts':weak.sum(0).tolist(),
              'negative_counts':(~y).sum(0).tolist(),'weak_jnr_range_db':[float(jnr[weak].min()),float(jnr[weak].max())],
              'weak_per_batch32':(weak.mean(0)*32).tolist(),'weak_context_gap_coverage':coverage}
    jobs={};errors={};calibration={};history={}
    for arm in ARMS:
        for seed in (2050,2051):
            job=f'{arm}_seed{seed}';directory=suite/job
            d=read(diagnostics/job/'diagnostic.json');jobs[job]=d
            z=dict(np.load(diagnostics/job/'check_actual.npz'));truth=z['targets'].astype(bool);w=z['weak_mask'].astype(bool)
            p=1/(1+np.exp(-np.clip(z['logits'].astype(float),-40,40)))>=d['thresholds']
            error=p!=truth;wf=error&w;sf=error&truth&~w;fp=error&~truth;bad=error.any(1)
            errors[job]={'label_errors':int(error.sum()),'weak_false_negatives':int(wf.sum()),
                         'strong_false_negatives':int(sf.sum()),'false_positives':int(fp.sum()),
                         'wrong_samples':int(bad.sum()),'only_weak_miss_samples':int((wf.any(1)&~sf.any(1)&~fp.any(1)).sum()),
                         'only_weak_miss_fraction_of_wrong_samples':float((wf.any(1)&~sf.any(1)&~fp.any(1)).sum()/bad.sum()),
                         'strong_recall_micro':float(p[truth&~w].mean()),'weak_recall_micro':float(p[w].mean()),
                         'oracle_fix_weak_only_exact':float((~(sf|fp).any(1)).mean())}
            calibration[job]=threshold_bootstrap(dict(np.load(directory/'evaluation/calibration/predictions.npz')))
            history[job]=[{k:r[k] for k in ('epoch','loss')}|{
                'check_weak_aupr':r['validation']['weak_component']['weak_macro_aupr']} for r in read(directory/'history.json')]
    result={'created':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds'),
            'suite':str(suite),'diagnostics':str(diagnostics),'training':training,'jobs':jobs,'check_errors':errors,
            'calibration_bootstrap':calibration,'history':history,
            'roles':{'new_inference':'check (1008) and fixed fit subset (1024)',
                     'threshold_bootstrap':'calibration (1008), diagnostic only','development_test_used':False},
            'limitations':['Only two trained seeds; frozen changes diagnose dependence, not independently trained gains.',
                           'Fit and check class priors differ; raw AUPR cannot establish overfitting across these splits.',
                           'Per-head averaged statistics describe the saved final checkpoint, not an attention training trajectory.',
                           'Gradient measurement is one fixed fit batch in eval mode and is parameterization dependent.']}
    out.mkdir(parents=True,exist_ok=True)
    (out/'analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
    print(json.dumps({'coverage':coverage,'training_positive_fraction':training['positive_fraction'],
        'calibration_threshold_intervals':{j:v['threshold_2p5_median_97p5'] for j,v in calibration.items()}},ensure_ascii=False))


def render(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font_manager.fontManager.addfont('/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf')
    plt.rcParams.update({'font.family':['DejaVu Sans','Droid Sans Fallback'],'axes.unicode_minus':False,
                         'axes.spines.top':False,'axes.spines.right':False})
    a=read(out/'analysis.json');jobs=a['jobs'];errors=a['check_errors'];colors=['#63758b','#378e9b','#8c71a8','#d36c43']
    def avg(arm,fn):return mean(fn(jobs[f'{arm}_seed{s}']) for s in (2050,2051))
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained');x=np.arange(4)
    bottom=np.zeros(4)
    for key,label,color in [('weak_false_negatives','弱正例漏检','#d36c43'),('strong_false_negatives','强正例漏检','#8c71a8'),('false_positives','缺席类别误报','#63758b')]:
        v=np.array([mean(errors[f'{arm}_seed{s}'][key] for s in (2050,2051)) for arm in ARMS])
        axes[0].bar(x,v,bottom=bottom,label=label,color=color);bottom+=v
    axes[0].set_xticks(x,[NAMES[k] for k in ARMS],rotation=12);axes[0].set_ylabel('标签判断错误数（双种子均值）');axes[0].legend(fontsize=8)
    for arm,color in zip(['single','capacity','multiscale'],colors[1:]):
        variants=['actual','uniform_all'];values=[avg(arm,lambda d:d['outcomes'][v]['exact']*100) for v in variants]
        axes[1].plot([0,1],values,'o-',color=color,label=NAMES[arm])
    axes[1].set_xticks([0,1],['已训练注意力','冻结后全均匀注意力']);axes[1].set_ylabel('check Exact %');axes[1].legend(fontsize=8)
    fig.suptitle('冻结模型诊断：check 1008个身份；未重新训练或拟合阈值')
    fig.savefig(out/'bottlenecks.png',dpi=160);plt.close(fig)
    def table(headers,rows):
        return '<table><tr>'+''.join('<th>'+escape(str(v))+'</th>' for v in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+escape(str(v))+'</td>' for v in row)+'</tr>' for row in rows)+'</table>'
    text='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>类别查询性能瓶颈分析</title><style>
body{font:16px/1.75 system-ui,sans-serif;background:#f3f6f8;color:#223448}main{max-width:1100px;margin:28px auto;background:white;padding:32px}h1,h2{color:#155e75}table{border-collapse:collapse;width:100%;table-layout:fixed;font-size:14px;margin:18px 0}td,th{border:1px solid #d8e2e8;padding:8px;overflow-wrap:anywhere;vertical-align:top}th{background:#155e75;color:white}.note{background:#edf5f6;border-left:5px solid #197c89;padding:16px}img{width:100%}code{overflow-wrap:anywhere}a{color:#16728c}</style><main>'''
    text+='<h1>第一条研究路线：性能提升受到哪些环节阻碍</h1><p>'+escape(a['created'])+'</p>'
    text+='<p class="note">可以继续完善。当前证据优先指向弱成分可分离表征不足，以及多层查询尚未形成有效的类别与尺度选择。普通BCE已经产生显著弱正例损失和梯度，不能将问题简单解释为负例数量多或弱正例没有梯度。所有新推理仅使用check与训练子集，没有在development-test上推理、计算指标或调参，没有更新权重。</p>'
    text+='<p>本次check为1008个身份，原主报告development-test为4032个身份，二者数值应分别解释。每个模型保留原calibration阈值；两个种子的均值不代表完整训练随机性验证。</p>'
    text+='<img alt="错误组成与均匀注意力替换" src="data:image/png;base64,'+base64.b64encode((out/'bottlenecks.png').read_bytes()).decode()+'">'
    text+='<h2>1. 总体错误主要来自弱正例漏检</h2>'
    text+=table(['模型','弱漏检/强漏检/误报数','错误样本中仅弱漏检占比','强正例召回 %','弱正例召回 %'],[
        [NAMES[arm],'/'.join(f"{mean(errors[f'{arm}_seed{s}'][k] for s in (2050,2051)):.1f}" for k in ('weak_false_negatives','strong_false_negatives','false_positives')),
         f"{mean(errors[f'{arm}_seed{s}']['only_weak_miss_fraction_of_wrong_samples'] for s in (2050,2051))*100:.2f}%",
         f"{mean(errors[f'{arm}_seed{s}']['strong_recall_micro'] for s in (2050,2051))*100:.2f}",
         f"{mean(errors[f'{arm}_seed{s}']['weak_recall_micro'] for s in (2050,2051))*100:.2f}"] for arm in ARMS])
    text+='<p>这里的召回按全部对应标签判断汇总（micro），与主报告逐类等权的macro不同。最深层查询减少了误报，同时增加弱漏检；要提高Exact，应优先改善弱成分在负例中的可分辨性，并同步约束误报。</p>'
    text+='<h2>2. 多层查询接近平均汇聚，类别选择较弱</h2>'
    text+=table(['模型','逐头归一化熵（1为均匀）','同一头不同类别的注意力余弦相似度','已训练 / 全均匀 / 层内均匀 Exact %','已训练 / 全均匀 弱AUPR'],[
        [NAMES[arm],f"{avg(arm,lambda d:float(np.mean(d['attention']['head_entropy_by_class']))):.5f}",
         f"{avg(arm,lambda d:float(np.asarray(d['attention']['head_class_cosine'])[:,~np.eye(6,dtype=bool)].mean())):.4f}",
         ' / '.join(f"{avg(arm,lambda d:d['outcomes'][v]['exact']*100):.2f}" if v!='uniform_within_levels' or arm=='multiscale' else '—' for v in ('actual','uniform_all','uniform_within_levels')),
         ' / '.join(f"{avg(arm,lambda d:d['outcomes'][v]['weak_aupr']):.5f}" for v in ('actual','uniform_all'))] for arm in ['single','capacity','multiscale']])
    text+='<p>第2/3/4层有1500/750/375个token。所有token共享一次softmax，打分相近时各层自然得到57.1%/28.6%/14.3%的权重。独立1×1投影与共享LayerNorm没有显式对齐层间语义。冻结层内均匀化保留每个头、类别、样本原有的层总权重；它显示精细时间权重暂未提供稳定附加收益，但不能证明重新训练的均匀池化或层门控一定更好。</p>'
    text+='<p>均匀注意力本身并非错误：FMB、AMA、SawFM、SinFM在仿真中可贯穿整个PRI，不同成分在时间上大量重叠。图像目标的局部区域选择机制不能直接承担混合信号解混。<a href="https://arxiv.org/abs/2107.10834">Query2Label</a>提供的是按类读取特征的机制，其雷达收益仍取决于输入特征是否已经分离出类别证据。</p>'
    text+='<h2>3. 弱正例有明显优化压力，瓶颈不只是损失权重</h2>'
    text+=table(['模型','弱正例占标签判断 %','弱正例占BCE损失 %','弱正例占绝对logit梯度 %','弱正例平均BCE'],[
        [NAMES[arm]]+[f"{avg(arm,lambda d:d['fit_loss_groups']['weak_positive'][key]*factor):.2f}" for key,factor in [('decision_fraction',100),('bce_mass_fraction',100),('absolute_logit_gradient_fraction',100),('mean_bce',1)]] for arm in ARMS])
    text+='<p>这些数字来自固定1024个fit身份的eval模式诊断。它们不能代表全部训练过程，但足以排除“弱正例没有产生显著误差信号”的说法。全训练集正标签比例为51.87%，负标签48.13%，不符合图像多标签任务中海量负标签的整体比例。<a href="https://github.com/Alibaba-MIIL/ASL">ASL</a>可作为独立损失对照，不能由名称和图像结果直接推定有效。更有针对性的候选是按类、按强弱组控制损失贡献，并监控固定误报工作点下的弱召回；须防止加权扩大误报。</p>'
    text+='<h2>4. 查询读入的证据与主分支不一致</h2><p>主分支先融合实虚部、幅度和多延迟相位，再做多尺度时序处理；查询分支直接读取复数编码器的实虚部，绕过显式相位/幅度融合和temporal_backbone。这是已经确认的代码事实，但“因此缺失关键证据”仍是假设。优先做最深层查询读取融合后时序特征的单因素对照，以验证时序相位信息是否限制了弱成分识别。当前width=16，原实虚部拼接与融合后特征均为128通道，时间长度均为375，可以沿用128→64投影并保持参数量一致；融合特征已由全局分支计算，无需额外骨干网络。</p>'
    text+='<p>残差系数tanh约0.14–0.21，并不表示查询贡献很小：其残差logit的标准差在多个类别上与全局logit相当或更大。冻结后删除查询分支会明显改变分布；这种诊断不能把共同训练的全局分支当作独立DFMS基线。单批梯度快照中Q/K投影梯度弱于V投影，提示读取规则可能比值映射学习慢，但不足以认定梯度消失；Adam的实际步长还取决于优化器状态。</p>'
    text+='<h2>5. 弱成分覆盖有限，现有仿真适用范围窄</h2>'
    text+=table(['类别','训练正例','弱正例','batch32预期弱正例','弱类别×共现组合×功率档：单元中位数 / 最少 / 最多'],[
        [label,a['training']['label_counts'][i],a['training']['weak_counts'][i],f"{a['training']['weak_per_batch32'][i]:.2f}",
         f"{a['training']['weak_context_gap_coverage'][i]['median']:.1f} / {a['training']['weak_context_gap_coverage'][i]['minimum']} / {a['training']['weak_context_gap_coverage'][i]['maximum']}"] for i,label in enumerate(CLASSES)])
    text+='<p>每类有31种包含其他类别的组合，弱功率档为12/18/24 dB，共93个覆盖单元；计数不是93个独立任务。当前每个训练身份只出现一个功率视图，每个复合样本至多削弱一个成分。已保存的配对原始视图可支持训练身份内的功率视图采样与一致性学习，但复用同一身份的不同视图不增加独立样本数。</p>'
    text+='<p>仿真固定强成分JNR为30 dB，24 dB差时弱成分JNR仍为6 dB。因此现有失败涉及强干扰下的可分离性，不能直接归因于弱分量低于噪声。AMA中心频率固定0且其生成波形为实值，类间统计差异明显；未验证随机全局相位、频偏、接收链变化、多弱成分和更宽JNR时的泛化。这些属于适用范围风险，不能据此认定当前开发集下降的因果来源。</p>'
    text+='<h2>6. 训练预算、选模与阈值存在不同目标</h2><p>8轮内训练BCE持续下降，但多数check弱AUPR在第7轮达到当前最大值，随后回落；单层某些种子在第8轮继续提高。两者并存，尚不能判定一律欠拟合或延长训练必然有效。应以相同24轮预算和预先固定的学习率日程做对照，同时记录强弱组损失、各层梯度和check弱识别曲线。</p>'
    text+='<p>当前按check弱AUPR选checkpoint，按calibration逐类F1选阈值，最终又关注Exact与弱召回。这些指标的最优点不同。弱分量排序改善后仍可能因阈值上升而漏检；降低阈值也可能增加六标签Exact的误报。校准阈值的自助区间如下，只用于估计不稳定性，没有更新主阈值或在check选阈值。</p>'
    text+=table(['种子/模型']+[c+' 阈值2.5% / 中位数 / 97.5%' for c in CLASSES],[
        [job]+[' / '.join(f'{v:.3f}' for v in a['calibration_bootstrap'][job]['threshold_2p5_median_97p5'][i]) for i in range(6)]
        for job in ('single_seed2050','single_seed2051','multiscale_seed2050','multiscale_seed2051')])
    text+='<p>1000次按组合分层重采样calibration身份；区间描述给定模型时校准样本变化引起的阈值波动，不包含训练随机性。训练与check先验不同，原始fit/check弱AUPR差值也不能独立证明过拟合。</p>'
    text+='<h2>下一轮优先顺序与判定条件</h2>'
    text+=table(['优先级','单因素验证','要回答的问题','通过条件'],[
        ['1','原DFMS、最深层查询、均匀读取容量对照；统一24轮预算','收益来自更好的读取规则，还是额外分类容量/优化？','多个种子check弱AUPR及误报约束召回提高，整体Exact不下降；最后使用新身份确认'],
        ['2','最深层查询只替换为融合后时序特征memory','查询是否缺少幅相与延迟结构证据？','大功率差分层收益，且误报在既定约束内'],
        ['3','多层每层独立归一化注意力，再学类别层权重；保留同容量对照','token数量先验与层间语义是否阻碍选择？','超过最深层与容量对照；层权重对样本/类别有可重复变化'],
        ['4','独立比较强弱组损失平衡或训练身份内功率配对一致性','弱正例可分性与跨功率稳定性是否改善？','弱AUPR与既定FPR目标下召回共同改善，不只提高平均F1']])
    text+='<p>逐层独立读取/选择可参考<a href="https://github.com/fundamentalvision/Deformable-DETR">Deformable DETR</a>的多尺度选择思想；雷达时序实现和小模型CPU预算应单独验证，当前没有实验证明该候选会提升。第一轮8作业的24轮训练与测评按已测CPU吞吐建议预留约55–75分钟；新数据生成和五种子确认另计。</p>'
    text+='<p>所有候选结构、训练步数、损失和阈值目标须在新一轮开始前固定；已消费的development-test不继续参与调参，新身份报告集仅做最终确认。</p>'
    text+='<p>可复查文件：<a href="analysis.json">analysis.json</a>。新诊断产物：<code>'+escape(a['diagnostics'])+'</code>。模型权重、主阈值和原实验源代码保持原有哈希。</p></main></html>'
    (out/'report.html').write_text(text)
    print(out/'report.html')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite',type=Path);p.add_argument('--diagnostics',type=Path)
    p.add_argument('--output',required=True,type=Path);p.add_argument('--render',action='store_true')
    args=p.parse_args()
    if args.render:render(args.output.resolve())
    else:analyze(args.suite.resolve(),args.diagnostics.resolve(),args.output.resolve())


if __name__=='__main__':main()
