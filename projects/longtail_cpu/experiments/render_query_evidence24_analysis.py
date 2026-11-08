"""Render completed evidence-source comparison; no model or decision changes."""
import argparse
import base64
from html import escape
import json
from pathlib import Path
from statistics import mean
import numpy as np

ARMS=['baseline','single','fused','uniform']
NAMES={'baseline':'DFMS基线','single':'最深层查询','fused':'融合后特征查询','uniform':'均匀汇聚容量对照'}


def read(path):return json.loads(Path(path).read_text())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',required=True,type=Path);args=parser.parse_args();out=args.report.resolve()
    r=read(out/'results.json');a=read(out/'analysis.json');suite=Path(r['suite'])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font_manager.fontManager.addfont('/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf')
    plt.rcParams.update({'font.family':['DejaVu Sans','Droid Sans Fallback'],'axes.unicode_minus':False,
                         'axes.spines.top':False,'axes.spines.right':False,'font.size':10})
    colors=['#65758b','#4391a0','#d46a45','#8d72a9']
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for arm,color in zip(ARMS,colors):
        for seed,style in [(2050,'-'),(2051,'--')]:
            h=a['histories'][f'{arm}_seed{seed}'];x=[v['epoch'] for v in h]
            axes[0].plot(x,[v['loss'] for v in h],color=color,ls=style,label=f'{NAMES[arm]} {seed}')
            axes[1].plot(x,[v['check_weak_aupr'] for v in h],color=color,ls=style)
    axes[0].set_xlabel('epoch');axes[0].set_ylabel('训练 BCE');axes[0].legend(fontsize=7,ncol=2)
    axes[1].set_xlabel('epoch');axes[1].set_ylabel('check 弱 Macro-AUPR')
    fig.suptitle('统一24轮cosine日程；实线2050，虚线2051')
    fig.savefig(out/'training_analysis.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for arm,color in zip(ARMS,colors):
        powers=[v for v in r['power_views'] if v['arm']==arm and v['role']=='development_test']
        gaps=[0,6,12,18,24]
        exact=[mean(v['exact'] for v in powers if v['gap_db']==g)*100 for g in gaps]
        recall=[mean(v['weak_recall'] for v in powers if v['gap_db']==g)*100 for g in gaps[2:]]
        axes[0].plot(gaps,exact,'o-',color=color,label=NAMES[arm])
        axes[1].plot(gaps[2:],recall,'o-',color=color,label=NAMES[arm])
    axes[0].set_xlabel('配对视图功率差 dB');axes[0].set_ylabel('开发报告集 Exact %');axes[0].set_xticks([0,6,12,18,24]);axes[0].legend(fontsize=8)
    axes[1].set_xlabel('配对视图功率差 dB');axes[1].set_ylabel('弱 Macro-Recall %');axes[1].set_xticks([12,18,24]);axes[1].legend(fontsize=8)
    fig.suptitle('同一冻结模型和F1阈值；双种子均值；开发报告集已消费')
    fig.savefig(out/'power_analysis.png',dpi=170);plt.close(fig)
    def table(headers,rows):
        return '<table><thead><tr>'+''.join('<th>'+escape(str(v))+'</th>' for v in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+escape(str(v))+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table>'
    def pic(name):return '<img alt="结果分析图" src="data:image/png;base64,'+base64.b64encode((out/name).read_bytes()).decode()+'">'
    def stat(role,arm,key,factor=100):
        x=r['aggregate'][role][arm][key];return f"{x['mean']*factor:.3f} ± {x['std']*factor:.3f}"
    text='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>24轮对照：完成进度与分析</title><style>
body{font:16px/1.75 system-ui,sans-serif;background:#f3f6f8;color:#234}main{max-width:1150px;margin:28px auto;background:white;padding:32px}h1,h2{color:#155e75}table{border-collapse:collapse;width:100%;table-layout:fixed;font-size:14px;margin:18px 0}td,th{border:1px solid #d8e2e8;padding:8px;overflow-wrap:anywhere;vertical-align:top}th{background:#155e75;color:white}.note{background:#edf5f6;border-left:5px solid #197c89;padding:16px}img{width:100%}a{color:#16728c}code{overflow-wrap:anywhere}</style><main>'''
    text+='<h1>24轮对照：完成进度与结果分析</h1>'
    text+=f"<p>北京时间2026-11-02 19:46开始，20:47:12全部完成，总耗时{a['pipeline']['wall_seconds']/60:.2f}分钟。8/8作业、192/192训练轮完成，无失败；104份主测评产物、1%/5%误报目标及check功率视图补充测评均已完成并核验。</p>"
    text+='<p class="note">最深层查询显示较一致的整体收益；融合后查询在check上最优，但开发报告集上的收益主要集中于等功率，默认阈值下的大功率差弱召回更低。融合后查询仍是候选，当前证据不足以认定它优于原最深层查询。两个数据角色均属于已消费开发数据。</p>'
    for role in ['check','development_test']:
        text+='<h2>'+role+('（1008个身份）' if role=='check' else '（4032个身份，开发报告集）')+'</h2>'
        text+=table(['模型','Exact %','Macro-F1 %','弱AUPR','弱宏召回 %','平均FPR %'],[[NAMES[arm],stat(role,arm,'exact'),stat(role,arm,'macro_f1'),stat(role,arm,'weak_aupr',1),stat(role,arm,'weak_recall'),stat(role,arm,'mean_fpr')] for arm in ARMS])
    text+='<p>±是两个种子的样本标准差。三种新增分类头模型均为300,211参数，DFMS为257,511参数；F1阈值均只在calibration拟合。</p>'
    text+='<h2>训练预算带来了什么</h2>'+pic('training_analysis.png')
    text+=table(['模型','最佳轮（2050/2051）','本轮前8轮最佳check弱AUPR均值','本轮24轮最佳check弱AUPR均值','单作业训练分钟均值'],[
        [NAMES[arm],'/'.join(str(j['best_epoch']) for j in a['jobs'] if j['arm']==arm),
         f"{mean(j['early8_best_check_weak_aupr'] for j in a['jobs'] if j['arm']==arm):.5f}",
         f"{mean(j['full24_best_check_weak_aupr'] for j in a['jobs'] if j['arm']==arm):.5f}",
         f"{mean(j['train_minutes'] for j in a['jobs'] if j['arm']==arm):.2f}"] for arm in ARMS])
    text+='<p>在本轮同一日程内，第9–24轮为所有模型继续提供明显增益；多数最佳轮在21–24轮，融合2051在18轮最佳。旧8轮实验使用常数学习率，本轮为24轮cosine，跨轮成绩不能把轮数与日程效应分开，也不能把共同训练收益全部归于架构。</p>'
    text+='<h2>功率差下的收益与退化</h2>'+pic('power_analysis.png')
    text+=table(['功率差 dB']+[NAMES[arm]+' Exact %' for arm in ARMS],[
        [g]+[f"{mean(v['exact'] for v in r['power_views'] if v['arm']==arm and v['role']=='development_test' and v['gap_db']==g)*100:.3f}" for arm in ARMS] for g in [0,6,12,18,24]])
    text+='<p>融合后查询在0 dB达到91.58%，原最深层为85.89%；24 dB时则为14.35%，原最深层16.51%，基线16.01%。24 dB弱召回分别约5.70%、8.64%、7.79%。这支持“融合后证据对部分场景有价值”，尚不支持解决极弱分量问题。</p>'
    text+='<h2>误报约束工作点</h2>'
    text+=table(['模型','校准FPR目标','报告集实际平均FPR %','弱宏召回 %','Exact %'],[
        [NAMES[arm],target,f"{v['actual_mean_fpr']*100:.3f}",f"{v['weak_macro_recall']*100:.3f}",f"{v['exact']*100:.3f}"]
        for arm in ARMS for target in ['fpr_0.01','fpr_0.05'] for v in [a['fpr_aggregate']['development_test'][arm][target]]])
    text+='<p>默认F1阈值下，融合后查询报告集弱召回17.52%、平均FPR1.75%，最深层查询为21.87%、3.04%。校准FPR目标5%时分别变为27.19%、5.13%与28.88%、5.42%。实际FPR不完全相等，这些数字只描述校准规则下的工作点，不能当作严格等FPR的优劣证明；没有在报告集重新拟合阈值。</p>'
    text+='<h2>配对差值与种子一致性</h2>'
    text+=table(['比较（报告集）','平均Exact差 pp','2050 / 2051差 pp','身份重采样条件95%区间 pp'],[
        [name,f"{v['mean_delta']*100:+.3f}",' / '.join(f'{x*100:+.3f}' for x in v['per_seed_delta']),
         ' / '.join(f'{x*100:+.3f}' for x in v['identity_conditional_95_ci'])] for name,v in a['paired_exact']['development_test'].items()])
    text+='<p>区间按组合分层、基础身份成对重采样2000次，并对两个已训练种子平均；只反映给定这些模型的样本不确定性。最深层相对基线和均匀汇聚对照，两种子均为正向；融合相对基线有一个种子Exact略降。不能据条件区间宣称训练随机性或新身份泛化已经验证。</p>'
    text+='<h2>逐类收益与注意力机制</h2>'
    labels=['FMB','AMA','SawFM','SinFM','MFT','ISRJ'];per={}
    for arm in ARMS:
        per[arm]=[read(suite/f'{arm}_seed{seed}/evaluation/development_test/metrics.json')['metrics']['calibrated']['weak_component'] for seed in [2050,2051]]
    text+=table(['类别']+[NAMES[arm]+' 弱AUPR' for arm in ARMS],[[label]+[f"{mean(v['per_label'][label]['weak_aupr'] for v in per[arm]):.5f}" for arm in ARMS] for label in labels])
    text+='<p>融合后特征在FMB和MFT上有收益，SinFM与ISRJ未表现出相对最深层的优势。尾类弱召回（AMA/SinFM）为融合33.26%、最深层43.36%、基线38.05%，仍需重点解决。查询注意力发生了变化：融合模型SinFM/MFT的四头平均注意力熵更低，但聚焦本身不能证明弱成分识别更好。</p>'
    text+='<h2>本轮决策与下一步</h2><p>保留最深层查询作为参考候选，融合后查询作为待确认候选；当前不采用融合替代作为已确认提升。先固定候选、选择指标与calibration误报工作点，在新身份上确认功率分层和逐类收益，并补充更多训练种子。后续结构研究应面向大功率差下的弱成分可分离证据及训练功率覆盖。已消费的development-test不继续用于调参，本次不追加训练或阈值搜索。</p>'
    text+='<p>核验了8个权重、68份冻结源文件、角色身份隔离、冻结阈值与预测元数据配对。训练进度文件已由完成标记校正为全部完成，保留了最后一次轮询的原记录。</p>'
    text+='<p><a href="report.html">自动汇总</a> · <a href="results.csv">逐次结果CSV</a> · <a href="analysis.json">统计与配对分析JSON</a> · <a href="results.json">全部结果JSON</a></p></main></html>'
    (out/'analysis.html').write_text(text)
    print(out/'analysis.html')


if __name__=='__main__':main()
