from pathlib import Path
import json, re, hashlib, math, copy
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from docx import Document
from docx.shared import Inches, Pt, Cm, RGBColor
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.opc.constants import RELATIONSHIP_TYPE as RT
from reportlab.platypus import SimpleDocTemplate, Paragraph as RLParagraph, Spacer, Table as RLTable, TableStyle, Image as RLImage, PageBreak, KeepTogether
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.enums import TA_LEFT
from xml.sax.saxutils import escape

ROOT=Path('/ultrafast_share/xuwang_group/wxwu/longtail')
OUT=ROOT/'reports/2026-11-02_文献调研'
OUT.mkdir(parents=True,exist_ok=True)
SOURCE=Path('/ultrafast_share/xuwang_group/wxwu/.codex/attachments/bf375ebd-bb40-4d35-a42a-b0b7a99bce45/长尾方法综述与试验建议(1).docx')
NAME='长尾方法综述与试验建议_补充调研版_20261102'
RUN=ROOT/'runs/joint_dfms_seed2050_78d6f2949c'
M=json.loads((RUN/'evaluation/development_test/metrics.json').read_text())
S=json.loads((RUN/'split.json').read_text())
Z=np.load(RUN/'evaluation/development_test/predictions.npz')
Y=Z['targets'].astype(bool); W=Z['weak_mask']; P=1/(1+np.exp(-Z['logits'])) >= np.array(M['decision']['thresholds'])
ERR=P!=Y; ORACLE=P.copy(); ORACLE[W]=True
DIAG={
 'date':'2026-11-02','source_docx':str(SOURCE),'source_docx_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
 'checkpoint':str(RUN/'best.pt'),'checkpoint_sha256':M['checkpoint_sha256'],'split_sha256':M['split_sha256'],
 'classification':'Consumed development diagnostics; not a locked test; no retraining or threshold search in this review.',
 'train_positive_labels':sum(S['fit_label_counts']),'train_negative_labels':8000*6-sum(S['fit_label_counts']),
 'train_positive_fraction':sum(S['fit_label_counts'])/(8000*6),'test_samples':len(Y),
 'correct_samples':int((~ERR.any(1)).sum()),'wrong_samples':int(ERR.any(1).sum()),
 'one_bit_error_samples':int((ERR.sum(1)==1).sum()),'one_bit_share_among_errors':float((ERR.sum(1)==1).sum()/ERR.any(1).sum()),
 'fp_only_samples':int(((P&~Y).any(1)&~(~P&Y).any(1)).sum()),
 'fn_only_samples':int(((~P&Y).any(1)&~(P&~Y).any(1)).sum()),
 'both_fp_fn_samples':int(((P&~Y).any(1)&(~P&Y).any(1)).sum()),
 'weak_true_positive':int(P[W].sum()),'weak_false_negative':int((~P[W]).sum()),
 'all_false_negative':int((~P&Y).sum()),'weak_fn_share':float((~P[W]).sum()/(~P&Y).sum()),
 'oracle_fix_weak_fn_only_exact':float((ORACLE==Y).all(1).mean()),
 'oracle_newly_correct_samples':int(((ORACLE==Y).all(1)&ERR.any(1)).sum()),
 'oracle_warning':'Uses ground truth to correct weak false negatives only, with all other bits unchanged. A diagnostic intervention, not an achievable performance forecast or a Bayes upper bound.',
 'metrics':{k:{q:v[q] for q in ['macro_f1','macro_aupr','subset_accuracy','hamming_loss']}|{'weak_component':v['weak_component']} for k,v in M['metrics'].items()},
 'verification_from_prior_inspection':{'tests_passed':34,'tests_seconds':24.57,'checkpoint_cpu_macro_f1':0.8955603096911515,'checkpoint_cpu_exact':0.4437003968253968,'checkpoint_cpu_weak_aupr':0.3031532113925987,'new_training_performed':False}
}
(OUT/'研究证据与诊断.json').write_text(json.dumps(DIAG,ensure_ascii=False,indent=2))

FONT='/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf'
font_manager.fontManager.addfont(FONT)
plt.rcParams.update({'font.family':['DejaVu Sans','Droid Sans Fallback'],'axes.unicode_minus':False,'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(1,3,figsize=(15,4.2),layout='constrained')
cl=S['classes']; cm=M['metrics']['calibrated']; weak=cm['weak_component']
x=np.arange(6);wr=np.array([weak['per_label'][c]['weak_recall'] for c in cl])*100
axs[0].bar(x,wr,color=['#cf5a45' if v<15 else '#197c89' for v in wr]);axs[0].set_xticks(x,cl,rotation=25);axs[0].set_ylim(0,65);axs[0].set_ylabel('召回率（%）');axs[0].set_title('弱分量识别主要由 AMA 支撑')
for a,v in zip(x,wr):axs[0].text(a,v+1.4,f'{v:.1f}',ha='center',fontsize=9)
bins=weak['power_gap_bins'][:5];g=[0,6,12,18,24];v=[b['recall']*100 for b in bins]
axs[1].plot(g,v,'o-',color='#197c89',lw=2);axs[1].set_xticks(g);axs[1].set_ylim(0,115);axs[1].set_xlabel('相对最强分量的功率差（dB）');axs[1].set_title('功率差增大后召回快速下降')
for a,b in zip(g,v):axs[1].annotate(f'{b:.1f}%',(a,b),xytext=(0,8),textcoords='offset points',ha='center',fontsize=9)
counts=[int((ERR.sum(1)==k).sum()) for k in [1,2,3]]
axs[2].bar(['错 1 个标签','错 2 个标签','错 3 个标签'],counts,color=['#197c89','#8dbfc6','#cf5a45']);axs[2].set_ylim(0,2500);axs[2].set_title('94.38% 的错误样本只错一位')
for i,a in enumerate(counts):axs[2].text(i,a+35,str(a),ha='center')
fig.suptitle('当前联合任务：4032 条已消费开发样本，冻结模型与校准阈值',fontsize=14)
chart=OUT/'图1_当前识别瓶颈.png';fig.savefig(chart,dpi=180,facecolor='white');plt.close(fig)

fig,ax=plt.subplots(figsize=(12,4.6));ax.axis('off')
nodes=[(.02,.37,.18,.25,'复数 I/Q\n保持原输入与六维输出'),(.27,.66,.23,.20,'现有全局统计分支\n保留强分量识别能力'),(.27,.11,.23,.25,'类别条件局部证据\n多尺度特征 + 6 个类别查询'),(.58,.37,.18,.25,'轻量融合\n输出六类 logits'),(.83,.37,.15,.25,'冻结阈值\n报告 Exact / 弱识别')]
for xx,yy,ww,hh,txt in nodes:
 ax.add_patch(plt.Rectangle((xx,yy),ww,hh,facecolor='#e9f3f5',edgecolor='#197c89',lw=1.7,joinstyle='round'));ax.text(xx+ww/2,yy+hh/2,txt,ha='center',va='center',fontsize=10)
for st,en in [((.20,.53),(.27,.76)),((.20,.45),(.27,.24)),((.50,.76),(.58,.53)),((.50,.24),(.58,.45)),((.76,.495),(.83,.495))]:ax.annotate('',xy=en,xytext=st,arrowprops={'arrowstyle':'->','lw':1.6,'color':'#44556b'})
ax.text(.5,.96,'优先验证的结构假设：保留全局证据，补充类别专属的局部证据',ha='center',fontsize=14)
ax.text(.5,.015,'后续逐项增加：线性频率 STFT 分支 → 同一训练身份的等功率教师蒸馏；推理只输入 I/Q',ha='center',fontsize=10,color='#475569')
chart2=OUT/'图2_建议算法路线.png';fig.savefig(chart2,dpi=180,bbox_inches='tight',facecolor='white');plt.close(fig)

doc=Document(SOURCE)
def setfont(run,size=None,bold=None,color=None):
 run.font.name='Calibri';run._element.get_or_add_rPr().rFonts.set(qn('w:eastAsia'),'等线')
 if size:run.font.size=Pt(size)
 if bold is not None:run.bold=bold
 if color:run.font.color.rgb=RGBColor.from_string(color)

normal=doc.styles['Normal'];normal.font.name='Calibri';normal.font.size=Pt(10.5);normal._element.get_or_add_rPr().rFonts.set(qn('w:eastAsia'),'等线')
normal.paragraph_format.line_spacing=1.2;normal.paragraph_format.space_after=Pt(6)
for level,size in [(1,16),(2,13),(3,11.5)]:
 st=doc.styles[f'Heading {level}'];st.font.name='Calibri';st.font.size=Pt(size);st.font.color.rgb=RGBColor.from_string('155E75');st._element.get_or_add_rPr().rFonts.set(qn('w:eastAsia'),'等线');st.paragraph_format.keep_with_next=True
for section in doc.sections:
 section.top_margin=Cm(2);section.bottom_margin=Cm(1.9);section.left_margin=Cm(2);section.right_margin=Cm(2)
 section.page_width=Cm(21);section.page_height=Cm(29.7)
 hp=section.header.paragraphs[0];hp.text='长尾复合干扰识别  |  文献核验与实验建议';setfont(hp.runs[0],8,color='64748B')
 fp=section.footer.paragraphs[0];fp.text='2026-11-02 增补版  ·  开发证据与待验证方案分列';fp.alignment=WD_ALIGN_PARAGRAPH.CENTER;setfont(fp.runs[0],8,color='64748B')
 fp.add_run('    ');fld=OxmlElement('w:fldSimple');fld.set(qn('w:instr'),'PAGE');fp._p.append(fld)

# Correct statements in place, preserving the six original sections and their structure.
for p in doc.paragraphs:
 t=p.text
 if t=='长尾学习方法综述与试验建议':
  p.text='长尾方法综述与试验建议';p.style=doc.styles['Title']
 elif t.startswith('更新时间：'):
  p.text='更新时间：2026-11-02｜联合功率不均衡任务补充调研版'
 elif t.startswith('本文根据本地文献库中的'):
  p.text='原稿根据本地 GCD、增量类别发现、开放世界持续学习与复合干扰资料整理。本版保留原六类方法框架，补充公开论文、作者代码核验、当前联合任务诊断和可执行实验路线。文献中的操作建议仅作为研究材料；本次没有据此启动训练、改动模型或访问新的锁定测试。'
 elif t.startswith('实验时应先把'):
  p.text='实验应区分标签频次、组合频次和分量功率可见度。历史基础六类池的每组合400条属于均衡数据；当前主任务已改用 joint_ama_v1/label_if3_mixed，训练8000条、标签IF=3，并混合0/6/12/18/24 dB相对功率差。旧v3只作历史分类参考。'
 elif t.startswith('当前 Route-A 主基线是'):
  p.text='历史Route-A v3五种子结果为 Macro-F1 0.925639±0.001556、Exact 0.651835±0.005762、Macro-AUPR 0.977776；它使用历史预训练初始化，属于已消费开发证据。当前可用联合任务基线为随机初始化DFMS-IQ+BCE，seed=2050、8 epochs：Macro-F1 0.895560、Exact 0.443700、弱Macro-AUPR 0.303164、弱宏召回0.140322。两套结果不可直接排名，当前结果没有五种子复现。'
 elif t.startswith('这组结果是开发证据。'):
  p.text='上述均为开发证据。新方法采用独立输出目录和manifest；check负责选模，calibration负责阈值，development-test仅用于冻结后的报告。已消费development-test不再用于方法或超参数搜索。可先双种子筛查，入围路线再用五种子确认；最终结论另需新身份锁定集。'
 elif t.startswith('注意：多标签任务的负标签远多于正标签'):
  p.text='修订：不能假定本项目负标签占绝对多数。当前8000×6个标签位中，正标签24897个（51.87%）、负标签23103个（48.13%）；类别间先验差异与弱正例困难更突出。ASL、DB等来自稀疏多标签场景的参数不应直接照搬；应控制正负权重、整体梯度尺度和逐类FPR。[R12–R15]'
 elif t.startswith('在本地审计中，Base-DB/ASL'):
  p.text='原稿记录历史Base-DB/ASL、冻结头BCE/Focal/DB未取得统一收益；本次未定位这些分支在当前joint协议下的权重与收据，不能据此认定它们在当前任务必然无效。doc.md可确认历史v3有BCE与Logit Adjustment五种子对照，整体增益不稳定；新联合任务需要独立受控比较。'
 elif t.startswith('stage 2：冻结 backbone'):
  p.text='stage 2：冻结backbone，只在fit身份池内部重训分类头；如需重采样，记录唯一身份与步数。calibration仅拟合最终决策参数，不用于分类头梯度训练。'
 elif t.startswith('当前结果已经说明阈值校准很重要'):
  p.text='历史v3的Exact曾由0.605704升至0.651835。当前联合任务也由0.370040升至0.443700，但弱宏召回由0.152617降至0.140322。因此须分开报告排序、阈值与表征收益；温度校准在保持0.5决策时不会改变二分类符号，也不会提升单标签排序AUPR。'
 elif t.startswith('文献启示：IGCD 使用'):
  p.text='原稿借用IGCD的支持集、伪标签和蒸馏思路，并记录某历史hard-negative分支因严格留出增益不足而停止；这些不是当前joint协议的直接结果。当前已有完整标签，优先使用成分真值和配对可见度监督，避免额外引入不可靠伪标签。'
 elif t.startswith('低功率 hard negative：'):
  p.text='功率配对正样本：保持成分、标签与背景，仅改变某分量功率；真正的hard negative应使该类缺席，再控制其他成分和噪声。两种配对分别记录。'
 elif t.startswith('BCE 主损失 + supervised contrastive'):
  p.text='BCE主损失+类别条件对比：在某标签的专属特征空间构造正对；仅共享一个标签的混合样本仍可能具有不同其他成分，不应将整个全局表示强行拉近。'
 elif t=='4. 推荐的试验优先级':
  p.text='4. 原稿试验优先级（历史建议，现阶段以第12–13节为准）'
 elif t.startswith('第一优先级：建立可信长尾基线'):
  p.text='原稿第一优先级：建立长尾对照。当前应先补联合任务的等功率训练对照与训练预算检查，再进入损失、采样与结构筛查。'
 elif t.startswith('所有方法用同一 v3 新身份池'):
  p.text='当前联合任务各方法使用同一joint_ama_v1训练身份池、相同开发角色与预算。v3不含逐分量功率真值，不承担弱分量结论。不同功率视图的身份必须一起分组。'
 elif t.startswith('用九挡配对视图评估'):
  p.text='历史九挡为绝对JNR扫描；当前五档为同一样本内相对分量功率差。两者分别报告，不能把低混合JNR与弱分量遮蔽等同。当前优先复用五档配对视图，再另设新身份的连续功率差与未见组合测试。'
 elif t=='6. 主要参考入口':
  p.text='6. 原稿本地资料入口（沿用历史记录）'
 for run in p.runs:setfont(run)

# A concise revision note near the front gives the reader the corrected interpretation.
anchor=next(p for p in doc.paragraphs if p.text.startswith('1. 使用说明'))
for txt in [
 '本版核心建议：先补齐等功率与训练预算对照，再优先验证“保留全局统计路径的类别条件局部证据”；其后逐项加入I/Q–时频融合及配对功率教师蒸馏。潜力排序属于本项目证据与文献机制的推断，尚无新增训练增益。',
 '阅读导航：第1–6节保留并修订原方法综述；第7节给出当前诊断；第8–10节比较同领域、跨领域与不同输入；第11节核验代码；第12–14节给出方案、实验与证据边界；第15节提供24项论文/技术资料入口。']:
 p=anchor.insert_paragraph_before(txt);p.paragraph_format.space_after=Pt(8)
 for run in p.runs:setfont(run,10.5)

def p(text,bold=False):
 q=doc.add_paragraph(text)
 if bold:
  for r in q.runs:setfont(r,bold=True,color='155E75')
 return q
def h(text,level=1):return doc.add_heading(text,level)
def bullet(text):
 q=doc.add_paragraph(text,style='List Bullet');return q
def link(par,text,url):
 e=OxmlElement('w:hyperlink');e.set(qn('r:id'),par.part.relate_to(url,RT.HYPERLINK,is_external=True))
 r=OxmlElement('w:r');pr=OxmlElement('w:rPr');c=OxmlElement('w:color');c.set(qn('w:val'),'16728C');pr.append(c);r.append(pr);t=OxmlElement('w:t');t.text=text;r.append(t);e.append(r);par._p.append(e)
def table(headers,rows,widths=None):
 t=doc.add_table(rows=1,cols=len(headers));t.style='Table Grid';t.alignment=WD_TABLE_ALIGNMENT.CENTER
 for cell,txt in zip(t.rows[0].cells,headers):cell.text=txt
 rep=OxmlElement('w:tblHeader');t.rows[0]._tr.get_or_add_trPr().append(rep)
 for row in rows:
  cells=t.add_row().cells
  for cell,txt in zip(cells,row):cell.text=str(txt)
 for ri,row in enumerate(t.rows):
  for ci,cell in enumerate(row.cells):
   cell.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
   if widths:cell.width=Cm(widths[ci])
   for pp in cell.paragraphs:
    pp.paragraph_format.space_after=Pt(4);pp.paragraph_format.space_before=Pt(3)
    for rr in pp.runs:setfont(rr,9,bold=(ri==0),color='FFFFFF' if ri==0 else None)
   if ri==0:
    sh=OxmlElement('w:shd');sh.set(qn('w:fill'),'155E75');cell._tc.get_or_add_tcPr().append(sh)
   elif ri%2==0:
    sh=OxmlElement('w:shd');sh.set(qn('w:fill'),'F0F6F7');cell._tc.get_or_add_tcPr().append(sh)
 doc.add_paragraph('')
 return t
def image(path,caption):
 doc.add_picture(str(path),width=Cm(16.8));q=doc.add_paragraph(caption);q.style='Caption'

doc.add_page_break()
h('7. 当前联合任务的证据与问题重定位')
p('本节数字来自本机冻结checkpoint、predictions.npz和metrics.json，不取自外部论文。此次只做既有预测诊断与文献调研，没有重新训练，也没有在development-test上搜索阈值。原始与CPU重载的弱AUPR有约0.000011差异，F1、Exact及弱召回复现。')
table(['项目','核验结果'],[
 ('主任务','标签IF=3 + 五档相对分量功率差；六类多标签，63个非空组合'),
 ('数据角色','fit=8000；calibration=1008；check=1008；development-test=4032'),
 ('弱定义','活动分量比同一样本最强分量低至少10 dB；单分量不标弱'),
 ('初始化与预算','随机初始化；BCE+AdamW；全局batch32；8 epochs；seed2050'),
 ('模型与耗时','DFMS-IQ，257511参数；历史训练收据记录两张A800约139秒'),
 ('结果边界','单种子开发基线；不是历史预训练v3成绩，也不是锁定测试'),
 ('已有工程验证','34项测试通过；主数据、等功率对照与五档视图哈希/身份核验通过')])
table(['指标','固定0.5','逐类F1校准'],[
 ('Macro-F1','0.871660','0.895560'),('Exact','0.370040','0.443700'),('Macro-AUPR','0.958331','0.958331'),
 ('弱Macro-AUPR','0.303164','0.303164'),('弱宏召回','0.152617','0.140322'),('尾类弱汇总召回','0.058997','0.300885')])
p('尾类为AMA、SinFM；尾类指标是按弱正例数量汇总的召回。弱Macro-AUPR只使用每类弱正例和该类缺席负例，随机排序的正例先验约0.143–0.153，与总体AUPR的任务不同，二者不应直接按差值解释“性能损失”。')
table(['类别','训练正例','弱召回','弱AUPR','缺席误报率'],[
 (c,S['fit_label_counts'][i],f"{weak['per_label'][c]['weak_recall']:.2%}",f"{weak['per_label'][c]['weak_aupr']:.4f}",f"{weak['per_label'][c]['negative_fpr']:.2%}") for i,c in enumerate(cl)])
image(chart,'图1：当前联合任务的瓶颈。功率曲线来自冻结预测的相对功率分层；不是新功率扫描实验。')
h('7.1 整体正确率为什么只有44.37%',2)
p('4032条样本中1789条六标签全部正确，2243条错误。其中2117条仅错一个标签，占错误样本94.38%；另有125条错两个、1条错三个。仅漏检的样本1950条，仅误报180条，同时含漏检与误报113条。弱分量共2088个，漏检1803个，占全部2070个标签漏检的87.10%。因此，优先补回“最后一个被掩盖的成分”比笼统追求更多网络容量更有针对性。')
p('诊断性oracle：用真值把全部弱正例漏检改正、其余预测位保持不变，Exact变为0.867312，新增1708条完全正确样本。该操作使用了答案，只用来定位错误来源；它不是可部署方法、可实现性能承诺或Bayes上界。实际改进仍必须同时抑制新增误报。',True)
h('7.2 对原稿五处关键修订',2)
for txt in [
 '历史与当前分开：0.651835是历史v3预训练五种子开发结果；0.443700是当前联合任务单种子从头训练结果。',
 '负标签并未全局占优：当前训练正标签比例51.87%，重点应从“默认负例海量”转向逐类先验、强弱可见度与难负例。',
 '校准集不训练头：分类头重训只能使用fit内部数据；calibration只拟合阈值、偏置等决策参数。',
 '弱分量不等于低混合JNR：本任务最强分量30 dB、被衰减分量最低6 dB；主要困难包含相对遮蔽。',
 '当前全部63种非空组合合法：仅枚举合法组合的独立Bernoulli MAP，通常等价于逐类0.5判决，只在全零预测时产生差别；要产生新的联合决策效果，必须额外学习相关项或损失敏感决策，且不能硬套训练组合频率。']:
 bullet(txt)

h('8. 相同领域：最相关论文与可迁移机制')
p('筛选依据为任务、接收通道、输入、标签形式与强弱分量设定，而不是论文中的最高准确率。论文各自的Accuracy、subset accuracy、低JNR定义、组合数量和数据身份不同，下面不做跨论文分数排名。未找到作者公开实现的工作仅列机制参考，不写成“已复现”。')
table(['文献','任务与输入','本项目可借鉴处','迁移边界'],[
 ('[R1] MLAMC，2020','复合信号；时频表示；多标签','按成分输出与多阈值决策','只核验出版社摘要；不推断未披露实现细节'),
 ('[R2] ML-CR-CV-CNN，2023','原始复数I/Q；4类/15组合；含功率比变化','类别原型与注意力解耦、阈值校准','最贴近强遮蔽问题；数据依申请；未找到作者公开代码'),
 ('[R3] Separation+Recognition，2024','从多接收通道分离后识别','分离与识别解耦的思路','其多通道观测优势不能直接用于本项目单通道'),
 ('[R4] DBRF-Net，2025','时间序列+CWT；9单类/20复合类','同一信号的双表示与自适应融合','主要是预定义复合类型分类、低混合JNR，不等同六标签弱识别'),
 ('[R5] 分割+融合注意力，2025','STFT图像分割后识别','给成分级局部证据显式监督','重叠成分未必可用互斥像素分割；需非互斥掩码'),
 ('[R6] 多特征轻量网络，2026','三种时频表示；轻量融合','比较表示互补和预算效率','预定义复合类别任务；不能据其分数推断当前Exact'),
 ('[R7] 先验引导双任务，2026-09','STFT+手工特征+软先验；分类与掩码','物理特征作为有置信门控的辅助证据','22类及额外分割真值；先验必须仅由fit学习')], [3.4,4,4.5,4.8])
p('最应精读[R2]：它已直接研究复数输入、功率比和强重叠，并用类别相关表征解耦。因此“复数CNN+注意力+原型”本身已有先例。当前项目可提出的研究问题，应落在经过身份审计的标签长尾与功率遮蔽联合协议、成分级局部证据、训练/测试组合先验变化，以及相同误报率下的恢复能力，而不是把现成模块组合宣称为已证实创新。')
p('2025–2026年的融合工作说明多表示值得做对照，但它们没有证明时频图像总优于I/Q。STFT/CWT是同一观测的确定性变换，不增加新的物理信息；可能改善的是有限模型的特征提取与优化。')

h('9. 不同领域：需求相似、输入不同的迁移地图')
table(['领域与输入','共享困难','文献/实现','建议迁移与不宜照搬之处'],[
 ('多声源音频：实波形/谱图','多事件重叠，弱事件被强背景遮蔽','PANNs[R8]、PSLA[R9]','优先借类别条件时间池化与波形/谱图融合；不直接用音频采样率、Mel频率或预训练权重'),
 ('多物体视觉：二维特征图','弱、小、局部目标不能被全局平均充分表达','Query2Label[R10]、ML-Decoder[R11]','把图像位置改为时间或时频token；6个类别查询；保持小容量'),
 ('长尾多标签视觉','标签频率与共现偏置、负梯度控制','ASL[R12]、DB[R13]、DR[R14]','作为低成本损失对照；本项目只有6类、正负总体接近，默认超参数不适用'),
 ('文本/医学主题：token序列','标签共现与长尾，盲目补尾会连带增加头类','BalancedLossNLP[R15]','借共现审计与分布偏移分析；文本没有线性物理叠加，无法证明信号可分离'),
 ('语音分离：单通道实波形','从混合观测恢复各成分','Conv-TasNet[R17]、Asteroid[R18]、MixIT[R19]','可借可学习分析/掩码/合成；需改为复数双通道与静默成分处理'),
 ('通信RF：复数I/Q','单通道同频干扰与目标信号分离','RF Challenge[R20]','与输入最接近；参考I/Q张量和U-Net/时域模型；其目标为已知通信SOI恢复，不是六标签识别'),
 ('声音事件蒸馏/混合监督','教师监督与复杂训练角色','DualKD[R21]、DCASE2024[R22]','借教师冻结、标签/帧级蒸馏；音频“weak label”指时间标注粗，不等于本项目弱功率')], [3.5,4,4,5.2])
p('迁移优先看机制能否对应当前错误：类别条件池化对应全局聚合；配对蒸馏对应同一分量的可见度变化；损失重加权对应梯度分配；分离模型对应物理叠加。不同领域的mAP提升不能换算成本项目的Exact提升。')

h('10. 输入表示与观测条件：应如何公平比较')
table(['输入/辅助信息','可验证假设','实现要点','必要对照'],[
 ('原始I/Q（现有）','保留相位与干涉结构','继续[B,2,3000]，RMS保留相对功率差','当前DFMS-IQ与预算匹配的改进'),
 ('线性频率STFT功率','弱窄带/短时结构更易显露','复数输入采用双边频谱；log功率；无需RGB截图','仅STFT、仅I/Q、两者融合'),
 ('多分辨率STFT','同时观察短时转发与较长调频轨迹','可先比较64/256点窗口、16/64点hop；配置固定','单分辨率与多分辨率、相近参数量'),
 ('CWT/其他变换','固定窗口难以覆盖多时间尺度','先证明比轻量STFT更有效，再承担计算成本','同预算STFT与CWT'),
 ('匹配滤波/相关等物理特征','突出MFT/ISRJ与参考LFM的转发结构','只使用部署时可得参考；加真实目标后重新审查','物理特征单独、软融合、参考失配'),
 ('干净成分/等功率视图（训练用）','可见度更好的教师能补充弱监督','不得作为推理输入；同身份所有视图同角色','相同额外视图预算的增强对照'),
 ('额外阵元/多脉冲','额外观测可能改善可辨识性','这会改变问题设定与采集条件','单通道与多通道分别报告')], [3.5,4.3,5,4])
p('10 MHz采样下，64/256点窗口分别覆盖6.4/25.6微秒；这些是候选设置，不是调优结论。不要把I/Q重采样到16 kHz来套音频模型，也不要使用音频单边频谱或听觉Mel压缩作为未经验证的默认值。时频动态范围必须保留24 dB弱分量差异，避免过度裁剪把信号证据抹掉。')
p('对单通道混合信号，监督模型可以利用分量先验做分离，但确定性时频变换不会让任意混合都变得唯一可分。应通过新身份的配对有/无分量样本检验可观测性；若独立检测器与多表示都接近随机，再考虑额外观测或生成协议，而不是无限增加网络。')

h('11. 开源代码核验与接入成本')
p('共核验13个公开仓库：读取截至2026-11-02可获取的固定commit、README及关键文件；未执行这些工程的完整训练。以下是静态代码审查，完整commit、文件SHA-256和来源链接保存在同目录“开源实现核验.json”。“未检测到根许可证”不等于断言没有许可证。')
repos=json.loads((OUT/'开源实现核验.json').read_text())
notes={
 'Alibaba-MIIL/ASL':('MIT','src/loss_functions/losses.py','直接输出损失sum；需与本项目BCE mean对齐尺度，避免把学习率变化误当方法收益。'),
 'Alibaba-MIIL/ML_Decoder':('MIT','src_files/ml_decoder/ml_decoder.py','默认768维、FFN2048，对25.75万参数模型过大；改用小维度需明确记录适配。'),
 'SlongLiu/query2labels':('MIT','lib/models/query2label.py','原实现Conv2d投影、类别query和GroupWiseLinear；迁移一维token，保持六维logits。'),
 'qiuqiangkong/audioset_tagging_cnn':('MIT','pytorch/models.py: AttBlock / Cnn14_DecisionLevelAtt','类别专属时间权重；音频前端和大型Cnn14权重不直接搬用。'),
 'YuanGongND/psla':('BSD-3-Clause','src/models/Models.py','注意力聚合与训练技巧；增强标签针对音频漏标，本项目真值完整，不能自动补标签。'),
 'wutong16/DistributionBalancedLoss':('未检测到根许可证','mllt/models/losses/resample_loss.py','旧MMCV/注册器、硬编码.cuda()及外部频次文件；提取数学公式做独立适配。'),
 'Kunmonkey/DR-Loss':('未检测到根许可证','mllt/models/losses/dr_loss.py','按batch维做class-wise排序，返回逐类向量；需稳定化、归约与DDP批语义审查。'),
 'Roche/BalancedLossNLP':('CC BY-NC 4.0（LICENSE.md）','Reuters/util_loss.py','DB损失的文本适配；只借损失与共现分析，不借文本模型。'),
 'facebookresearch/classifier-balancing':('BSD条款（LICENSE）','main.py / 分类头重训流程','原设定为单标签softmax；多标签需六个二分类头，不能直接套类别互斥假设。'),
 'naplab/Conv-TasNet':('README声明CC BY-NC-SA 3.0 US','conv_tasnet.py','作者最小实现；可参考结构，工程化可优先看Asteroid。'),
 'asteroid-team/asteroid':('MIT','asteroid/models/conv_tasnet.py','现代PyTorch分离工具链；复数I/Q、缺席类与输出数仍需适配。'),
 'RFChallenge/icassp2024rfchallenge':('未检测到根许可证','src/unet_model.py / README.md','此U-Net文件是TensorFlow；README另列PyTorch WaveNet；并非可直接导入本项目的PyTorch头。'),
 'google-research/sound-separation':('Apache-2.0','README.md / MixIT相关模型','适合未来无干净真值的混合训练；当前已可生成干净成分，先监督对照。')}
for row in repos:
 repo=row['repository'];lic,entry,note=notes[repo]
 h(repo,2)
 p(f"固定版本：{row.get('commit','不可用')[:12]}；最近核验提交日期：{row.get('commit_date','未知')[:10]}；许可信息：{lic}。")
 q=p('入口：'+entry+'。'+note+' ');link(q,'仓库',row['url'])
 file=next((f for f in row.get('files',[]) if f.get('path')==entry.split(' / ')[0] and not f.get('error')),None)
 if not file:file=next((f for f in row.get('files',[]) if f.get('path')!='README.md' and not f.get('error')),None)
 if file:link(q,' ｜ 核验源码',file['url'])
p('DR-Loss特别注意：原实现将exp与logsumexp相乘再取log，极端logits或混合精度下要用等价稳定形式；class-wise比较以本地batch为单位，双GPU时每rank16条不等于全局32条。ASL特别注意：不能把其sum归约直接与现有mean归约BCE比较。ML-Decoder特别注意：原默认query在非零样本模式下被冻结，与Query2Label可学习类别query的实现不同。')

h('12. 哪些下一步最可能提升整体识别正确率')
p('下列优先级是基于错误归因、同领域先例、跨领域机制与改动成本的研究判断，不是已训练验证的增益排序。主要目标为提高六标签Exact，同时要求弱识别改善、误报与强分量性能受控。')
table(['优先级','路线','为什么优先/主要风险','验证成本'],[
 ('P0 必做','等功率对照、8/24轮预算、冻结决策比较','最后一轮仍最佳；先排除训练不足和阈值口径差异。不能保证延长训练一定有益。','低'),
 ('P1 首选结构','保留全局路径的类别条件局部证据','94.38%错误为单标签；全局均值/方差可能稀释类别证据。弱特征是否局部可辨仍需验证。','低—中'),
 ('P2 输入互补','I/Q + 线性频率STFT轻量分支','同领域与音频均有先例，可突出不同结构；可能只是容量增加，且功率图损失相位。','中'),
 ('P3 配对训练','训练身份内等功率教师→混合功率学生','现有成分和噪声配对有价值；全局一致性曾失败，需类别条件、教师可靠性门控。','中'),
 ('P4 损失对照','弱正例有界权重；ASL/DB/DR择一筛查','实现成本低，直接改变困难正例梯度；可能增加头类误报，不能代替表征。','低—中'),
 ('P5 后续升级','监督复数成分估计 + 识别','更接近物理叠加机制；生成器可重建干净成分，但工程/计算与分离伪影成本高。','高')], [2.5,4.5,6.6,2])
image(chart2,'图2：建议首选结构。先验证类别条件局部证据，输入融合和蒸馏各自单独消融。')
h('12.1 首选：全局路径 + 类别条件局部证据',2)
p('保留当前mean/std全局路径作为强分量基线，另从编码器较浅层或中层取得时序特征。用6个类别query或6组注意力权重分别聚合证据，产生类别专属特征，再与全局logits轻量融合。当前encoder已有forward_pyramid接口，可复用多尺度输出；先比较单层与多层，不一次加入原型、Transformer和蒸馏。[R2,R8,R10,R11]')
p('最小消融：H0现有统计池化；H1同容量普通投影；H2类别条件池化；H3全局+类别条件；H4入围结构再加多尺度。只有H3/H4在相同误报与参数预算下改善弱AUPR和Exact，才支持“局部类别证据”假设。权重图可辅助定位，但不能单凭好看的注意力图证明因果机制。')
p('接入建议：新建models/dfms_query.py，沿用build_model入口及[B,2,3000]→[B,6]协议。先用64–128维、单层跨注意力或更小的类别池化；不要照搬ML-Decoder默认768维。保留全局支路不等于强性能必然保持，仍需按冻结gap_0视图验证。')
h('12.2 次选：I/Q 与时频表示互补',2)
p('在模型内部从同一归一化I/Q计算可配置STFT，保持数据与评测入口不变。I/Q分支保留相位，时频分支输入线性频率log功率；先做轻量2D卷积再聚合。必须与“参数增加相当的I/Q模型”比较，区分表示收益和容量收益。物理相关特征可作为更轻的替代分支，但其参考LFM必须部署可得。[R4–R9]')
h('12.3 配对等功率教师：训练辅助，推理无额外输入',2)
p('教师只看fit身份的gap_0视图，学生看同身份混合功率视图；标签、干净分量参数和噪声对应。教师冻结后，仅在正确且置信足够的类别上施加soft目标，优先蒸馏类别专属证据或logits，BCE持续约束全部六类。不要强迫全部全局特征相等，也不要把教师预测替代已知真值。[R21,R22提供蒸馏机制参考；等功率配对为本项目适配提案]')
p('可先用 L = L_BCE + λ·L_KD，其中L_KD按类别可靠性和真实弱标记门控。弱真值只参与训练权重，推理仍仅输入I/Q。需要对照：普通额外等功率增强、全logits一致性、类别条件蒸馏，以及相同额外视图数量/优化步数。若历史全局一致性的退化重现，应停止该分支，不能用蒸馏名称掩盖相同方法。')
h('12.4 低成本损失：针对困难正例，控制误报',2)
p('先做有界弱正例权重：只提高真实弱正例的正项BCE权重（如候选1/2/4），保留缺席负例和非弱正例约束，并归一化损失尺度。与按标签频率加权分开，避免把“弱”与“尾”混为一谈。ASL、DB、DR选少量固定网格逐一比较，所有频次只从fit计算；不启动采样×损失×网络的大笛卡尔积。[R12–R15,R23,R24]')
p('当前FMB在固定0.5下的弱召回67.70%伴随51.56%误报；校准后弱召回4.21%、误报1.81%。因此提高召回必须在可比FPR下判断。把阈值一味调低会制造“改善弱识别”的假象。')
h('12.5 分离路线：先监督下限，再考虑无监督',2)
p('生成器能按sample seed重建干净分量，但当前prepared数据只保留混合波形与功率；要做波形监督需新增成分目标、噪声目标及对应manifest。可设6个类别固定槽位加噪声槽，避免把同类分量视作互斥说话人；缺席类采用静默约束。优先尝试复数双通道时域/U-Net或掩码模型，评估分离前后识别、能量泄漏与混合重构一致性。[R3,R17–R20]')
p('复数源可能相消，混合幅度掩码并不总是0到1；不能机械套用实语音的幅度比掩码或对缺席源计算未处理的SI-SDR。当前已有生成真值，不优先采用MixIT无监督目标；其价值在未来实测混合、缺少干净成分时再评估。')

h('13. 最小实验计划、角色隔离与验收标准')
p('本节为后续实施建议。本次只交付调研与文档，以下作业均未运行。先小规模筛查，入围后五种子确认，避免一次投入大矩阵。')
table(['编号','实验','唯一主要变化','回答的问题'],[
 ('E00','复现joint当前基线','8轮、seed2050，输出新目录','链路和阈值是否一致'),
 ('E01','joint预算检查','24轮；其余相同','8轮是否限制性能'),
 ('E02','equal训练对照','与E01同预算、身份、混合开发池','训练中出现功率差是否有帮助'),
 ('E03','类别条件局部头','与预算匹配基线比较','局部类别证据是否改善排序和Exact'),
 ('E04','I/Q+STFT','固定入围结构，单独增加表示','表示互补还是单纯容量收益'),
 ('E05','配对教师蒸馏','对入围模型单独加KD','收益是否超过相同视图预算的增强'),
 ('E06','有界弱正例BCE','对入围模型单独改权重','梯度分配能否改善FPR受控召回'),
 ('E07','通用损失对照','ASL/DB/DR中先筛一项','专门设计是否超过通用损失')])
p('E01/E02先用相同两种子筛查，例如2050/2051；E03及以后逐级推进。入围路线与相应基线扩至2050–2054五种子，报告每个种子的差值，而不是只比最好的种子。若预算增加，则所有比较组同步增加；教师训练和额外视图成本单列。现有139秒只是旧基线训练收据，不能用来承诺新结构耗时。')
h('13.1 指标与决策规则',2)
for txt in [
 '整体主指标：六标签Exact；并报告Macro-F1、Macro-AUPR、Hamming及按标签基数分层结果。均衡组合池中样本Exact与宏组合Exact相同，不必当作两份独立证据。',
 '弱识别指标：弱Macro-AUPR、弱宏/微召回、尾类弱召回、逐类与功率差分层；补充在校准FPR目标1%/5%下的弱召回，并报告测试时实际FPR，阈值不能保证跨分布严格满足目标。',
 '决策对照：固定0.5、逐类F1阈值、以Exact为目标的低自由度阈值坐标搜索；搜索只在calibration上运行，候选规则在check上选择。不得在development-test上挑规则。',
 '检查排序收益：阈值或单标签单调温度/偏置可改变判决，不能改善该标签AUPR排序。F1/Exact上升但弱AUPR不变时，应归为决策收益。',
 '预先固定选模协议：保持弱AUPR选epoch可与当前协议直接比较；若改为校准后Exact选epoch，则每个epoch先在calibration拟合决策、再用check选点，并给基线相同流程。两种协议分列。']:
 bullet(txt)
h('13.2 建议的晋级门槛与统计报告',2)
p('可预先约定工程晋级门槛，例如Exact平均提升至少2个百分点、弱Macro-AUPR提升至少0.02、gap_0或非弱召回下降不超过1个百分点，且逐类缺席FPR无明显失控；这些数字是建议决策门槛，不是统计显著性结论或增益预测。任务方应在新报告集使用前确定门槛。')
p('统计上同时报告五种子配对差值和按基础身份分组的置信区间。配对功率视图共享分量与噪声，重采样时必须捆绑，不能当作独立测试样本；五个模型种子也不等于五份独立测试数据。仅选择某个功率档、某个种子或某个决策的最好值不能支持稳定整体增益。')
h('13.3 数据隔离与扩展',2)
p('所有同身份的功率视图和有/无分量反事实样本必须进入同一角色。反事实对还须保留其组成分量seed、接收机噪声seed和背景身份，避免因噪声seed不同而漏掉成分复用。当前prepared数据只保留部分身份字段，新的反事实协议需要扩展manifest，不能依赖行号当独立身份。')
p('当前开发池已经被阅读。后续方法与超参数只用fit/check/calibration；新身份development确认集用于阶段汇报，最终另建未消费锁定集。若要论证“标签长尾×功率差”的交互作用，还需增加标签均衡/长尾×等功率/混合功率的2×2协议；当前mixed/equal只回答标签IF=3条件下的功率训练影响。')
p('泛化测试另列：新物理参数范围、连续或未见功率差、多个同时弱分量、未见已知组合、纯噪声和真实目标。六类全部非空组合测试不覆盖无干扰背景，不能把现有逐类FPR外推成实际系统虚警率。')

h('14. 应降级或停止的路线，以及接手注意事项')
table(['路线/常见误区','当前建议','理由'],[
 ('只换大Transformer或音频大模型','低优先级','目前未证明容量是瓶颈；输入物理尺度和预训练领域不匹配'),
 ('继续只按逆标签频率过采样','只作受控对照','FMB样本最多仍弱识别差；会连带改变组合与身份覆盖'),
 ('63类softmax替代六标签主输出','不作主线','训练组合记忆可能掩盖成分能力，未见组合泛化受限'),
 ('硬编码训练组合先验或普通合法MAP','不作核心创新','当前63组合全合法；训练长尾与均衡报告池先验不同'),
 ('校准后召回提高就宣称表征改善','拒绝该解释','必须检查相同FPR与弱AUPR；排序与决策收益分开'),
 ('通用对比学习/伪标签/GCD先上','后置','当前已有完整监督与已知闭集；标签交叠不等于类身份相同'),
 ('全面特征一致性或重建辅助任务','先看历史失败账本','历史已有梯度冲突和anchor下降；需提出可证伪差异'),
 ('多通道分离论文直接对比单通道','不公平','额外观测改变可辨识性和采集成本')])
p('工程接手：本机现用src/configs/tests尚有未跟踪内容，Git存在大量旧文件删除记录，README提到的/tmp历史压缩备份已不存在。应先保存当前可运行快照、依赖锁与运行收据，再实施新方法。此项是已有工程状态，不影响本报告对冻结结果的复核。')
p('停止条件建议：相同FPR下弱AUPR/召回无改善、Exact仅来自更宽松阈值、强分量显著退化、预算匹配后增益消失、严格身份留出后失效，任一出现均需降级路线并记录失败原因。不要持续读取同一报告集寻找可发表的分数。')

h('15. 论文、技术资料与公开实现索引')
p('检索截至2026-11-02。来源优先采用出版社/会议、作者arXiv与机构页面，以及论文链接的作者仓库。本文为针对当前项目的选择性调研，不声称穷尽所有文献。部分出版社正文受访问限制，已标明摘要或搜索索引核验；不据此臆测代码可用性与复现实验细节。仓库可访问及静态审查不等于端到端可运行。')
REFS=[
 ('R1','Zhu, M., et al. Automatic modulation recognition of compound signals using a deep multi-label classifier: A case study with radar jamming signals. Signal Processing, 169, 107393, 2020.','https://doi.org/10.1016/j.sigpro.2019.107393','出版社摘要核验；MLAMC任务与时频/多标签思路。'),
 ('R2','Meng, Y.; Yu, L.; Wei, Y. Multi-Label Radar Compound Jamming Signal Recognition Using Complex-Valued CNN with Jamming Class Representation Fusion. Remote Sensing, 15(21), 5180, 2023.','https://doi.org/10.3390/rs15215180','已核验全文PDF；4类15组合、JNR/PR、LVQ表征、类别解耦与阈值；作者公开代码未找到。'),
 ('R3','Zhou, H., et al. Compound radar jamming recognition based on signal source separation. Signal Processing, 214, 109246, 2024.','https://doi.org/10.1016/j.sigpro.2023.109246','出版社摘要核验；多接收通道分离后识别。'),
 ('R4','Lu, W.; Li, J.; Xie, F.; Liu, H. Compound Jamming Recognition Under Low JNR Setting Based on a Dual-Branch Residual Fusion Network. Sensors, 25(18), 5881, 2025.','https://doi.org/10.3390/s25185881','出版社全文搜索索引核验；时域+CWT，预定义复合类型分类。'),
 ('R5','Li, P.; Yang, J.; Lin, J. Radar Compound Jamming Recognition Based on Image Segmentation and Fused Attention Residual Network. Sensors, 25(7), 2124, 2025.','https://doi.org/10.3390/s25072124','出版社PDF索引核验；STFT分割与识别。'),
 ('R6','Zha, W., et al. Research on Multi-Feature Fusion and Lightweight Recognition for Radar Compound Jamming. Sensors, 26(4), 1296, 2026.','https://doi.org/10.3390/s26041296','出版社/PMC全文搜索索引核验；多时频表示与轻量融合。'),
 ('R7','Qiu, T., et al. Prior-Guided Lightweight Dual-Task Network for Composite Active Jamming Recognition and Time-Frequency Parameter Estimation in Radar Remote Sensing. Remote Sensing, 18(17), 2946, 2026.','https://doi.org/10.3390/rs18172946','2026-09-01；出版社方法与实验节搜索索引核验；22类、先验软门控与非互斥掩码。'),
 ('R8','Kong, Q., et al. PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition. IEEE/ACM TASLP, 2020.','https://arxiv.org/abs/1912.10211','作者论文与代码；类别条件时间聚合、Wavegram/Logmel融合。'),
 ('R9','Gong, Y.; Chung, Y.-A.; Glass, J. PSLA: Improving Audio Tagging with Pretraining, Sampling, Labeling, and Aggregation. IEEE/ACM TASLP, 2021.','https://arxiv.org/abs/2102.01243','作者论文与代码；训练/聚合技巧，标签增强适用条件需区分。'),
 ('R10','Liu, S., et al. Query2Label: A Simple Transformer Way to Multi-Label Classification. arXiv:2107.10834, 2021.','https://arxiv.org/abs/2107.10834','作者预印本与代码；类别查询提取局部判别特征。'),
 ('R11','Ridnik, T., et al. ML-Decoder: Scalable and Versatile Classification Head. WACV, 2023; preprint 2021.','https://openaccess.thecvf.com/content/WACV2023/papers/Ridnik_ML-Decoder_Scalable_and_Versatile_Classification_Head_WACV_2023_paper.pdf','会议PDF与作者代码；不要误写为CVPR2022。'),
 ('R12','Ridnik, T., et al. Asymmetric Loss for Multi-Label Classification. ICCV, 2021.','https://arxiv.org/abs/2009.14119','作者论文与代码；正负不对称聚焦。'),
 ('R13','Wu, T., et al. Distribution-Balanced Loss for Multi-Label Classification in Long-Tailed Datasets. ECCV, 2020.','https://arxiv.org/abs/2007.09654','论文与作者代码；共现重平衡、负项正则。'),
 ('R14','Lin, D., et al. Distributionally Robust Loss for Long-Tailed Multi-Label Image Classification. ECCV, 2024.','https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04926.pdf','已核验全文与作者dr_loss.py；类别级排序、负梯度约束、训练测试依赖偏移。'),
 ('R15','Huang, Y., et al. Balancing Methods for Multi-label Text Classification with Long-Tailed Class Distribution. EMNLP, 2021.','https://aclanthology.org/2021.emnlp-main.643/','ACL论文页与作者代码；长尾文本的标签共现。'),
 ('R16','Kang, B., et al. Decoupling Representation and Classifier for Long-Tailed Recognition. ICLR, 2020.','https://arxiv.org/abs/1910.09217','作者论文与代码；原始为单标签任务，需多标签适配。'),
 ('R17','Luo, Y.; Mesgarani, N. Conv-TasNet: Surpassing Ideal Time-Frequency Magnitude Masking for Speech Separation. IEEE/ACM TASLP, 2019.','https://arxiv.org/abs/1809.07454','作者论文、最小实现与Asteroid实现；单通道实语音分离。'),
 ('R18','Pariente, M., et al. Asteroid: the PyTorch-based audio source separation toolkit for researchers. arXiv:2005.04132, 2020.','https://arxiv.org/abs/2005.04132','作者论文与工具仓库；分离模型和复现实验配方。'),
 ('R19','Wisdom, S., et al. Unsupervised Sound Separation Using Mixture Invariant Training. NeurIPS, 2020.','https://research.google/pubs/unsupervised-sound-separation-using-mixture-invariant-training/','作者机构论文页与google-research代码；无干净源条件的混合训练。'),
 ('R20','RF Challenge organisers. Data-Driven Signal Separation in Radio Spectrum: ICASSP 2024 SP Grand Challenge. 2024.','https://signalprocessingsociety.org/publications-resources/data-challenges/data-driven-signal-separation-radio-spectrum-icassp-2024','IEEE SPS与组织者代码；单通道复数RF、目标通信信号恢复。'),
 ('R21','Xiao, Y.; Das, R. K. Dual Knowledge Distillation for Efficient Sound Event Detection. ICASSP 2024 model-compression workshop / arXiv:2402.02781.','https://arxiv.org/abs/2402.02781','作者预印本；蒸馏稳定化参考，不是本项目配对功率蒸馏的直接验证。'),
 ('R22','Cornell, S., et al. DCASE 2024 Task 4: Sound Event Detection with Heterogeneous Data and Missing Labels. DCASE, 2024.','https://arxiv.org/abs/2406.08056','组织者技术论文；弱/强时间标注、缺失标签与mean-teacher。'),
 ('R23','Menon, A. K., et al. Long-tail learning via logit adjustment. ICLR, 2021.','https://openreview.net/pdf?id=37nvvqkCo5','会议论文；原softmax设定，不能直接把结论外推到六类BCE。'),
 ('R24','Cui, Y., et al. Class-Balanced Loss Based on Effective Number of Samples. CVPR, 2019.','https://openaccess.thecvf.com/content_CVPR_2019/html/Cui_Class-Balanced_Loss_Based_on_Effective_Number_of_Samples_CVPR_2019_paper.html','会议论文；有效样本数权重，不直接处理分量遮蔽。')]
for rid,title,url,note in REFS:
 q=p(f'[{rid}] {title}');link(q,' 论文/技术资料',url);p('核验与用途：'+note)

h('本地可追溯证据',2)
for name in ['README.md','doc.md','data/power_truth_audit.json','runs/joint_dfms_seed2050_78d6f2949c/evaluation/development_test/metrics.json','runs/joint_dfms_seed2050_78d6f2949c/history.json','src/longtail/models/dfms.py','src/longtail/engine/trainer.py','src/longtail/weak_metrics.py']:
 p(str(ROOT/name))
p('本报告附件：开源实现核验.json记录固定commit与关键文件哈希；研究证据与诊断.json记录原稿哈希、当前checkpoint/划分哈希和本次错误归因。未新增训练成绩。')

# Harmonize newly added paragraphs and the retained original content.
for para in doc.paragraphs:
 for run in para.runs:
  setfont(run)
for para in doc.paragraphs:
 if re.match(r'^\d+\.\d+\s',para.text) and not para.style.name.startswith('Heading'):
  para.style='Heading 2'
 elif re.match(r'^[1-6]\.\s',para.text) and not para.style.name.startswith('Heading'):
  para.style='Heading 1'

DOCX=OUT/(NAME+'.docx');doc.save(DOCX)

# Build a readable PDF from the same DOCX body. This is a content-equivalent
# export via ReportLab; it is not a claimed Microsoft Word pagination render.
pdfmetrics.registerFont(TTFont('CJK',FONT))
pdfmetrics.registerFont(TTFont('Latin','/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'))
styles={
 'body':ParagraphStyle('body',fontName='CJK',fontSize=9.4,leading=15,spaceAfter=6,wordWrap='CJK'),
 'h1':ParagraphStyle('h1',fontName='CJK',fontSize=15,leading=21,textColor=colors.HexColor('#155E75'),spaceBefore=14,spaceAfter=10,keepWithNext=True,wordWrap='CJK'),
 'h2':ParagraphStyle('h2',fontName='CJK',fontSize=11.5,leading=17,textColor=colors.HexColor('#155E75'),spaceBefore=10,spaceAfter=6,keepWithNext=True,wordWrap='CJK'),
 'title':ParagraphStyle('title',fontName='CJK',fontSize=24,leading=32,textColor=colors.HexColor('#155E75'),spaceAfter=14,wordWrap='CJK'),
 'cell':ParagraphStyle('cell',fontName='CJK',fontSize=8,leading=12,wordWrap='CJK'),
 'cellhead':ParagraphStyle('cellhead',fontName='CJK',fontSize=8.3,leading=12,textColor=colors.white,wordWrap='CJK'),
 'caption':ParagraphStyle('caption',fontName='CJK',fontSize=8,leading=12,textColor=colors.HexColor('#64748B'),spaceAfter=9,wordWrap='CJK')}
from docx.text.paragraph import Paragraph as DParagraph
from docx.table import Table as DTable
def fontify(text):
 parts=[]
 for fragment in re.split(r'([\u2e80-\uffef]+)',text):
  if not fragment:continue
  safe=escape(fragment).replace('\n','<br/>')
  parts.append(safe if ord(fragment[0])>=0x2e80 else '<font name="Latin">'+safe+'</font>')
 return ''.join(parts)
def paragraph_html(para):
 parts=[]
 for item in para.iter_inner_content():
  if item.__class__.__name__=='Hyperlink':parts.append(f'<link href="{escape(item.url)}" color="#16728C">{fontify(item.text)}</link>')
  else:parts.append(fontify(item.text))
 return ''.join(parts)
story=[]
for child in doc.element.body:
 tag=child.tag.rsplit('}',1)[-1]
 if tag=='p':
  para=DParagraph(child,doc)
  if child.xpath('.//w:br[@w:type="page"]'):story.append(PageBreak())
  pics=child.xpath('.//a:blip')
  for blip in pics:
   rel=doc.part.related_parts[blip.get(qn('r:embed'))];target=OUT/('_pdf_image_'+hashlib.sha256(rel.blob).hexdigest()[:10]+'.png');target.write_bytes(rel.blob)
   from PIL import Image as PILImage
   with PILImage.open(target) as im:iw,ih=im.size
   pw=480;story.append(RLImage(str(target),width=pw,height=pw*ih/iw));story.append(Spacer(1,4))
  txt=paragraph_html(para)
  if not txt.strip():
   if not pics:story.append(Spacer(1,3))
   continue
  st=styles['h1'] if para.style.name=='Heading 1' else styles['h2'] if para.style.name.startswith('Heading') else styles['title'] if para.style.name=='Title' else styles['caption'] if para.style.name=='Caption' else styles['body']
  if para.style.name.startswith('List'):txt=fontify('• ')+txt
  story.append(RLParagraph(txt,st))
 elif tag=='tbl':
  tb=DTable(child,doc);n=len(tb.columns);data=[]
  for ri,row in enumerate(tb.rows):
   data.append([RLParagraph(fontify(cell.text),styles['cellhead' if ri==0 else 'cell']) for cell in row.cells])
  if n==2:ws=[125,355]
  elif n==3:ws=[160,160,160]
  elif n==4:ws=[87,127,139,127]
  elif n==5:ws=[100,95,95,95,95]
  else:ws=[480/n]*n
  t=RLTable(data,colWidths=ws,repeatRows=1,hAlign='LEFT')
  t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#155E75')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#F0F6F7')]),('GRID',(0,0),(-1,-1),.35,colors.HexColor('#CBD5E1')),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6)]));story.append(t);story.append(Spacer(1,9))

PDF=OUT/(NAME+'.pdf')
def page(canvas,doc_):
 def mixed(x,y,text):
  for frag in re.split(r'([\u2e80-\uffef]+)',text):
   if not frag:continue
   font='CJK' if ord(frag[0])>=0x2e80 else 'Latin';canvas.setFont(font,8);canvas.drawString(x,y,frag);x+=pdfmetrics.stringWidth(frag,font,8)
 canvas.setFillColor(colors.HexColor('#64748B'));mixed(57,815,'长尾复合干扰识别 | 文献核验与实验建议')
 mixed(57,27,'2026-11-02 增补版 · 阅读版PDF（与Word内容相同，分页独立）');canvas.setFont('Latin',8);canvas.drawRightString(538,27,str(doc_.page))
 canvas.setStrokeColor(colors.HexColor('#CBD5E1'));canvas.line(57,807,538,807)
SimpleDocTemplate(str(PDF),pagesize=(595.28,841.89),rightMargin=57,leftMargin=57,topMargin=47,bottomMargin=46,title='长尾方法综述与试验建议：补充调研版',author='项目研究调研').build(story,onFirstPage=page,onLaterPages=page)
for q in OUT.glob('_pdf_image_*.png'):q.unlink()
(OUT/'文献索引.json').write_text(json.dumps([dict(id=r[0],title=r[1],url=r[2],evidence=r[3],access_date='2026-11-02') for r in REFS],ensure_ascii=False,indent=2))
print(json.dumps({'docx':str(DOCX),'pdf':str(PDF),'references':len(REFS),'repositories':len(repos),'paragraphs':len(doc.paragraphs),'tables':len(doc.tables),'diagnostics':DIAG|{'metrics':'see evidence JSON'}},ensure_ascii=False,indent=2))
