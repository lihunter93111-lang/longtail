"""Build a static UTF-8 reading page; numeric authority remains result JSON."""
from pathlib import Path
import html

HERE=Path(__file__).resolve().parent
if __name__=='__main__':
    text=(HERE/'RESULTS.txt').read_text(encoding='utf-8')
    paragraphs=text.split('\n\n');parts=[]
    for i,p in enumerate(paragraphs):
        if not p.strip():continue
        first,*rest=p.splitlines()
        if i==0:parts.append('<h1>'+html.escape(first)+'</h1>');continue
        if first.startswith(tuple('一二三四五六七八')) or first=='结论先读':
            parts.append('<h2>'+html.escape(first)+'</h2>');p='\n'.join(rest)
        if 'split1 Exact' in p:
            parts.append('<pre>'+html.escape(p)+'</pre>')
        else:parts.append('<p>'+html.escape(p).replace('\n','<br>')+'</p>')
        if first.startswith('二、'):
            parts.append('<figure><img src="results/figures/01_confirmation_exact.png"><figcaption>同池新确认，五种子均值；主模型+专家为共同仿射。</figcaption></figure>')
        if first.startswith('三、'):
            parts.append('<figure><img src="results/figures/03_label_tradeoffs.png"><figcaption>平均标签代价；平均保护通过不代表逐种子通过。</figcaption></figure>')
    parts.insert(2,'<figure><img src="results/figures/02_paired_differences.png"><figcaption>开发与确认的成对差，区间未做多重比较校正。</figcaption></figure>')
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>2026-11-11 实验结果</title><style>body{max-width:1100px;margin:32px auto;padding:0 24px;color:#17212b;background:#fafbfc;font:16px/1.85 "Microsoft YaHei",sans-serif}h1{font-size:30px}h2{margin-top:40px;font-size:23px;border-bottom:1px solid #cad3dd;padding-bottom:8px}p{white-space:normal}figure{margin:28px 0}img{max-width:100%;background:white}figcaption{color:#596674;font-size:14px}pre{padding:18px;background:white;overflow:auto;font-size:14px;line-height:1.8}nav{padding:14px;background:#e9f0f5}a{color:#126492}@media print{body{max-width:none;font-size:11pt}figure{break-inside:avoid}h2{break-after:avoid}}</style><nav><a href="NEXT_ROUND_PLAN.txt">后续分工</a> · <a href="COLLABORATOR_GUIDE.txt">师弟目录导航</a> · <a href="REPRODUCING.txt">复现说明</a> · <a href="results/confirmation_means.csv">完整指标表</a></nav>'''+''.join(parts)+'</html>'
    (HERE/'RESULTS.html').write_text(page,encoding='utf-8')
    print('READABLE_REPORT_BUILT')
