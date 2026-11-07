"""Render only measured epochs 10–150; keep their axes independent of epochs1–9."""
import argparse, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args();out=args.output
sel=json.loads((out/'selection.json').read_text(encoding='utf-8'));budget=sel['selected_max_epochs']
runs=[json.loads((out/f'A_{s}/history.json').read_text(encoding='utf-8')) for s in (2050,2051,2052)]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
colors=['#1873ba','#df7527','#278b60'];ep=np.arange(10,151)
fig,axs=plt.subplots(1,3,figsize=(17,5),constrained_layout=True)
for ax,key,title in zip(axs,['train_BCE','early_stop_BCE','early_stop_macro_AP'],['Training BCE (dropout on)','Early-stop BCE (eval mode)','Early-stop macro-AP']):
    v=np.array([[r[key] for r in run[9:]] for run in runs]);scale=100 if key.endswith('AP') else 1;v*=scale
    for seed,values,c in zip((2050,2051,2052),v,colors):ax.plot(ep,values,color=c,lw=.8,alpha=.30,label=f'seed {seed}')
    mean=v.mean(0);sd=v.std(0,ddof=1);ax.plot(ep,mean,color='#172c45',lw=1.7,label='3-seed mean');ax.fill_between(ep,mean-sd,mean+sd,color='#587a9d',alpha=.18,label='mean +/- 1 SD')
    ax.axvline(budget,color='#b73540',ls='--',lw=1.3,label=f'budget {budget}');ax.set(xlim=(10,150),xlabel='Epoch',ylabel=title+(' (%)' if scale==100 else ''),title=title);ax.set_xticks([10,30,50,70,90,110,130,150]);ax.grid(alpha=.18)
axs[0].set_yscale('log');axs[0].legend(fontsize=8)
fig.suptitle(f'A-baseline diagnosis: epochs 10-150 | proposed budget {budget} | 3 seeds | 8000 fit / 1008 early-stop',fontsize=14)
fig.savefig(out/'loss_curves.png');fig.savefig(out/'loss_curves.pdf');plt.close(fig)
fig,ax=plt.subplots(figsize=(9,5),constrained_layout=True)
for key,label,c in [('fit_eval_BCE','Fit BCE (eval mode)','#1873ba'),('early_stop_BCE','Early-stop BCE (eval mode)','#df7527')]:
    v=np.array([[r[key] for r in run if r['epoch']%10==0] for run in runs]);e=np.arange(10,151,10);mean=v.mean(0);sd=v.std(0,ddof=1)
    ax.plot(e,mean,'o-',color=c,label=label);ax.fill_between(e,np.maximum(mean-sd,0),mean+sd,color=c,alpha=.15)
ax.axvline(budget,color='#b73540',ls='--',label=f'proposed budget {budget}');ax.set(xlabel='Epoch',ylabel='Unweighted BCE',title='Fit vs. early-stop loss: 3-seed mean +/- SD (eval mode)',xlim=(9,151),ylim=(0,None));ax.set_xticks([10,30,50,70,90,110,130,150]);ax.grid(alpha=.2);ax.legend();fig.savefig(out/'eval_loss_comparison.png');plt.close(fig)
