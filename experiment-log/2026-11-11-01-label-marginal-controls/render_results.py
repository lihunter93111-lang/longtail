"""Render research figures and numeric tables from immutable result JSON."""
from common import *
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    out=HERE/'results/figures';out.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
    data={s:read(ROOT/'analysis'/f'{s}_confirmation.json') for s in CONFIG['splits']}
    methods=['reference','natural','marginal','combo','natural_affine','marginal_affine','combo_affine']
    names=['W32 main','A: natural','B: marginal','C: combo','Main + A','Main + B','Main + C']
    fig,axes=plt.subplots(1,2,figsize=(12.5,4.8),sharey=True,layout='constrained')
    for ax,(split,a) in zip(axes,data.items()):
        x=np.arange(len(methods));values=[a['means'][m]['f1_exact']*100 for m in methods]
        ax.bar(x,values,color=['#697b8e','#8197b0','#4f9d93','#d49442','#8197b0','#4f9d93','#d49442'])
        for i,v in enumerate(values):ax.text(i,v+.09,f'{v:.2f}',ha='center',fontsize=9)
        ax.set_xticks(x,names,rotation=38,ha='right');ax.set_ylim(79,87);ax.set_title(split+' | fresh confirmation')
        ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    axes[0].set_ylabel('Exact match (%) | five-seed mean')
    fig.suptitle('Matched A/B/C controls on W32; Main + expert uses common affine fusion')
    fig.savefig(out/'01_confirmation_exact.png');fig.savefig(out/'01_confirmation_exact.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,5),sharey=True,layout='constrained')
    choices=[('combo','marginal'),('marginal','natural'),('combo','natural'),('combo_affine','marginal_affine'),('marginal_affine','natural_affine'),('combo_affine','natural_affine')]
    names=['Single: C - B','Single: B - A','Single: C - A','Affine: C - B','Affine: B - A','Affine: C - A']
    for ax,(split,a) in zip(axes,data.items()):
        for role,offset,color in [('development',-.12,'#9aa5b1'),('confirmation',.12,'#156d9f')]:
            rows=read(ROOT/'analysis'/f'{split}_{role}.json')['comparisons']
            ci=np.array([next(r['exact_difference'] for r in rows if (r['candidate'],r['reference'])==pair) for pair in choices])*100
            yy=np.arange(len(ci))+offset
            ax.errorbar(ci[:,0],yy,xerr=np.array([ci[:,0]-ci[:,1],ci[:,2]-ci[:,0]]),fmt='o',capsize=3,color=color,label=role)
        ax.axvline(0,color='#333333',lw=.8);ax.set_yticks(np.arange(6),names);ax.invert_yaxis()
        ax.set_title(split);ax.set_xlabel('Paired Exact difference (percentage points)');ax.grid(axis='x',alpha=.2)
    axes[1].legend(loc='upper right');fig.suptitle('Five-seed paired t intervals (95%); unadjusted, no pooling across splits')
    fig.savefig(out/'02_paired_differences.png');fig.savefig(out/'02_paired_differences.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,3.6),layout='constrained')
    for ax,(split,a) in zip(axes,data.items()):
        row=next(r for r in a['comparisons'] if r['candidate']=='combo_affine' and r['reference']=='marginal_affine')
        values=np.array([row['mean_fpr_difference'],row['mean_weak_recall_difference']])*100
        im=ax.imshow(values,cmap='coolwarm',vmin=-3,vmax=3,aspect='auto')
        ax.set_xticks(range(6),LABELS,rotation=25,ha='right');ax.set_yticks([0,1],['FPR delta\nlower is better','Weak recall delta\nhigher is better'])
        ax.set_title(split+' | affine C minus B')
        for i in range(2):
            for j in range(6):ax.text(j,i,f'{values[i,j]:+.2f}',ha='center',va='center',fontsize=10)
    fig.colorbar(im,ax=axes,label='Percentage points',shrink=.75)
    fig.suptitle('Fresh confirmation: mean label trade-offs; individual-seed guard pass rate = 0/5 in both splits')
    fig.savefig(out/'03_label_tradeoffs.png');fig.savefig(out/'03_label_tradeoffs.pdf');plt.close(fig)
    rows=[]
    for split,a in data.items():
        for method,metrics in a['means'].items():rows.append({'split':split,'method':method,**metrics})
    path=HERE/'results/confirmation_means.csv';path.parent.mkdir(exist_ok=True)
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print('FIGURES_RENDERED',flush=True)

if __name__=='__main__':main()
