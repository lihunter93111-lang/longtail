"""Standalone scientific figures; uncertainties distinguish seeds and identities."""
from common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def figures():
    output=HERE/'figures';output.mkdir(exist_ok=True);plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
    roles=[r for r in ['development','confirmation'] if all((ROOT/'analysis'/f'{s}_{r}.json').exists() for s in CONFIG['splits'])]
    for role in roles:
        fig,axes=plt.subplots(1,2,figsize=(11,5),sharey=True,sharex=True)
        names=[('marginal_segment','marginal_global','B: segment - global'),('marginal_segment','marginal_none','B: segment - none'),
               ('combo_segment','combo_global','C: segment - global'),('combo_segment','combo_none','C: segment - none'),
               ('marginal_segment_guarded','marginal_none','B: guarded segment - none'),('combo_segment_guarded','combo_none','C: guarded segment - none')]
        for ax,split in zip(axes,CONFIG['splits']):
            data=read(ROOT/'analysis'/f'{split}_{role}.json');lookup={(r['candidate'],r['reference']):r for r in data['comparisons']}
            for i,(a,b,label) in enumerate(names):
                row=lookup[a,b];mean,low,high=np.asarray(row['exact_difference'])*100;ax.errorbar(mean,i,xerr=[[mean-low],[high-mean]],fmt='o',capsize=3,color='#2463a5')
                ax.scatter(np.asarray(row['seed_differences'])*100,np.full(5,i)+np.linspace(-.1,.1,5),s=12,color='#a9bfd4',alpha=.85)
            ax.axvline(0,color='#777',lw=1,ls='--');ax.set_yticks(range(len(names)),[n[2] for n in names]);ax.set_title(split);ax.set_xlabel('Exact-match difference (percentage points)');ax.grid(axis='x',alpha=.2)
        axes[0].invert_yaxis()
        fig.suptitle('Expert x readout: '+role+'\nMean and unadjusted 95% paired-seed t interval; dots = five seeds',fontsize=12);fig.tight_layout()
        for suffix in ['png','pdf']:fig.savefig(output/f'factorial_{role}.{suffix}',bbox_inches='tight')
        plt.close(fig)
        fig,axes=plt.subplots(1,2,figsize=(10,4.6),sharey=True)
        names=[('marginal_segment','marginal_none','B segment'),('combo_segment','combo_none','C segment'),('marginal_segment_guarded','marginal_none','B guarded'),('combo_segment_guarded','combo_none','C guarded')]
        for split,color,offset in [('split1','#2463a5',-.1),('split2','#d17d22',.1)]:
            path=ROOT/split/('physical_'+role+'_results')/'summary.json'
            if not path.exists():continue
            data=read(path);lookup={(r['candidate'],r['reference']):r for r in data['comparisons']}
            for ax,key,title in zip(axes,['gap15_TPR_difference','absent_FPR_difference'],['Target recall at 15 dB gap','False positives after target removal']):
                for i,(a,b,label) in enumerate(names):
                    m,l,h=np.array(lookup[a,b][key])*100;ax.errorbar(m,i+offset,xerr=[[m-l],[h-m]],fmt='o',capsize=3,color=color,label=split if i==0 else None)
                ax.set_title(title);ax.axvline(0,color='#777',lw=1,ls='--');ax.set_yticks(range(len(names)),[n[2]+' - none' for n in names]);ax.set_xlabel('Difference (percentage points)');ax.grid(axis='x',alpha=.2)
        axes[0].invert_yaxis();axes[0].legend(loc='best');fig.suptitle('Paired physical diagnostics: '+role+'\nFixed background/noise; seed intervals are unadjusted',fontsize=12);fig.tight_layout()
        for suffix in ['png','pdf']:fig.savefig(output/f'physical_{role}.{suffix}',bbox_inches='tight')
        plt.close(fig)
        path=ROOT/'analysis'/f'background_diagnostics_{role}.json'
        if path.exists():
            data=read(path);fig,axes=plt.subplots(1,2,figsize=(11,4.5),sharey=True,sharex=True)
            groups=['tail','middle','head']
            contrasts=[('combo_none','marginal_none','C - B, no readout','#b96919',-.10),
                       ('combo_segment','combo_none','Segment - none, C expert','#2463a5',.10)]
            for ax,split in zip(axes,data['splits']):
                lookup={(r['candidate'],r['reference']):r for r in split['comparisons']}
                for candidate,reference,label,color,offset in contrasts:
                    for i,name in enumerate(groups):
                        cell=lookup[candidate,reference]['groups'][name]
                        if not cell['n']:continue
                        row=cell['difference'];mean=100*row['mean'];lo,hi=np.asarray(row['ci95'])*100
                        ax.errorbar(mean,i+offset,xerr=[[mean-lo],[hi-mean]],fmt='o',capsize=3,color=color,label=label if i==0 else None)
                ax.axvline(0,color='#777',ls='--',lw=1);ax.set_title(split['split']);ax.set_yticks(range(3),['Tail (fit count <= 50)','Middle (51-249)','Head (>= 250)'])
                ax.set_xlabel('Exact-match difference (pp)');ax.grid(axis='x',alpha=.2)
            axes[0].invert_yaxis();axes[0].legend(loc='best',fontsize=8)
            fig.suptitle('Where the increments occur: '+role+'\nActual historical fit quotas; five-seed means and unadjusted 95% t intervals',fontsize=12);fig.tight_layout()
            for suffix in ['png','pdf']:fig.savefig(output/f'backgrounds_{role}.{suffix}',bbox_inches='tight')
            plt.close(fig)
        path=ROOT/'analysis'/f'matrix_{role}.json'
        if path.exists():
            data=read(path)
            if data['cells']:
                names=list(read(HERE/'matrix_protocol.json')['cells']);fig,axes=plt.subplots(1,2,figsize=(11,5),sharey=True)
                for ax,(a,b) in zip(axes,[('combo','marginal'),('marginal','natural')]):
                    for split,color,offset in [('split1','#2463a5',-.1),('split2','#d17d22',.1)]:
                        lookup={r['cell']:r for r in data['comparisons'] if r['split']==split and r['candidate']==a and r['reference']==b}
                        for i,cell in enumerate(names):
                            if cell not in lookup:continue
                            row=lookup[cell]['differences']['f1_exact'];mean=100*row['mean'];bounds=row['ci95']
                            err=None if bounds is None else [[mean-100*bounds[0]],[100*bounds[1]-mean]]
                            ax.errorbar(mean,i+offset,xerr=err,fmt='o',capsize=3,color=color,label=split if i==0 else None)
                            if row['n_seeds']>1:
                                ax.scatter(100*np.asarray(row['values']),np.full(row['n_seeds'],i+offset)+np.linspace(-.035,.035,row['n_seeds']),s=18,color=color,alpha=.4,zorder=3)
                            ax.annotate('n='+str(row['n_seeds']),(mean,i+offset),xytext=(4,5),textcoords='offset points',fontsize=8,color=color)
                    ax.axvline(0,color='#777',ls='--',lw=1);ax.set_yticks(range(len(names)),names);ax.set_title(a+' - '+b);ax.set_xlabel('Exact-match difference (pp)');ax.grid(axis='x',alpha=.2)
                axes[0].invert_yaxis();axes[0].legend();fig.suptitle('Matched A/B/C quota controls: '+role+'\nMeans and unadjusted 95% seed t intervals; small dots = seeds; no interval for n=1',fontsize=12);fig.tight_layout()
                for suffix in ['png','pdf']:fig.savefig(output/f'matrix_{role}.{suffix}',bbox_inches='tight')
                plt.close(fig)
    print('FIGURES_WRITTEN',roles,flush=True)

if __name__=='__main__':figures()
