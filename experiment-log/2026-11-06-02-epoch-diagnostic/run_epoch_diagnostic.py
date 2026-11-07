"""Three-seed A-only 150-epoch diagnosis using fit and early-stop roles only."""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, random, subprocess, sys, time
from pathlib import Path

import torch
import numpy as np
from sklearn.metrics import average_precision_score
from torch.nn import functional as F

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('lt_abc_source',HERE.parent/'2026-11-06-01-lt-abc/run_lt_abc.py')
base=importlib.util.module_from_spec(spec); spec.loader.exec_module(base)


def state_hash(model):
    h=hashlib.sha256()
    for k,v in model.state_dict().items():
        h.update(k.encode()); h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def eval_logits(model,x):
    model.eval()
    with torch.no_grad(): return torch.cat([model(b) for b in x.split(64)])


def plot_and_select(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    runs=[json.loads((out/f'A_{s}/history.json').read_text(encoding='utf-8')) for s in (2050,2051,2052)]
    assert all(len(r)==150 for r in runs)
    rows=[r for run in runs for r in run]; base.table(out/'curves.csv',rows)
    a=np.array([[r['early_stop_macro_AP'] for r in run] for run in runs])
    best=np.maximum.accumulate(a,axis=1); target=best[:,-1]
    comparisons=[]
    for n in range(10,151,10):
        current=best[:,n-1]; gap=target-current
        comparisons.append(dict(epoch_budget=n,mean_best_AP=float(current.mean()),std_best_AP=float(current.std(ddof=1)),mean_gap_to_150=float(gap.mean()),max_seed_gap=float(gap.max()),passes=bool(gap.mean()<=.001 and gap.max()<=.002)))
    selected=next(r['epoch_budget'] for r in comparisons if r['passes'])
    base.table(out/'horizon_selection.csv',comparisons)
    checkpoints=[]; summary=[]
    for seed,run in zip((2050,2051,2052),runs):
        for r in run:
            if r['epoch']%10==0: checkpoints.append(r)
        sub=run[:selected]; ap_epoch=1+int(np.argmax([r['early_stop_macro_AP'] for r in sub]))
        summary.append(dict(seed=seed,selected_budget_best_epoch=ap_epoch,peak_AP_epoch_150=1+int(np.argmax([r['early_stop_macro_AP'] for r in run])),min_early_BCE_epoch=1+int(np.argmin([r['early_stop_BCE'] for r in run])),epoch8=run[7],selected_horizon=run[selected-1],epoch150=run[-1],late130_to150_train_BCE_change=run[-1]['train_BCE']-run[129]['train_BCE'],late130_to150_early_BCE_change=run[-1]['early_stop_BCE']-run[129]['early_stop_BCE'],late130_to150_AP_change=run[-1]['early_stop_macro_AP']-run[129]['early_stop_macro_AP']))
    base.table(out/'checkpoints_10_150.csv',checkpoints)
    selection=dict(selected_max_epochs=selected,criterion='mean best-so-far AP gap <= .001 and every seed gap <= .002 versus horizon150; candidates10:10:150',source_roles=['fit_A','early_stop'],seeds=[2050,2051,2052],per_seed=summary,ceiling_only=selected==150,evidence_boundary='A-baseline-only exploratory budget diagnosis; not B/C convergence or new development evidence',calibration_used=False,development_used=False,locked_test_used=False)
    base.write(out/'selection.json',selection)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
    fig,axs=plt.subplots(1,3,figsize=(17,5),constrained_layout=True)
    epochs=np.arange(1,151); colors=['#1873ba','#df7527','#278b60']
    for ax,key,label in zip(axs,['train_BCE','early_stop_BCE','early_stop_macro_AP'],['Training BCE (dropout on)','Early-stop BCE (eval mode)','Early-stop macro-AP']):
        values=np.array([[r[key] for r in run] for run in runs]); scale=100 if key.endswith('AP') else 1; values*=scale
        for seed,v,c in zip((2050,2051,2052),values,colors): ax.plot(epochs,v,color=c,alpha=.28,lw=.8,label=f'seed {seed}')
        mean=values.mean(0); sd=values.std(0,ddof=1); ax.plot(epochs,mean,color='#172c45',lw=1.8,label='3-seed mean');ax.fill_between(epochs,mean-sd,mean+sd,color='#587a9d',alpha=.18,label='mean +/- 1 SD')
        ax.axvline(selected,color='#b73540',ls='--',lw=1.3,label=f'budget {selected}'); ax.set(xlim=(10,150),xlabel='Epoch',ylabel=label+(' (%)' if scale==100 else ''),title=label); ax.set_xticks([10,30,50,70,90,110,130,150]);ax.grid(alpha=.18)
    axs[0].set_yscale('log');axs[0].legend(fontsize=8)
    fig.suptitle(f'LT-ABC A baseline: epochs 10-150 | proposed horizon {selected} | 8000 fit / 1008 early-stop',fontsize=14)
    fig.savefig(out/'loss_curves.png');fig.savefig(out/'loss_curves.pdf');plt.close(fig)
    # Eval-mode fit/early-stop comparison at exactly the requested checkpoints.
    fig,ax=plt.subplots(figsize=(9,5),constrained_layout=True)
    for key,label,color in [('fit_eval_BCE','Fit BCE (eval mode)','#1873ba'),('early_stop_BCE','Early-stop BCE (eval mode)','#df7527')]:
        v=np.array([[r[key] for r in run if r['epoch']%10==0] for run in runs]); ep=np.arange(10,151,10); mean=v.mean(0);sd=v.std(0,ddof=1);ax.plot(ep,mean,'o-',color=color,label=label);ax.fill_between(ep,mean-sd,mean+sd,color=color,alpha=.15)
    ax.axvline(selected,color='#b73540',ls='--',label=f'proposed horizon {selected}'); ax.set(xlabel='Epoch',ylabel='Unweighted BCE',title='Comparable loss: both evaluated with dropout disabled',xlim=(10,150));ax.grid(alpha=.2);ax.legend();fig.savefig(out/'eval_loss_comparison.png');plt.close(fig)
    lines=['# 10–150 epoch training-budget diagnostic','',f'Proposed A-baseline max_epochs: **{selected}**. Save/select the checkpoint with best early-stop macro-AP within this horizon.','',selection['evidence_boundary'],'','The rule compares best-so-far early-stop macro-AP at horizons 10,20,...,150. Tolerances: mean gap <=0.001 AP and each seed gap <=0.002 AP. These are engineering tolerances, not equivalence tests.','','| Seed | Best AP epoch within selected horizon | Best AP epoch within 150 | Minimum early-stop BCE epoch |','|---|---:|---:|---:|']
    for r in summary:lines.append(f'| {r["seed"]} | {r["selected_budget_best_epoch"]} | {r["peak_AP_epoch_150"]} | {r["min_early_BCE_epoch"]} |')
    lines.extend(['','Raw curves: curves.csv. Checkpoints: checkpoints_10_150.csv. Loss figure uses epochs10–150 with three-seed mean +/- sample SD. Training BCE uses dropout on; eval_loss_comparison.png compares fit and early-stop in the same eval mode.','','This diagnostic never loads calibration/development/locked-test waveforms. Prior eight-epoch screening stays unchanged. A/B/C need a common revised-budget experiment before revising their ranking.'])
    (out/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    base.write(out/'status.json',dict(status='completed',selected_max_epochs=selected,calibration_used=False,development_used=False,locked_test_used=False))


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    out=args.output;out.mkdir(parents=True,exist_ok=False)
    pf=json.loads((args.source/'preflight.json').read_text(encoding='utf-8'));m=json.loads((args.source/'manifest.json').read_text(encoding='utf-8'));init=Path(pf['init_path'])
    assert base.sha(init)==base.EXPECTED_INIT
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    device='cuda:0';assert torch.cuda.is_available()
    preflight=dict(source_run=str(args.source),source_manifest_sha256=base.sha(args.source/'manifest.json'),init_sha256=base.sha(init),script_sha256=base.sha(Path(__file__)),imported_script_sha256=base.sha(Path(base.__file__)),plan_sha256=base.sha(HERE/'plan.md'),git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=base.REPO,text=True).strip(),command=sys.argv,epochs=150,seeds=[2050,2051,2052],updates_per_epoch=250,batch=32,lr=.0001,weight_decay=.0001,precision='FP32; TF32 off',roles_accessed=['fit_A','early_stop'],calibration_used=False,development_used=False,locked_test_used=False)
    base.write(out/'preflight.json',preflight)
    base.write(out/'status.json',dict(status='loading_fit_and_early_stop',development_used=False,locked_test_used=False))
    # Validate selected source contents; do not open any other role files.
    assert base.sha(Path(m['fit_path']))==m['fit_sha256']
    assert base.sha(Path(m['dev_path']))==m['dev_sha256']
    fx,fy=base.load(m['fit_path'],m['fit_A']);cx,cy=base.load(m['dev_path'],m['early_stop'])
    fx=fx.to(device);fy=torch.from_numpy(fy).to(device);cx=cx.to(device);cyt=torch.from_numpy(cy).to(device)
    for seed in (2050,2051,2052):
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        model=base.build_backbone('dfms_iq',num_classes=6,width=16,dropout=.2).to(device);base.initialize_f2(model,init)
        ih=state_hash(model);source_receipt=json.loads((args.source/f'A_{seed}/receipt.json').read_text(encoding='utf-8'));assert ih==source_receipt['initial_state_sha256']
        opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4);gen=torch.Generator().manual_seed(seed)
        od=out/f'A_{seed}';od.mkdir();history=[];best=-1;best_epoch=0;start=time.time();orders=[]
        for epoch in range(1,151):
            model.train();order=torch.randperm(8000,generator=gen);oh=hashlib.sha256(order.numpy().tobytes()).hexdigest();orders.append(oh)
            if epoch<=8:assert oh==source_receipt['order_sha256'][epoch-1]
            losses=[]
            for ix in order.to(device).split(32):
                opt.zero_grad(set_to_none=True);z=model(fx[ix]);loss=F.binary_cross_entropy_with_logits(z,fy[ix]);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();losses.append(loss.detach())
            train_loss=float(torch.stack(losses).double().mean().cpu());assert np.isfinite(train_loss)
            z=eval_logits(model,cx);early_loss=float(F.binary_cross_entropy_with_logits(z,cyt).cpu());probs=torch.sigmoid(z).cpu().numpy();aps=average_precision_score(cy,probs,average=None);ap=float(aps.mean())
            fit_loss=None
            if epoch%10==0:
                fz=eval_logits(model,fx);fit_loss=float(F.binary_cross_entropy_with_logits(fz,fy).cpu())
            row=dict(seed=seed,epoch=epoch,train_BCE=train_loss,early_stop_BCE=early_loss,early_stop_macro_AP=ap,fit_eval_BCE=fit_loss,**{f'early_AP_{name}':float(v) for name,v in zip(('FMB','AMA','SawFM','SinFM','MFT','ISRJ'),aps)})
            history.append(row);base.write(od/'history.json',history)
            if ap>best:
                best=ap;best_epoch=epoch;torch.save(dict(model_state_dict=model.state_dict(),epoch=epoch,early_stop_macro_AP=ap),od/'best_AP.pth')
            if epoch%10==0 or epoch==8:
                recovery=dict(model_state_dict=model.state_dict(),optimizer=opt.state_dict(),epoch=epoch,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),order_rng=gen.get_state(),history=history)
                torch.save(recovery,od/'recovery.tmp.pth');(od/'recovery.tmp.pth').replace(od/'recovery.pth')
                if epoch==8:torch.save(recovery,od/'epoch8.pth')
            base.write(out/'status.json',dict(status='training',seed=seed,epoch=epoch,max_epochs=150,seeds_total=3,train_BCE=train_loss,early_stop_BCE=early_loss,early_stop_macro_AP=ap,development_used=False,locked_test_used=False))
            if epoch%10==0 or epoch in (1,8):print(json.dumps(row),flush=True)
        receipt=dict(seed=seed,epochs=150,updates=37500,initial_state_sha256=ih,source_first8_order_hashes_verified=True,order_sha256=orders,best_AP_epoch=best_epoch,best_early_stop_AP=best,best_checkpoint_sha256=base.sha(od/'best_AP.pth'),elapsed_seconds=time.time()-start,development_used=False,locked_test_used=False)
        base.write(od/'receipt.json',receipt)
    plot_and_select(out)


if __name__=='__main__':
    try:main()
    except Exception:
        import traceback
        traceback.print_exc()
        if '--output' in sys.argv:
            out=Path(sys.argv[sys.argv.index('--output')+1]);out.mkdir(parents=True,exist_ok=True);base.write(out/'status.json',dict(status='failed',error=traceback.format_exc(),development_used=False,locked_test_used=False))
        sys.exit(1)
