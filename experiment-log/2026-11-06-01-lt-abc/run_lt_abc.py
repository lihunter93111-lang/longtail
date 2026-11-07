"""LT-ABC-v1 preparation and paired training; never opens a locked test.

Fresh candidate generation is a separate MATLAB step. This entry point waits
for its two atomic receipts, audits candidates, then trains and calibrates all
nine models. Development scoring is deliberately a separate stage.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, platform, random, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path

import torch  # Load CUDA/OpenMP DLLs before SciPy/sklearn on this Windows host.
import h5py
import numpy as np
import scipy
import sklearn
from sklearn.metrics import average_precision_score
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / 'projects/CJ_Imbalance_0716'))
from cjlt.models.backbone_adapter import build_backbone
from experiments.run_route_a_c1_baseline import initialize_f2

EXPECTED_INIT = '4390ae3cf88c0ac023e399d2bfe6f8357093ef8d23c73fda9c7f4bc78a47e190'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''): h.update(b)
    return h.hexdigest()


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def table(path, rows):
    with Path(path).open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def quotas(q):
    a,b = np.array(q['A_counts']),np.array(q['B_counts'])
    bits = ((np.arange(1,64)[:,None] >> np.arange(6)) & 1)
    assert q['code_order'] == list(range(1,64))
    assert q['C_counts'] == a.tolist() and a.sum() == b.sum() == 8000
    assert np.minimum(a,b).sum() == 7844
    for v in (a,b):
        assert (v > 0).all() and (v @ bits).tolist() == q['margins']
        assert [int(v[bits.sum(1)==k].sum()) for k in range(1,7)] == q['cardinality_counts']
    assert len(set(q['tail_codes']))==13
    assert all(a[c-1]==41 and b[c-1]==53 for c in q['tail_codes'])
    p=a@bits/8000
    wp=np.clip(.5+.25/p,.5,3); wn=np.clip(.5+.25/(1-p),.5,3)
    d=p*wp+(1-p)*wn; wp/=d; wn/=d
    np.testing.assert_allclose(wp,q['w_positive']); np.testing.assert_allclose(wn,q['w_negative'])
    np.testing.assert_allclose(p*wp+(1-p)*wn,1)
    return a,b,wp,wn


def smoke(q, init, device):
    _,_,wp,wn=quotas(q)
    torch.manual_seed(2050)
    model=build_backbone('dfms_iq',num_classes=6,width=16,dropout=.2).to(device)
    initialize_f2(model,init)
    x=torch.randn(2,2,3000,device=device); y=torch.tensor([[1,0,0,0,0,0],[0,1,1,0,0,1]],device=device).float()
    z=model(x); assert z.shape==(2,6) and torch.isfinite(z).all()
    raw=F.binary_cross_entropy_with_logits(z,y,reduction='none')
    torch.testing.assert_close(raw.mean(),F.binary_cross_entropy_with_logits(z,y))
    torch.testing.assert_close((raw*torch.ones_like(y)).mean(),raw.mean())
    w=y*torch.tensor(wp,device=device)+(1-y)*torch.tensor(wn,device=device)
    loss=(raw*w).mean(); assert torch.isfinite(loss)
    loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4); opt.step()
    import io
    buffer=io.BytesIO(); torch.save(model.state_dict(),buffer); buffer.seek(0)
    model.eval()
    with torch.no_grad(): before=model(x).detach().clone()
    model.load_state_dict(torch.load(buffer,map_location=device,weights_only=False))
    with torch.no_grad(): torch.testing.assert_close(model(x),before,rtol=0,atol=0)
    pred=np.array([[1,0,0,0,0,0],[0,1,0,0,0,1]])
    assert (pred==y.cpu().numpy()).all(1).mean()==.5
    return {'batch_loss':float(loss.detach()),'shape':[2,6],'exact_toy':.5,'passed':True}


def candidates(root, role, labels):
    rec=json.loads((root/role/'generation_receipt.json').read_text(encoding='utf-8'))
    path=Path(rec['path']); rows=json.loads((root/role/'generation_manifest.json').read_text(encoding='utf-8'))
    with h5py.File(path,'r') as f:
        y=np.asarray(f['known_label'],dtype=np.uint8).T
        assert y.shape==(len(rows),6)
        noise=np.asarray(f['receiver_noise_seed']).reshape(-1).astype(np.int64)
        power=np.full((len(y),6),np.nan); components=[]
        for i,r in enumerate(rows):
            names=r['active_type_codes']; names=[names] if isinstance(names,str) else names
            assert names==[labels[j] for j in np.flatnonzero(y[i])]
            v=np.asarray(f[f['component_nominal_JNR_dB'][0,i]]).reshape(-1)
            s=np.asarray(f[f['component_seeds'][0,i]]).reshape(-1).astype(np.int64)
            assert len(v)==len(names)==len(s) and np.isfinite(v).all() and ((v>=10)&(v<=30)).all()
            power[i,[labels.index(n) for n in names]]=v
            components.append(s.tolist())
    assert len(set(noise))==len(noise)
    seeds=np.array([int(r['sample_seed']) for r in rows])
    assert len(set(seeds))==len(seeds)
    codes=y@(1<<np.arange(6))
    return dict(path=path,sha256=sha(path),rows=rows,y=y,noise=noise,power=power,components=components,seeds=seeds,codes=codes)


def audit_power(pool, a, b):
    out=[]
    for j in range(6):
        av=pool['power'][a,j]; bv=pool['power'][b,j]; av=av[np.isfinite(av)]; bv=bv[np.isfinite(bv)]
        ah=np.histogram(av,[10,15,20,25,30])[0]/len(av)
        bh=np.histogram(bv,[10,15,20,25,30])[0]/len(bv)
        out.append(dict(label=j,max_bin_difference=float(abs(ah-bh).max()),mean_difference=float(abs(av.mean()-bv.mean())),A_mean=float(av.mean()),B_mean=float(bv.mean()),A_hist=json.dumps(ah.tolist()),B_hist=json.dumps(bh.tolist())))
    return out


def prepare(args,q):
    a,b,_,_=quotas(q); fit=candidates(args.output,'fit_pool',q['labels']); dev=candidates(args.output,'dev_pool',q['labels'])
    assert not set(fit['seeds'])&set(dev['seeds']) and not set(fit['noise'])&set(dev['noise'])
    assert not {tuple(s) for s in fit['components']}&{tuple(s) for s in dev['components']}
    # Fresh seed ranges must also be disjoint from both historical raw sources.
    history=[]
    for path in args.historical_pool:
        with h5py.File(path,'r') as f: old=np.asarray(f['receiver_noise_seed']).reshape(-1).astype(np.int64)
        overlap=len(set(old)&(set(fit['noise'])|set(dev['noise'])))
        history.append(dict(path=str(path),sha256=sha(path),noise_overlap=overlap))
        assert overlap==0, 'fresh candidate identity overlaps historical raw pool'
    rng=np.random.default_rng(q['fit_data_seed']); attempts=[]
    for attempt in range(100):
        ai=[]; bi=[]; supply=[]
        for code,(na,nb) in enumerate(zip(a,b),1):
            ids=np.flatnonzero(fit['codes']==code)
            assert len(ids)>=max(na,nb),f'insufficient code {code}'
            # Stratify within code by mean active-component JNR, proportional bins.
            mean=np.nanmean(fit['power'][ids],axis=1)
            bins=np.minimum(((mean-10)//5).astype(int),3)
            n=int(max(na,nb)); groups=[rng.permutation(ids[bins==k]) for k in range(4)]
            shares=np.array([len(g) for g in groups])*n/len(ids)
            take=np.floor(shares).astype(int)
            for k in np.argsort(-(shares-take),kind='stable')[:n-take.sum()]: take[k]+=1
            chosen=np.concatenate([g[:k] for g,k in zip(groups,take)])
            chosen=rng.permutation(chosen)
            core=chosen[:min(na,nb)].tolist(); extra=chosen[min(na,nb):].tolist()
            ai.extend(core+(extra if na>nb else [])); bi.extend(core+(extra if nb>na else []))
            supply.append(dict(code=code,available=len(ids),A_target=int(na),B_target=int(nb),A_actual=int(na),B_actual=int(nb),shared=int(min(na,nb))))
        # Keep shared identities at exactly the same positions. Map sorted new IDs.
        ai=np.array(ai); only_a=sorted(set(ai)-set(bi)); only_b=sorted(set(bi)-set(ai)); mapping=dict(zip(only_a,only_b))
        bi=np.array([mapping.get(int(i),int(i)) for i in ai])
        power=audit_power(fit,ai,bi); ok=all(r['max_bin_difference']<=.01 and r['mean_difference']<=.25 for r in power)
        attempts.append(dict(attempt=attempt,passed=ok,power=power))
        if ok: break
    write(args.output/'power_selection_attempts.json',attempts)
    assert ok, 'power constraints infeasible after 100 predetermined attempts; stop B'
    assert len(set(ai))==len(set(bi))==8000 and len(set(ai)&set(bi))==7844
    for idx,counts in ((ai,a),(bi,b)):
        assert np.bincount(fit['codes'][idx],minlength=64)[1:].tolist()==counts.tolist()
    rng=np.random.default_rng(q['role_seed']); roles={'early_stop':[],'calibration':[],'development':[]}
    for code in range(1,64):
        ix=rng.permutation(np.flatnonzero(dev['codes']==code)); assert len(ix)>=96
        roles['early_stop']+=ix[:16].tolist(); roles['calibration']+=ix[16:32].tolist(); roles['development']+=ix[32:96].tolist()
    m={'fit_A':ai.tolist(),'fit_B':bi.tolist(),**roles,'fit_path':str(fit['path']),'dev_path':str(dev['path']),'fit_sha256':fit['sha256'],'dev_sha256':dev['sha256'],'historical_sources':history,'locked_test_used':False}
    write(args.output/'manifest.json',m); table(args.output/'quota_audit.csv',supply); table(args.output/'power_audit.csv',power)
    replacement=audit_power(fit,np.array(only_a),np.array(only_b)); table(args.output/'replacement_power_audit.csv',replacement)
    for name,pool,idx in [('fit_A',fit,ai),('fit_B',fit,bi)]+[(r,dev,idx) for r,idx in roles.items()]:
        records=[]
        for i in idx:
            i=int(i); records.append(dict(raw_row=i,source_file_hash=pool['sha256'],sample_seed=int(pool['seeds'][i]),receiver_noise_seed=int(pool['noise'][i]),component_seeds=pool['components'][i],code=int(pool['codes'][i]),component_JNR_dB=[None if not np.isfinite(v) else float(v) for v in pool['power'][i]],generation_batch=str(pool['path'].parent),generation_manifest_row=i))
        write(args.output/(name+'_identities.json'),records)
        yy=pool['y'][idx].astype(int)
        write(args.output/(name+'_distribution.json'),dict(label_positive=yy.sum(0).tolist(),label_negative=(len(yy)-yy.sum(0)).tolist(),cooccurrence=(yy.T@yy).tolist(),cardinality=np.bincount(yy.sum(1),minlength=7).tolist()))
    return m


def load(path,idx):
    idx=np.array(idx); order=np.argsort(idx); inverse=np.argsort(order)
    with h5py.File(path,'r') as f:
        y=np.asarray(f['known_label'][:,idx[order]],dtype=np.float32).T[inverse]
        z=np.asarray(f['iq_complex'][:,idx[order]])
    x=np.stack([z['real'].T,z['imag'].T],1).astype(np.float32)[inverse]
    x/=np.sqrt(np.mean(x[:,0]**2+x[:,1]**2,axis=1)+1e-12)[:,None,None]
    assert x.shape==(len(y),2,3000) and np.isfinite(x).all()
    return torch.from_numpy(np.ascontiguousarray(x)),y


def predict(model,x,device):
    model.eval(); out=[]
    with torch.no_grad():
        for batch in x.split(64): out.append(model(batch.to(device)).cpu().numpy())
    return np.concatenate(out)


def thresholds(y,z):
    p=(1/(1+np.exp(-z))).astype(np.float64); result=[]
    for j in range(6):
        scores=[]
        for t in np.linspace(.05,.95,181):
            pred=p[:,j]>=t; tp=(pred & (y[:,j]==1)).sum()
            scores.append(2*tp/max(pred.sum()+y[:,j].sum(),1))
        result.append(float(np.linspace(.05,.95,181)[np.argmax(scores)]))
    neg=p[y[:,1]==0,1]; candidates=np.r_[np.unique(neg),np.nextafter(neg.max(),np.inf)]
    t=float(next(t for t in candidates if (neg>=t).mean()<=.05))
    return dict(f1=result,ama_fpr05=t)


def train(args,q,m,device):
    _,_,wp,wn=quotas(q); cx,cy=load(m['dev_path'],m['early_stop']); kx,ky=load(m['dev_path'],m['calibration'])
    xs={}; ys={}
    for arm in 'AB': xs[arm],ys[arm]=load(m['fit_path'],m['fit_'+arm])
    xs['C']=xs['A']; ys['C']=ys['A']; receipts=[]
    for seed in q['seed_screen']:
        initial_hash=None
        for arm in 'ABC':
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
            model=build_backbone('dfms_iq',num_classes=6,width=16,dropout=.2).to(device); initialize_f2(model,args.init)
            h=hashlib.sha256()
            for k,v in model.state_dict().items(): h.update(k.encode()); h.update(v.detach().cpu().numpy().tobytes())
            ih=h.hexdigest(); initial_hash=initial_hash or ih; assert ih==initial_hash
            od=args.output/f'{arm}_{seed}'; od.mkdir(); ck=od/'best.pth'
            opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
            best=-1; best_ep=0; history=[]; start=time.time(); order_hashes=[]
            # A/C use the same independent order RNG; dropout RNG is unaffected.
            gen=torch.Generator().manual_seed(seed)
            for ep in range(1,9):
                model.train(); total=0.; order=torch.randperm(8000,generator=gen); order_hashes.append(hashlib.sha256(order.numpy().tobytes()).hexdigest())
                for ix in order.split(32):
                    x=xs[arm][ix].to(device); y=torch.from_numpy(ys[arm][ix.numpy()]).to(device)
                    opt.zero_grad(set_to_none=True); z=model(x); raw=F.binary_cross_entropy_with_logits(z,y,reduction='none')
                    if arm=='C': raw=raw*(y*torch.tensor(wp,dtype=torch.float32,device=device)+(1-y)*torch.tensor(wn,dtype=torch.float32,device=device))
                    loss=raw.mean(); assert torch.isfinite(loss); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5); opt.step(); total+=float(loss.detach())
                z=predict(model,cx,device); ap=float(average_precision_score(cy,1/(1+np.exp(-z)),average='macro'))
                row=dict(arm=arm,seed=seed,epoch=ep,loss=total/250,early_stop_macro_AP=ap); history.append(row); print(json.dumps(row),flush=True)
                if ap>best:
                    best=ap; best_ep=ep; torch.save({'model_state_dict':model.state_dict(),'epoch':ep},ck)
                write(od/'history.json',history)
                write(args.output/'status.json',dict(status='training',**row,locked_test_used=False))
            state=torch.load(ck,map_location=device,weights_only=False); model.load_state_dict(state['model_state_dict'])
            # Checkpoint reload must exactly reproduce its selected validation AP.
            rz=predict(model,cx,device); assert abs(float(average_precision_score(cy,1/(1+np.exp(-rz)),average='macro'))-best)<1e-7
            kz=predict(model,kx,device); th=thresholds(ky,kz); write(od/'thresholds.json',th)
            np.savez_compressed(od/'calibration.npz',logits=kz,labels=ky)
            receipt=dict(arm=arm,seed=seed,best_epoch=best_ep,initial_state_sha256=ih,checkpoint_sha256=sha(ck),manifest_sha256=sha(args.output/'manifest.json'),protocol_sha256=sha(args.protocol),code_sha256=sha(Path(__file__)),init_sha256=sha(args.init),elapsed_seconds=time.time()-start,updates=2000,epochs=8,order_sha256=order_hashes,thresholds=th,locked_test_used=False)
            write(od/'receipt.json',receipt); receipts.append(receipt)
        paired=receipts[-3:]; assert paired[0]['order_sha256']==paired[1]['order_sha256']==paired[2]['order_sha256']
    write(args.output/'runs.json',receipts)
    table(args.output/'runs.csv',[{k:r[k] for k in ('arm','seed','best_epoch','initial_state_sha256','checkpoint_sha256','elapsed_seconds','updates')} for r in receipts])
    write(args.output/'thresholds.json',{f'{r["arm"]}_{r["seed"]}':r['thresholds'] for r in receipts})
    write(args.output/'status.json',dict(status='all_nine_models_and_thresholds_frozen',development_scored=False,locked_test_used=False))


def score(args,q,m,device):
    runs=json.loads((args.output/'runs.json').read_text(encoding='utf-8'))
    assert len(runs)==9 and {(r['arm'],r['seed']) for r in runs}=={(a,s) for a in 'ABC' for s in q['seed_screen']}
    x,y=load(m['dev_path'],m['development']); code=y.astype(int)@(1<<np.arange(6)); result={}; rows=[]; slices=[]; perlabel=[]
    a=np.array(q['A_counts']); b=np.array(q['B_counts'])
    groups={'overall':list(range(1,64)),'target':q['tail_codes'],'all_min':q['all_min_codes'],'remaining_min':sorted(set(q['all_min_codes'])-set(q['tail_codes'])),'donor':(np.flatnonzero(b<a)+1).tolist(),'common':q['common_codes']}
    assert len(groups['common'])==13
    for r in runs:
        arm,seed=r['arm'],r['seed']; od=args.output/f'{arm}_{seed}'
        assert sha(od/'best.pth')==r['checkpoint_sha256'] and sha(args.output/'manifest.json')==r['manifest_sha256']
        model=build_backbone('dfms_iq',num_classes=6,width=16,dropout=.2).to(device)
        model.load_state_dict(torch.load(od/'best.pth',map_location=device,weights_only=False)['model_state_dict'])
        z=predict(model,x,device); p=(1/(1+np.exp(-z))).astype(np.float64); pred=p>=np.array(r['thresholds']['f1']); exact=(pred==y).all(1)
        np.savez_compressed(od/'development.npz',logits=z,labels=y,predictions=pred,code=code,raw_rows=np.array(m['development']),seed=seed,role='development')
        metrics={g:float(np.mean([exact[code==c].mean() for c in codes])) for g,codes in groups.items()}
        ap=average_precision_score(y,p,average=None); fpr=((pred==1)&(y==0)).sum(0)/(y==0).sum(0)
        recall=((pred==1)&(y==1)).sum(0)/(y==1).sum(0)
        metrics.update(AMA_AP=float(ap[1]),FMB_FPR=float(fpr[0]),macro_AP=float(ap.mean()),one_bit_error=float(((pred!=y).sum(1)==1).mean()),fixed05_exact=float(((p>=.5)==y).all(1).mean()),reference_weighted_exact=float(np.average([exact[code==c].mean() for c in range(1,64)],weights=a)))
        ama=p[:,1]>=r['thresholds']['ama_fpr05']; metrics['AMA_recall_fpr05']=float(ama[y[:,1]==1].mean()); metrics['AMA_actual_fpr05']=float(ama[y[:,1]==0].mean())
        tp=((pred==1)&(y==1)).sum(0); fp=((pred==1)&(y==0)).sum(0); fn=((pred==0)&(y==1)).sum(0)
        metrics['macro_F1']=float((2*tp/np.maximum(2*tp+fp+fn,1)).mean())
        for j,label in enumerate(q['labels']): perlabel.append(dict(arm=arm,seed=seed,label=label,AP=float(ap[j]),recall=float(recall[j]),FPR=float(fpr[j]),false_positive=int(fp[j]),false_negative=int(fn[j])))
        for c in range(1,64): slices.append(dict(arm=arm,seed=seed,code=c,n=int((code==c).sum()),calibrated_exact=float(exact[code==c].mean()),fixed05_exact=float(((p[code==c]>=.5)==y[code==c]).all(1).mean())))
        result[arm,seed]=dict(exact=exact,p=p,metrics=metrics); rows.append(dict(arm=arm,seed=seed,**metrics))
    table(args.output/'development_metrics.csv',rows); table(args.output/'development_combinations.csv',slices); table(args.output/'development_labels.csv',perlabel)
    diffs=[]
    for arm in 'BC':
        for seed in q['seed_screen']:
            diffs.append(dict(comparison=arm+'-A',seed=seed,**{k:100*(v-result['A',seed]['metrics'][k]) for k,v in result[arm,seed]['metrics'].items()}))
    table(args.output/'paired_differences_pp.csv',diffs)
    decisions={}
    for arm in 'BC':
        dd=[r for r in diffs if r['comparison']==arm+'-A']; mean={k:float(np.mean([r[k] for r in dd])) for k in result['A',q['seed_screen'][0]]['metrics']}
        sd={k:float(np.std([r[k] for r in dd],ddof=1)) for k in mean}; primary='target' if arm=='B' else 'AMA_AP'; positives=sum(r[primary]>0 for r in dd)
        checks={'primary_mean':mean[primary]>=(1. if arm=='B' else .2),'positive_seeds':positives>=2,'overall':mean['overall']>=-.5}
        if arm=='B': checks.update(common=mean['common']>=-1,donor=mean['donor']>=-1)
        else: checks.update(FMB_FPR=mean['FMB_FPR']<=1)
        decisions[arm]=dict(paired_mean_pp=mean,paired_std_pp=sd,positive_seeds=positives,checks=checks,eligible_for_confirmation=all(checks.values()),maximum_degradation_pp={k:float(max(r[k] for r in dd) if k in ('FMB_FPR','AMA_actual_fpr05','one_bit_error') else min(r[k] for r in dd)) for k in mean})
    # Same identity resample for all seeds/models. AP bootstrap uses a weighted
    # sorted precision computation, equivalent to sklearn sample_weight.
    rng=np.random.default_rng(20261110); ix_by_code=[np.flatnonzero(code==c) for c in range(1,64)]
    cache={}
    for key,v in result.items():
        order=np.argsort(-v['p'][:,1],kind='stable'); scores=v['p'][order,1]
        ends=np.r_[np.flatnonzero(np.diff(scores)),len(scores)-1]
        cache[key]=(order,ends,y[order,1])
    boot={'B':[],'C':[]}; target=np.isin(code,q['tail_codes'])
    for iteration in range(10000):
        ids=np.concatenate([rng.choice(ix,len(ix),replace=True) for ix in ix_by_code]); weights=np.bincount(ids,minlength=len(y))
        bd=[]; cd=[]
        for seed in q['seed_screen']:
            bd.append(np.average((result['B',seed]['exact'].astype(float)-result['A',seed]['exact'])[target],weights=weights[target]))
            aps=[]
            for arm in 'AC':
                order,ends,truth=cache[arm,seed]; w=weights[order]; tp=np.cumsum(w*truth)[ends]; total=np.cumsum(w)[ends]
                aps.append(float(np.sum(np.diff(np.r_[0,tp])*np.divide(tp,total,out=np.zeros_like(tp),where=total>0))/tp[-1]))
            cd.append(aps[1]-aps[0])
        boot['B'].append(float(np.mean(bd))*100); boot['C'].append(float(np.mean(cd))*100)
        if iteration%1000==0: print(f'BOOTSTRAP {iteration}/10000',flush=True)
    for arm in 'BC': decisions[arm]['identity_bootstrap_primary_95ci_pp']=np.percentile(boot[arm],[2.5,97.5]).tolist()
    write(args.output/'decision.json',decisions)
    lines=['# LT-ABC-v1 development screening','', 'Units: absolute percentage points. Identity bootstrap intervals describe development identities, not model seed uncertainty. Locked test used: false.','']
    for arm,d in decisions.items():
        lines.extend([f'## {arm} − A',f'Eligible for independent confirmation: {d["eligible_for_confirmation"]}',f'Checks: {d["checks"]}',f'Mean paired changes: {d["paired_mean_pp"]}',f'Seed standard deviations: {d["paired_std_pp"]}',f'Primary identity bootstrap 95% CI: {d["identity_bootstrap_primary_95ci_pp"]}',''])
    (args.output/'decision.md').write_text('\n'.join(lines),encoding='utf-8')
    write(args.output/'status.json',dict(status='screening_complete',decision=decisions,locked_test_used=False,confirmation_run=False))


def main():
    p=argparse.ArgumentParser(); p.add_argument('--protocol',type=Path,default=HERE/'2026-11-06-lt-abc-quotas.json'); p.add_argument('--output',type=Path,required=True); p.add_argument('--init',type=Path,required=True); p.add_argument('--historical-pool',type=Path,action='append',default=[]); p.add_argument('--stage',choices=['smoke','prepare','train','score','pipeline'],default='pipeline'); args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True); q=json.loads(args.protocol.read_text(encoding='utf-8')); quotas(q)
    assert sha(args.init)==EXPECTED_INIT,'F2 hash mismatch'
    device='cuda:0'; assert torch.cuda.is_available()
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    pf=dict(started_utc=datetime.now(timezone.utc).isoformat(),git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),git_diff=subprocess.check_output(['git','status','--short'],cwd=REPO,text=True),python=sys.version,executable=sys.executable,platform=platform.platform(),torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),numpy=np.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__,h5py=h5py.__version__,init_path=str(args.init),init_sha256=sha(args.init),protocol_sha256=sha(args.protocol),script_sha256=sha(Path(__file__)),command=sys.argv,locked_test_used=False)
    pf['smoke']=smoke(q,args.init,device); write(args.output/'preflight.json',pf)
    pf['source_files']={str(s.relative_to(REPO)):sha(s) for s in list(HERE.glob('*.py'))+list(HERE.glob('*.m'))+list(HERE.glob('*.ps1'))}
    pf['generator_sources']={str(s.relative_to(REPO)):sha(s) for s in (REPO/'data_generation/0716/可配置组合干扰数据生成').rglob('*.m')}
    pf['external_files']=[dict(path=str(s),bytes=s.stat().st_size) for s in [args.init]+args.historical_pool]
    write(args.output/'preflight.json',pf)
    print('PREFLIGHT_AND_BATCH_SMOKE_PASS',flush=True)
    if args.stage=='smoke': return
    if args.stage in ('pipeline','prepare'):
        start=time.time()
        while not all((args.output/role/'generation_receipt.json').exists() for role in ('fit_pool','dev_pool')):
            write(args.output/'status.json',dict(status='waiting_for_atomic_generation_receipts',elapsed_seconds=time.time()-start,locked_test_used=False))
            if time.time()-start>7200: raise RuntimeError('candidate generation timeout; inspect generation.log')
            time.sleep(10)
        assert len(args.historical_pool)==2, 'both historical raw pools required for identity exclusion'
        m=prepare(args,q); print('DATA_ACCEPTANCE_PASS',flush=True)
    else: m=json.loads((args.output/'manifest.json').read_text(encoding='utf-8'))
    if args.stage in ('pipeline','train'): train(args,q,m,device)
    if args.stage in ('pipeline','score'): score(args,q,m,device)


if __name__=='__main__':
    try: main()
    except Exception:
        import traceback
        traceback.print_exc()
        # Preserve failed state and never silently reuse a partial run.
        if '--output' in sys.argv:
            write(Path(sys.argv[sys.argv.index('--output')+1])/'status.json',dict(status='failed',error=traceback.format_exc(),locked_test_used=False))
        sys.exit(1)
