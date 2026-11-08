"""CPU workers for ordered calibration, frozen heads and one-bit local heads."""
import json
from pathlib import Path
import time
import hashlib
import h5py
import numpy as np
import torch
from torch import nn

from .io import read_json,write_json,sha256
from .engine.checkpoint import load_checkpoint,save_checkpoint
from .models.factory import build_model
from .models.decision_heads import FiveLabelHead,LocalResidual,residual_logits,pair_margin_loss
from .data.dataset import IQDataset
from .data.loaders import make_loader
from .data.sampling import sampler_epoch_seed,sampling_probabilities
from .metrics import evaluate_metrics,probabilities
from .weak_metrics import weak_metrics
from .staged_decision import fit_rules,get_rule

CLASSES=['FMB','AMA','SawFM','SinFM','MFT','ISRJ']
HEADS=['head_natural','head_mixed']
LOCALS=['local_natural','local_pair_bce','local_pair_margin']


def record_metric(records,logits,thresholds):
    m=evaluate_metrics(records['targets'],logits,CLASSES,thresholds)
    w=weak_metrics(records['targets'],logits,CLASSES,thresholds,records['weak_mask'],records['relative_power_db'])
    y=records['targets'].astype(bool);p=probabilities(logits)>=thresholds;strong=y&~records['weak_mask']
    m.update(exact=m['subset_accuracy'],weak_component=w,tail_f1=float(np.mean([m['per_label'][c]['f1'] for c in ['AMA','SinFM']])),
             mean_fpr=float(np.mean([v['negative_fpr'] for v in m['per_label'].values()])),
             saw_weak_recall=w['per_label']['SawFM']['weak_recall'],target_weak_recall=float(np.mean([w['per_label'][c]['weak_recall'] for c in ['SawFM','MFT','ISRJ']])),
             weak_micro_recall=w['weak_micro_recall'],nonweak_recall=float(p[strong].mean()))
    return m


def load_source(plan,seed,kind):
    source=plan['references'][str(seed)][kind];assert sha256(source['checkpoint'])==source['sha256']
    payload,config=load_checkpoint(source['checkpoint']);model=build_model(config['model'],num_classes=6)
    model.load_state_dict(payload['model'],strict=True);model.eval();model.requires_grad_(False)
    return model,config


def extract_cache(suite,seed,stage,role,pair=False):
    plan=read_json(suite/'plan.json');folder=suite/'cache'/f'seed{seed}';folder.mkdir(parents=True,exist_ok=True)
    tag=f'pair_{role}' if pair else 'fit' if role=='fit' else f'stage{stage}_{role}'
    dest=folder/(tag+'.npz');source_file=folder/(tag+'_source.json')
    report=role=='development_test'
    if report:
        frozen=read_json(suite/f'stage{stage}'/'frozen.json');assert frozen['plan_sha256']==sha256(suite/'plan.json')
        assert sha256(suite/f'stage{stage}'/'selection.json')==frozen['selection_sha256']
        for path,digest in frozen['files'].items():assert sha256(path)==digest
    if dest.exists():
        assert sha256(dest)==read_json(source_file)['cache_sha256'];return dict(np.load(dest))
    manifest=read_json(plan['data']['stage_manifests'][str(stage)]);split=read_json(suite/f'split_stage{stage}.json')
    if pair:
        paired=read_json(plan['data']['pair_manifest']);file=Path(paired['files'][role]['path']);indices=np.arange(paired['files'][role]['samples'])
    else:
        key='train' if role=='fit' else 'development';file=Path(plan['data']['stage_manifests'][str(stage)]).parent/manifest['files'][key]['path']
        indices=np.asarray(split['roles'][role])
    model,config=load_source(plan,seed,'B');dataset=IQDataset(file,indices,config['dataloader']['preprocess'],cache='memory')
    fields={};features=[];base=[]
    try:
        loader=make_loader(dataset,config['dataloader'],batch_size=64)
        with torch.inference_mode():
            for batch in loader:
                feat=model.forward_features(batch['signal']);features.append(feat.numpy());base.append(model.classifier(feat).numpy())
                for key in ['target','index','identity','weak_mask','relative_power_db','component_jnr_db','component_power_db']:
                    if key in batch:fields.setdefault(key,[]).append(batch[key].numpy())
        records={key:np.concatenate(value) for key,value in fields.items()};records['targets']=records.pop('target').astype(np.uint8)
        records['indices']=records.pop('index');records['identities']=records.pop('identity')
        records['features']=np.concatenate(features);records['base_logits']=np.concatenate(base)
        if role!='fit':
            for kind in (['FMB'] if pair else ['A','FMB']):
                expert,_=load_source(plan,seed,kind);values=[]
                with torch.inference_mode():
                    for batch in make_loader(dataset,config['dataloader'],batch_size=64):values.append(expert(batch['signal']).numpy())
                records[kind+'_logits']=np.concatenate(values);del expert
    finally:dataset.close()
    if pair:
        with h5py.File(file) as f:
            for name in ['pair_id','pair_target','pair_side','edge_id']:records[name]=f[name][:].ravel()
            records['physical_descriptor']=f['physical_descriptor'][:].T
    np.savez_compressed(dest,**records)
    write_json(source_file,{'cache_sha256':sha256(dest),'data_file':str(file),'data_sha256':sha256(file),
        'backbone_sha256':plan['references'][str(seed)]['B']['sha256'],'role':role,'stage':stage,'pair':pair,
        'model_mode':'eval, all source parameters frozen','model_input':'single I/Q only'})
    return records


def fused_logits(records,logits=None):
    z=(records['base_logits'] if logits is None else logits).copy();z[:,0]=records['FMB_logits'][:,0];return z


def fit_bundle(records,logits):
    a=records['A_logits'].copy();a[:,0]=records['FMB_logits'][:,0]
    return fit_rules(records['targets'],fused_logits(records,logits),records['weak_mask'],a)


def phase1(suite,seed,report=False):
    output=suite/'stage1'/f'seed{seed}';output.mkdir(parents=True,exist_ok=True)
    if not report:
        cal=extract_cache(suite,seed,1,'calibration');decision=fit_bundle(cal,cal['base_logits'])
        write_json(output/'decisions.json',decision)
        roles=['calibration','check']
    else:decision=read_json(output/'decisions.json');roles=['development_test']
    for role in roles:
        records=extract_cache(suite,seed,1,role);z=fused_logits(records);metrics={}
        for rule,th in decision['thresholds'].items():
            if th is not None:metrics[rule]=record_metric(records,z,th)
        a=records['A_logits'].copy();a[:,0]=records['FMB_logits'][:,0]
        metrics['A_f1']=record_metric(records,a,decision['A_f1_thresholds'])
        write_json(output/f'metrics_{role}.json',{'seed':seed,'role':role,'metrics':metrics})


def draw_indices(labels,seed,epoch,mode,draws=8000):
    natural=torch.randperm(len(labels),generator=torch.Generator().manual_seed(sampler_epoch_seed(seed,epoch,10)))
    if mode=='natural':return natural[:draws].numpy()
    if mode!='mixed':raise ValueError('Unknown fixed-budget sampler')
    combo=torch.multinomial(torch.from_numpy(sampling_probabilities(labels,'combo_aware')),draws//2,replacement=True,
                            generator=torch.Generator().manual_seed(sampler_epoch_seed(seed,epoch,11)))
    joined=torch.cat([natural[:draws-draws//2],combo])
    return joined[torch.randperm(draws,generator=torch.Generator().manual_seed(sampler_epoch_seed(seed,epoch,12)))].numpy()


def head_logits(records,checkpoint):
    if checkpoint is None:return records['base_logits'].copy()
    payload=torch.load(checkpoint,map_location='cpu',weights_only=True)
    weight=payload['state_dict']['linear.weight'];bias=payload['state_dict']['linear.bias'];z=records['base_logits'].copy()
    with torch.inference_mode():z[:,1:]=(torch.from_numpy(records['features'])@weight.T+bias).numpy()
    return z


def train_head(suite,seed,method,fit,check):
    plan=read_json(suite/'plan.json');output=suite/'stage2'/f'seed{seed}';output.mkdir(parents=True,exist_ok=True)
    status=output/(method+'_status.json');checkpoint=output/(method+'.pt')
    if status.exists() and read_json(status)['status']=='completed':return
    torch.set_num_threads(1);torch.manual_seed(seed)
    source,_=load_checkpoint(plan['references'][str(seed)]['B']['checkpoint']);state=source['model']
    model=FiveLabelHead(state['classifier.1.weight'],state['classifier.1.bias']);del source
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=24)
    x=torch.from_numpy(fit['features']);y=torch.from_numpy(fit['targets'].astype(np.float32));validation=torch.from_numpy(check['features'])
    history=[];best=-1.;started=time.time();mode='natural' if method=='head_natural' else 'mixed'
    for epoch in range(1,25):
        model.train();indices=draw_indices(fit['targets'],seed,epoch,mode);loss_sum=0.;epoch_start=time.time()
        for start in range(0,8000,32):
            rows=indices[start:start+32];optimizer.zero_grad(set_to_none=True)
            loss=nn.functional.binary_cross_entropy_with_logits(model(x[rows]),y[rows,1:]);loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True);optimizer.step();loss_sum+=loss.item()
        scheduler.step();model.eval()
        with torch.inference_mode():pred=model(validation).numpy()
        score=evaluate_metrics(check['targets'][:,1:],pred,CLASSES[1:])['macro_aupr']
        row={'epoch':epoch,'steps':epoch*250,'samples':8000,'loss':loss_sum/250,'check_aupr':score,
             'unique_samples':len(np.unique(indices)),'draw_order_sha256':hashlib.sha256(indices.astype('<i8').tobytes()).hexdigest(),'epoch_seconds':time.time()-epoch_start}
        history.append(row)
        payload={'kind':'frozen_feature_five_label_head','state_dict':model.state_dict(),'epoch':epoch,'history':history,'optimizer':optimizer.state_dict(),
                 'optimizer_steps':epoch*250,'seed':seed,'backbone_sha256':plan['references'][str(seed)]['B']['sha256'],
                 'fit_cache_sha256':sha256(suite/'cache'/f'seed{seed}'/'fit.npz'),'trainable_parameters':sum(p.numel() for p in model.parameters())}
        save_checkpoint(output/('last_'+method+'.pt'),payload)
        if score>best:best=score;save_checkpoint(checkpoint,payload)
        write_json(output/('history_'+method+'.json'),history)
        print(json.dumps({'stage':2,'seed':seed,'method':method,**row}),flush=True)
    assert payload['optimizer_steps']==6000
    write_json(status,{'status':'completed','optimizer_steps':6000,'epochs':24,'seconds':time.time()-started,
                       'checkpoint_sha256':sha256(checkpoint),'backbone_unchanged':True,'trainable_parameters':645})


def stage_rule(suite):return read_json(suite/'stage1/selection.json')['selected']


def fit_and_measure(suite,seed,stage,methods,predictor,report=False):
    output=suite/f'stage{stage}'/f'seed{seed}';output.mkdir(parents=True,exist_ok=True);rule=stage_rule(suite)
    if not report:
        cal=extract_cache(suite,seed,stage,'calibration');decisions={}
        for method in methods:decisions[method]=fit_bundle(cal,predictor(method,cal))
        guard=fit_bundle(cal,cal['base_logits'])
        write_json(output/'decisions.json',{'rule':rule,'models':decisions,'B_guard':guard});roles=['calibration','check']
    else:
        bundle=read_json(output/'decisions.json');decisions=bundle['models'];guard=bundle['B_guard'];roles=['development_test']
    for role in roles:
        records=extract_cache(suite,seed,stage,role);metrics={};fallbacks={}
        for method in methods:
            th,fallback=get_rule(decisions[method],rule);fallbacks[method]=fallback
            metrics[method]=record_metric(records,fused_logits(records,predictor(method,records)),th)
        metrics['B_f1_guard']=record_metric(records,fused_logits(records),guard['thresholds']['f1'])
        a=records['A_logits'].copy();a[:,0]=records['FMB_logits'][:,0]
        metrics['A_f1']=record_metric(records,a,guard['A_f1_thresholds'])
        write_json(output/f'metrics_{role}.json',{'seed':seed,'role':role,'rule':rule,'fallbacks':fallbacks,'metrics':metrics})


def phase2(suite,seed,report=False):
    output=suite/'stage2'/f'seed{seed}'
    if not report:
        fit=extract_cache(suite,seed,2,'fit');check=extract_cache(suite,seed,2,'check')
        for method in HEADS:train_head(suite,seed,method,fit,check)
    fit_and_measure(suite,seed,2,['base']+HEADS,lambda m,r:head_logits(r,None if m=='base' else output/(m+'.pt')),report)


def previous_head(suite,seed):
    selected=read_json(suite/'stage2/selection.json')['selected']
    return None if selected=='base' else suite/'stage2'/f'seed{seed}'/(selected+'.pt')


def probe_cache(suite,seed):
    extract_cache(suite,seed,3,'calibration');extract_cache(suite,seed,3,'check')
    for role in ['fit','check']:extract_cache(suite,seed,3,role,pair=True)


def train_local(suite,seed,method,fit,check,pairs,active):
    plan=read_json(suite/'plan.json');output=suite/'stage3'/f'seed{seed}';output.mkdir(parents=True,exist_ok=True)
    status=output/(method+'_status.json');checkpoint=output/(method+'.pt')
    if status.exists() and read_json(status)['status']=='completed':return
    torch.set_num_threads(1);torch.manual_seed(seed)
    model=LocalResidual(active=active);optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=24)
    head=previous_head(suite,seed);x=torch.from_numpy(fit['features']);y=torch.from_numpy(fit['targets'].astype(np.float32))
    base=torch.from_numpy(head_logits(fit,head));cv=torch.from_numpy(check['features']);cb=torch.from_numpy(head_logits(check,head))
    selected_pairs=np.flatnonzero(np.isin(pairs['pair_target'][::2],np.array([2,4,5])[active]))
    pair_rows=np.ravel(np.c_[2*selected_pairs,2*selected_pairs+1]);px=torch.from_numpy(pairs['features'][pair_rows])
    py=torch.from_numpy(pairs['targets'][pair_rows].astype(np.float32));pb=torch.from_numpy(head_logits(pairs,head)[pair_rows])
    pair_target=torch.from_numpy(pairs['pair_target'][pair_rows][::2].astype(np.int64))
    history=[];best=-1.;started=time.time()
    for epoch in range(1,25):
        model.train();indices=draw_indices(fit['targets'],seed,epoch,'natural')
        generator=np.random.default_rng(sampler_epoch_seed(seed,epoch,20));draw_pairs=generator.integers(0,len(selected_pairs),size=1000)
        loss_sum=0.;hinge_sum=0.;epoch_start=time.time()
        for step in range(250):
            natural=indices[step*32:(step+1)*32] if method=='local_natural' else indices[step*24:(step+1)*24]
            xx=x[natural];yy=y[natural];bb=base[natural];target=None
            if method!='local_natural':
                group=draw_pairs[step*4:(step+1)*4];rows=np.ravel(np.c_[2*group,2*group+1])
                xx=torch.cat([xx,px[rows]]);yy=torch.cat([yy,py[rows]]);bb=torch.cat([bb,pb[rows]]);target=pair_target[group]
            optimizer.zero_grad(set_to_none=True);z=residual_logits(bb,model(xx))
            loss=nn.functional.binary_cross_entropy_with_logits(z[:,1:],yy[:,1:]);hinge=torch.tensor(0.)
            if method=='local_pair_margin':
                hinge=pair_margin_loss(z[24:],target);loss=loss+.1*hinge
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True);optimizer.step()
            loss_sum+=loss.item();hinge_sum+=hinge.item()
        scheduler.step();model.eval()
        with torch.inference_mode():z=residual_logits(cb,model(cv)).numpy()
        score=evaluate_metrics(check['targets'][:,1:],z[:,1:],CLASSES[1:])['macro_aupr']
        row={'epoch':epoch,'steps':epoch*250,'samples':8000,'natural_samples':8000 if method=='local_natural' else 6000,
             'paired_views':0 if method=='local_natural' else 2000,'loss':loss_sum/250,'pair_hinge':hinge_sum/250,'check_aupr':score,
             'natural_draw_sha256':hashlib.sha256(indices.astype('<i8').tobytes()).hexdigest(),
             'pair_draw_sha256':hashlib.sha256(draw_pairs.astype('<i8').tobytes()).hexdigest(),'epoch_seconds':time.time()-epoch_start}
        history.append(row)
        payload={'kind':'frozen_feature_local_residual','state_dict':model.state_dict(),'epoch':epoch,'history':history,'optimizer_steps':epoch*250,'optimizer':optimizer.state_dict(),
                 'seed':seed,'active_targets':list(active),'backbone_sha256':plan['references'][str(seed)]['B']['sha256'],
                 'previous_head_sha256':sha256(head) if head else None,'trainable_parameters':sum(p.numel() for p in model.parameters())}
        save_checkpoint(output/('last_'+method+'.pt'),payload)
        if score>best:best=score;save_checkpoint(checkpoint,payload)
        write_json(output/('history_'+method+'.json'),history)
        print(json.dumps({'stage':3,'seed':seed,'method':method,**row}),flush=True)
    write_json(status,{'status':'completed','epochs':24,'optimizer_steps':6000,'seconds':time.time()-started,
                      'backbone_and_previous_head_unchanged':True,'checkpoint_sha256':sha256(checkpoint)})


def local_logits(suite,seed,method,records):
    base=head_logits(records,previous_head(suite,seed))
    if method=='base':return base
    payload=torch.load(suite/'stage3'/f'seed{seed}'/(method+'.pt'),map_location='cpu',weights_only=True)
    model=LocalResidual(active=payload['active_targets']);model.load_state_dict(payload['state_dict']);model.eval()
    with torch.inference_mode():return residual_logits(torch.from_numpy(base),model(torch.from_numpy(records['features']))).numpy()


def phase3(suite,seed,report=False):
    gate=read_json(suite/'stage3/observability.json');active=[c in gate['enabled_targets'] for c in [2,4,5]]
    methods=['base']+(LOCALS if any(active) else [])
    if not report and any(active):
        fit=extract_cache(suite,seed,3,'fit');check=extract_cache(suite,seed,3,'check');pairs=extract_cache(suite,seed,3,'fit',pair=True)
        for method in LOCALS:train_local(suite,seed,method,fit,check,pairs,active)
    fit_and_measure(suite,seed,3,methods,lambda m,r:local_logits(suite,seed,m,r),report)
    if report:
        records=extract_cache(suite,seed,3,'development_test',pair=True)
        decisions=read_json(suite/'stage3'/f'seed{seed}'/'decisions.json');metrics={}
        from sklearn.metrics import roc_auc_score
        for method in methods:
            z=fused_logits(records,local_logits(suite,seed,method,records));threshold,fallback=get_rule(decisions['models'][method],decisions['rule'])
            m=record_metric(records,z,threshold);m['one_bit_targets']={}
            for c in [0,2,4,5]:
                rows=records['pair_target']==c;yy=records['targets'][rows,c];score=z[rows,c]
                m['one_bit_targets'][CLASSES[c]]={'auroc':float(roc_auc_score(yy,score)),
                    'recall':float((probabilities(score)[yy.astype(bool)]>=threshold[c]).mean()),
                    'fpr':float((probabilities(score)[~yy.astype(bool)]>=threshold[c]).mean())}
            metrics[method]=m
        write_json(suite/'stage3'/f'seed{seed}'/'metrics_pair_report.json',{'role':'pair_report','decision_fit_role':'natural calibration only','metrics':metrics})
