"""Frozen check/fit diagnostics; no model, optimizer or threshold is updated."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time


def worker(suite, output, job_id):
    import numpy as np
    import torch
    from torch.nn import functional as F
    from torch.utils.data import Subset
    from longtail.io import read_json, write_json, sha256
    from longtail.engine.checkpoint import load_checkpoint
    from longtail.data.loaders import make_dataset, make_loader
    from longtail.models.factory import build_model
    from longtail.metrics import evaluate_metrics, probabilities
    from longtail.weak_metrics import weak_metrics
    plan = read_json(suite / 'plan.json')
    job = next(j for j in plan['jobs'] if j['id']==job_id)
    os.sched_setaffinity(0,set(job['cpus']));torch.set_num_threads(4);torch.set_num_interop_threads(1)
    started = time.time();directory = Path(job['output']);checkpoint = directory / 'best.pt'
    before = sha256(checkpoint);payload,config = load_checkpoint(checkpoint)
    model = build_model(config['model'],num_classes=6).eval();model.load_state_dict(payload['model'])
    thresholds = payload['decision']['thresholds'];manifest=read_json(directory/'dataset.json')
    data_root=Path(config['data']['manifest']).parent
    check=make_dataset(manifest,data_root,payload['split'],'check',config['dataloader'])
    fit=make_dataset(manifest,data_root,payload['split'],'fit',config['dataloader'])
    fit_indices=np.sort(np.random.default_rng(20261103).choice(len(fit),1024,replace=False))
    is_query=hasattr(model,'query_decoder');mode='actual';lengths=[];head_count=0
    entropy_sum=None;top_sum=None;cosine_sum=None;mass_sum=None;uniform_checks=0
    if is_query:
        block=model.query_decoder[0].cross_attention
        original=block.forward;dimension=block.embed_dim;heads=block.num_heads
        def attention(query,key,value,**kwargs):
            nonlocal entropy_sum,top_sum,cosine_sum,mass_sum,head_count,uniform_checks
            result,w=original(query,key,value,need_weights=True,average_attn_weights=False)
            offsets=np.cumsum([0]+lengths)
            if mode=='actual_check':
                n=w.shape[-1]
                entropy=-(w*w.clamp_min(1e-30).log()).sum(-1)/np.log(n)
                top=w.max(-1).values*n
                norm=F.normalize(w,dim=-1)
                cosine=norm@norm.transpose(-1,-2)
                mass=torch.stack([w[...,offsets[i]:offsets[i+1]].sum(-1) for i in range(len(lengths))],dim=-1)
                arrays=[x.sum(0).detach().numpy() for x in (entropy,top,cosine,mass)]
                if entropy_sum is None:entropy_sum,top_sum,cosine_sum,mass_sum=arrays
                else:
                    entropy_sum+=arrays[0];top_sum+=arrays[1];cosine_sum+=arrays[2];mass_sum+=arrays[3]
                head_count+=len(w)
            if mode=='uniform_all':
                w=torch.full_like(w,1/w.shape[-1])
            elif mode=='uniform_within_levels':
                w=torch.cat([w[...,offsets[i]:offsets[i+1]].mean(-1,keepdim=True).expand(
                    *w.shape[:-1],lengths[i]) for i in range(len(lengths))],dim=-1)
            if mode in ('actual_check','uniform_all','uniform_within_levels'):
                v=F.linear(value,block.in_proj_weight[2*dimension:],block.in_proj_bias[2*dimension:])
                v=v.reshape(len(value),value.shape[1],heads,dimension//heads).transpose(1,2)
                manual=F.linear((w@v).transpose(1,2).reshape(len(query),query.shape[1],dimension),
                                block.out_proj.weight,block.out_proj.bias)
                if mode=='actual_check' and uniform_checks==0:
                    torch.testing.assert_close(manual,result,atol=3e-6,rtol=3e-5);uniform_checks+=1
                if mode!='actual_check':result=manual
            return result,w.mean(1) if kwargs.get('need_weights',True) else None
        block.forward=attention

    def collect(dataset,current_mode):
        nonlocal mode,lengths
        mode=current_mode;parts={};global_z=[];query_z=[]
        with torch.inference_mode():
            for batch in make_loader(dataset,config['dataloader'],batch_size=32):
                if is_query:
                    # Set before attention is called; all full-length samples have this pyramid.
                    lengths=[manifest['signal_length']//(2**i) for i in model.memory_levels]
                    z,d=model.forward_diagnostics(batch['signal'])
                    global_z.append(d['global_logits'].numpy());query_z.append(
                        (d['query_logits']*d['query_scale']).numpy())
                else:z=model(batch['signal'])
                values={'logits':z.numpy(),'targets':batch['target'].numpy(),
                        'indices':batch['index'].numpy(),'identities':batch['identity'].numpy(),
                        **{k:batch[k].numpy() for k in ('weak_mask','relative_power_db','component_jnr_db')}}
                for k,v in values.items():parts.setdefault(k,[]).append(v)
        z={k:np.concatenate(v) for k,v in parts.items()}
        if is_query:z.update(global_logits=np.concatenate(global_z),query_residual=np.concatenate(query_z))
        return z

    def metrics(z,logits=None):
        logits=z['logits'] if logits is None else logits
        m=evaluate_metrics(z['targets'],logits,manifest['classes'],thresholds,
                           combination_counts=payload['split']['fit_combination_counts'])
        w=weak_metrics(z['targets'],logits,manifest['classes'],thresholds,z['weak_mask'],
                       z['relative_power_db'],payload['split']['fit_label_counts'])
        return {'exact':m['subset_accuracy'],'macro_f1':m['macro_f1'],'bce':float(np.mean(
            np.logaddexp(0,logits)-z['targets']*logits)),'weak_aupr':w['weak_macro_aupr'],
            'weak_recall':w['weak_macro_recall'],'weak_per_label':w['per_label']}

    actual=collect(check,'actual_check')
    frozen=dict(np.load(directory/'evaluation/check/predictions.npz'))
    for key in ('targets','identities','indices'):np.testing.assert_array_equal(actual[key],frozen[key])
    np.testing.assert_allclose(actual['logits'],frozen['logits'],atol=1e-5,rtol=1e-5)
    outcomes={'actual':metrics(actual)}
    expected=read_json(directory/'evaluation/check/metrics.json')['metrics']['calibrated']['subset_accuracy']
    assert abs(outcomes['actual']['exact']-expected)<1e-12
    dest=output/job_id;dest.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(dest/'check_actual.npz',**actual)
    attention_stats=None;residual=None;gradient=None
    if is_query:
        attention_stats={'head_entropy_by_class':(entropy_sum/head_count).tolist(),
                         'head_peak_over_uniform_by_class':(top_sum/head_count).tolist(),
                         'head_class_cosine':(cosine_sum/head_count).tolist(),
                         'head_level_mass':(mass_sum/head_count).tolist(),
                         'memory_lengths':lengths,'labels':manifest['classes'],
                         'manual_attention_value_projection_verified':bool(uniform_checks)}
        for variant in ('uniform_all','uniform_within_levels'):
            if variant=='uniform_within_levels' and len(model.memory_levels)==1:continue
            z=collect(check,variant);outcomes[variant]=metrics(z)
            outcomes[variant]['logit_abs_delta_mean']=float(np.abs(z['logits']-actual['logits']).mean())
            np.savez_compressed(dest/f'check_{variant}.npz',**z)
        outcomes['zero_query']=metrics(actual,actual['global_logits'])
        p=probabilities(actual['logits']);g=probabilities(actual['global_logits']);y=actual['targets'].astype(bool)
        final=p>=thresholds;global_pred=g>=thresholds;delta=actual['query_residual']
        residual={'gate_tanh':model.query_scale.tanh().detach().tolist(),
                  'global_logit_std_by_class':actual['global_logits'].std(0).tolist(),
                  'residual_std_by_class':delta.std(0).tolist(),
                  'residual_mean_by_class':delta.mean(0).tolist(),
                  'label_queries_norm':model.label_queries.weight.norm(dim=1).detach().tolist(),
                  'changed_label_decisions':int((global_pred!=final).sum()),
                  'corrected_label_decisions':int(((global_pred!=y)&(final==y)).sum()),
                  'spoiled_label_decisions':int(((global_pred==y)&(final!=y)).sum())}
    fit_records=collect(Subset(fit,fit_indices),'actual_fit');fit_metric=metrics(fit_records)
    np.savez_compressed(dest/'fit_sample_actual.npz',**fit_records)
    z=fit_records['logits'];y=fit_records['targets'].astype(bool);weak=fit_records['weak_mask'].astype(bool)
    losses=np.logaddexp(0,z)-y*z;abs_grad=np.abs(probabilities(z)-y)
    groups={'weak_positive':weak,'strong_positive':y&~weak,'negative':~y}
    grouped={name:{'count':int(mask.sum()),'decision_fraction':float(mask.mean()),
                   'mean_bce':float(losses[mask].mean()),'bce_mass_fraction':float(losses[mask].sum()/losses.sum()),
                   'absolute_logit_gradient_fraction':float(abs_grad[mask].sum()/abs_grad.sum())}
             for name,mask in groups.items()}
    if is_query:
        mode='gradient';model.zero_grad(set_to_none=True)
        batch=next(iter(make_loader(Subset(fit,fit_indices[:64]),config['dataloader'],batch_size=64)))
        loss=F.binary_cross_entropy_with_logits(model(batch['signal']),batch['target']);loss.backward()
        grad=block.in_proj_weight.grad.detach()
        gradient={'role':'fit','samples':64,'mode':'eval; no optimizer step',
                  'attention_in_projection_rms_gradient_q_k_v':[float(x.square().mean().sqrt()) for x in grad.chunk(3)],
                  'label_queries_rms_gradient':float(model.label_queries.weight.grad.square().mean().sqrt()),
                  'query_scale_gradient':model.query_scale.grad.tolist()}
    fit.close();check.close()
    assert sha256(checkpoint)==before
    result={'job':job_id,'checkpoint_sha256':before,'role':'check; fit subset for diagnosis only',
            'fit_subset_size':1024,'fit_subset_rng':20261103,'outcomes':outcomes,'fit_sample':fit_metric,
            'fit_loss_groups':grouped,'attention':attention_stats,'residual':residual,'gradient_snapshot':gradient,
            'thresholds':thresholds,'thresholds_refitted':False,'optimizer_steps':0,
            'seconds':time.time()-started,
            'limitations':['Frozen replacements change context; they are dependence diagnostics, not independently retrained architectures.',
                           'Fit metrics use a fixed 1024-sample subset, not all training identities.',
                           'Gradient snapshot uses one fixed fit batch in eval mode; it is not a training-trajectory measurement.',
                           'No development_test is read by this diagnostic.']}
    write_json(dest/'diagnostic.json',result)
    print(job_id,{k:round(v['exact'],4) for k,v in outcomes.items()},'seconds',round(result['seconds'],1),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite',required=True,type=Path);parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--job');args=parser.parse_args();suite=args.suite.resolve();output=args.output.resolve()
    if args.job:worker(suite,output,args.job)
    else:
        from longtail.io import read_json,write_json,sha256
        plan=read_json(suite/'plan.json')
        if output.exists():raise FileExistsError('Use a new diagnostic directory')
        output.mkdir(parents=True)
        write_json(output/'plan.json',{'suite':str(suite),'jobs':plan['jobs'],
            'code_sha256':sha256(Path(__file__)),'role':'check and fixed fit subset',
            'variants':['actual','uniform_all','uniform_within_levels','zero_query'],
            'fit_subset_rng':20261103,'fit_subset_size':1024,'optimizer_steps':0,
            'threshold_refits':0,'development_test_used':False})
        processes=[]
        for job in plan['jobs']:
            env=dict(os.environ);env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1')
            processes.append(subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--suite',str(suite),
                '--output',str(output),'--job',job['id']],env=env))
        codes=[p.wait() for p in processes]
        if any(codes):raise RuntimeError(str(codes))
        write_json(output/'completion.json',{'jobs_completed':len(codes),'failures':[]})


if __name__=='__main__':main()
