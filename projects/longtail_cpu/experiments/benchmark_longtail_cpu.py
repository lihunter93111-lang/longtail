"""Compare finite 25-job CPU throughput before freezing the training plan."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

CORES=list(range(8,24))+list(range(40,56))


def worker(threads,slot,out):
    os.sched_setaffinity(0,set(CORES[slot*threads:(slot+1)*threads]))
    import torch
    import numpy as np
    from longtail.models.dfms import DFMSIQ
    torch.set_num_threads(threads);torch.set_num_interop_threads(1);torch.manual_seed(2050+slot)
    model=DFMSIQ();optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
    x=torch.randn(32,2,3000);y=torch.randint(0,2,(32,6)).float();times=[]
    for i in range(7):
        start=time.perf_counter();optimizer.zero_grad(set_to_none=True)
        z=model(x);loss=torch.nn.functional.binary_cross_entropy_with_logits(z,y);loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True);optimizer.step()
        if i>=2:times.append(time.perf_counter()-start)
    model.eval();evaluation=[]
    with torch.inference_mode():
        for i in range(4):
            xx=torch.randn(64,2,3000);start=time.perf_counter();z=model(xx)
            assert z.shape==(64,6) and torch.isfinite(z).all()
            if i:evaluation.append(time.perf_counter()-start)
    value={'threads':threads,'slot':slot,'cpus':sorted(os.sched_getaffinity(0)),
           'train_batch32_seconds':float(np.median(times)),'eval_batch64_seconds':float(np.median(evaluation))}
    (out/f'threads{threads}_slot{slot}.json').write_text(json.dumps(value))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--threads',type=int);p.add_argument('--slot',type=int);args=p.parse_args();out=args.output.resolve()
    if args.threads:worker(args.threads,args.slot,out);return
    if out.exists():raise FileExistsError('Use a new benchmark directory')
    out.mkdir(parents=True);profiles=[]
    for threads,parallel in [(1,25),(2,16),(4,8)]:
        started=time.time();processes=[]
        for slot in range(parallel):
            env=dict(os.environ);env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS=str(threads),MKL_NUM_THREADS=str(threads),OPENBLAS_NUM_THREADS='1')
            processes.append(subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--output',str(out),
                '--threads',str(threads),'--slot',str(slot)],env=env))
        codes=[p.wait() for p in processes]
        if any(codes):raise RuntimeError(str(codes))
        records=[json.loads((out/f'threads{threads}_slot{slot}.json').read_text()) for slot in range(parallel)]
        # Fixed per-slot measured latency and dynamic reuse of the first free slot.
        available=[0.]*parallel
        for _ in range(25):
            slot=min(range(parallel),key=lambda i:available[i]);r=records[slot]
            available[slot]+=6000*r['train_batch32_seconds']+400*r['eval_batch64_seconds']+10
        estimate=max(available)+120
        profiles.append({'threads_per_job':threads,'maximum_parallel_jobs':parallel,'physical_cores':threads*parallel,
            'estimated_25_job_minutes':estimate/60,'benchmark_wall_seconds':time.time()-started,'records':records})
        print(json.dumps({k:v for k,v in profiles[-1].items() if k!='records'}),flush=True)
    best=min(profiles,key=lambda p:p['estimated_25_job_minutes'])
    result={'profiles':profiles,'recommended_threads':best['threads_per_job'],'recommended_parallel_jobs':best['maximum_parallel_jobs'],
            'estimated_wall_minutes':best['estimated_25_job_minutes'],'synthetic_compute_only':True,
            'caveat':'Real data, checkpoint I/O and changing shared-machine load add uncertainty.'}
    (out/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='profiles'}),flush=True)


if __name__=='__main__':main()
