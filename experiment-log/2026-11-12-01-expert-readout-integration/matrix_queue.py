"""Two-worker matched-triad queue; claims and outcomes persist for audit."""
import argparse,os,subprocess
from common import *

def worker(queue_path,worker_name):
    queue_path=Path(queue_path);queue=read(queue_path);queue_hash=sha(queue_path)
    assert queue['approved_development_decision'] and (HERE/queue['stage_goal']).is_file()
    assert sha(HERE/queue['stage_goal'])==queue['stage_goal_sha256']
    folder=ROOT/'matrix/queue';folder.mkdir(exist_ok=True);deadline=datetime.fromisoformat(queue['cutoff']).timestamp()
    for number,job in enumerate(queue['jobs']):
        key=f"{number:03d}_{job['split']}_{job['cell']}_{job['seed']}";marker=ROOT/'matrix'/job['split']/job['cell']/f"matched_triad_{job['seed']}.json"
        if marker.exists():continue
        claim=folder/(key+'.claim.json')
        try:
            fd=os.open(claim,os.O_WRONLY|os.O_CREAT|os.O_EXCL)
            with os.fdopen(fd,'w',encoding='utf-8') as stream:json.dump({'worker':worker_name,'pid':os.getpid(),'queue_sha256':queue_hash,'job':job,'time':stamp()},stream)
        except FileExistsError:continue
        durations=[]
        for p in (ROOT/'matrix').glob('*/*/models/*/receipt.json'):
            durations.append(read(p)['training']['elapsed_seconds'])
        estimate=max(queue['minimum_seconds_per_model'],float(np.quantile(durations,.8)) if durations else 0)*3*queue['reserve_factor']
        if time.time()+estimate>deadline:
            write(folder/(key+'.not_started.json'),{'reason':'insufficient reserved time for full triad','estimated_seconds':estimate,'time':stamp()});continue
        log=folder/(key+'.log');command=[sys.executable,'-u',str(HERE/'train_matrix.py'),'--split',job['split'],'--cells',job['cell'],'--seeds',str(job['seed'])]
        started=time.time()
        with log.open('w',encoding='utf-8') as stream:result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,cwd=REPO)
        passed=result.returncode==0 and marker.exists() and read(marker)['passed']
        write(folder/(key+'.outcome.json'),{'passed':passed,'returncode':result.returncode,'wall_seconds':time.time()-started,'log_sha256':sha(log),'time':stamp()})
        print('QUEUE_TRIAD_FINISHED',worker_name,key,passed,round(time.time()-started,1),flush=True)
        if not passed:raise RuntimeError('Triad failed: '+key+'; inspect log before recovery')
    print('QUEUE_WORKER_FINISHED',worker_name,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--queue',required=True);p.add_argument('--worker',required=True);a=p.parse_args();worker(a.queue,a.worker)
