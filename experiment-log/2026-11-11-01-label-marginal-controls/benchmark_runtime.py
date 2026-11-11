"""Wall-clock normal-forward cost including sigmoid and thresholds; no data loading."""
from common import *
from model_runtime import load_decision
import time

def main():
    setup();torch.manual_seed(20261111)
    records=[]
    for method in ['natural','marginal','combo','natural_mean','marginal_mean','combo_mean','natural_affine','marginal_affine','combo_affine']:
        model=load_decision(ROOT/'split1/decision_models'/f'{method}_2300.json','cuda')
        parameters=sum(p.numel() for p in model.parameters())
        for batch in [1,32]:
            x=torch.randn(batch,2,3000,device='cuda');timings=[]
            with torch.inference_mode():
                for _ in range(20):_=(torch.sigmoid(model(x))>=model.thresholds)
                torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
                for _ in range(3):
                    begin=time.perf_counter()
                    for _ in range(50):_=(torch.sigmoid(model(x))>=model.thresholds)
                    torch.cuda.synchronize();timings.append((time.perf_counter()-begin)*1000/50)
                peak=torch.cuda.max_memory_allocated()
            records.append({'method':method,'batch':batch,'parameters':parameters,'milliseconds_per_batch':timings,
                'median_ms_per_sample':float(np.median(timings)/batch),'peak_allocated_bytes':peak})
        del model,x;torch.cuda.empty_cache()
    write(ROOT/'analysis/runtime_cost.json',{'device':torch.cuda.get_device_name(),'torch':torch.__version__,'tf32':False,
        'threads':torch.get_num_threads(),'warmup':20,'rounds':3,'iterations':50,'includes':'model forward, sigmoid, threshold; inputs already on GPU',
        'records':records,'source_sha256':sha(Path(__file__)),'time':stamp()})
    print('RUNTIME_COST_COMPLETE',flush=True)

if __name__=='__main__':main()
