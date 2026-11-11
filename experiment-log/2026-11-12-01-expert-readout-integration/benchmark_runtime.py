"""Exclusive-GPU warm inference timing; extra supervision is reported separately."""
import argparse,subprocess
from common import *
from runtime import FactorialRuntime

def benchmark():
    # Timing concurrent with optimization/inference would not estimate deployable cost.
    process_query="Get-CimInstance Win32_Process | Where-Object { $_.Name -like 'python*' -and $_.CommandLine -match 'train_matrix.py|matrix_queue.py|evaluate_physical.py|factorial_fusion.py|score_matrix.py|export_runtime.py|verify_portable.py|supervision_control.py' } | Select-Object -ExpandProperty ProcessId"
    conflicts=subprocess.check_output(['powershell','-NoProfile','-Command',process_query],text=True).split()
    assert not conflicts,conflicts
    m=role_manifest('split1','development');x,_=base.load(m['path'],m['indices'][:64]);x=x.cuda();rows=[]
    methods=['reference','reference_affine','natural_none','marginal_none','combo_none','combo_global','combo_segment','combo_segment_guarded']
    for method in methods:
        path=ROOT/'supervision_control/split1/models/reference_affine_2200.json' if method=='reference_affine' else ROOT/'split1/decision_models'/f'{method}_2200.json'
        record=read(path);model=FactorialRuntime('split1',record).cuda().eval()
        params=sum(v.numel() for v in model.parameters());buffers=sum(v.numel() for v in model.buffers())
        for batch in [1,64]:
            torch.cuda.reset_peak_memory_stats()
            with torch.inference_mode():
                for _ in range(20):model(x[:batch])
                torch.cuda.synchronize();times=[]
                for _ in range(100):
                    start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                    start.record();model(x[:batch]);end.record();end.synchronize();times.append(start.elapsed_time(end))
            rows.append({'method':method,'seed':2200,'batch_size':batch,'parameters':params,'buffer_elements':buffers,
                'fitted_affine_coefficients':12 if record['kind']=='single_affine' else (0 if record['kind']=='single' else 24),
                'median_batch_ms':float(np.median(times)),'p95_batch_ms':float(np.quantile(times,.95)),
                'median_ms_per_identity':float(np.median(times)/batch),'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                'backbones':1 if record['kind'] in ['single','single_affine'] else 2})
        del model
    result={'device':torch.cuda.get_device_name(0),'torch':torch.__version__,'cuda':torch.version.cuda,'records':rows,
        'scope':'GPU resident normalized I/Q forward only; warmup20,100 CUDA event measurements; no concurrent campaign workloads',
        'not_included':['disk I/O','host-device transfer','IQ normalization','training','extra fusion_fit supervision'],
        'model_note':'one representative seed; timing includes PCA and residual heads, including selected epoch-zero heads; parameter count is registered neural parameters, affine coefficients are listed separately and also reside in buffers; buffers include thresholds and PCA statistics',
        'time':stamp()}
    write(ROOT/'runtime_benchmark.json',result);print('BENCHMARK_COMPLETE',len(rows),flush=True)

if __name__=='__main__':setup();benchmark()
