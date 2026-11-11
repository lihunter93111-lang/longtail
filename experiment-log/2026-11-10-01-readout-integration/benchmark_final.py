"""Measure frozen deployment entry points on one documented GPU."""
from common import *
from final_campaign import LOADERS


def measure(model, inputs):
    with torch.inference_mode():
        for _ in range(15):
            model(inputs)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        durations = []
        for _ in range(50):
            start,end = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            start.record()
            model(inputs)
            end.record()
            end.synchronize()
            durations.append(start.elapsed_time(end))
    return {"median_batch_ms":float(np.median(durations)),"p10_batch_ms":float(np.quantile(durations,.1)),
            "p90_batch_ms":float(np.quantile(durations,.9)),"peak_allocated_bytes":torch.cuda.max_memory_allocated()}


def main():
    setup()
    frozen = read(ROOT/"final_freeze.json")
    path,indices = role_paths("split1")["calibration"]
    inputs,_ = base.load(path,indices[:128])
    rows = []
    for item in frozen["models"].values():
        if item["split"]!="split1" or item["seed"]!=2200:
            continue
        assert base.sha(item["path"])==item["sha256"]
        model,_ = LOADERS[item["loader"]](item["path"])
        variants = [(item["name"],model)]
        if item["name"]=="baseline":
            variants.append(("baseline_direct_backbone",model.backbone))
            with torch.inference_mode():
                torch.testing.assert_close(model(inputs[:32].cuda()),model.backbone(inputs[:32].cuda()),rtol=0,atol=0)
        for name,runtime in variants:
            for batch in (1,32,128):
                values = inputs[:batch].cuda()
                result = measure(runtime,values)
                rows.append({"model":name,"batch":batch,"parameters":sum(p.numel() for p in runtime.parameters()),
                    "buffer_bytes":sum(b.numel()*b.element_size() for b in runtime.buffers()),
                    "checkpoint_bytes":Path(item["path"]).stat().st_size,
                    **result,"median_ms_per_sample":result["median_batch_ms"]/batch})
                del values
        del variants,runtime,model
        torch.cuda.empty_cache()
    helper.table(ROOT/"analysis/final_inference_cost.csv",rows)
    write(ROOT/"analysis/final_inference_cost.json",{"rows":rows,"gpu":torch.cuda.get_device_name(),
          "torch":torch.__version__,"cuda":torch.version.cuda,"cpu_threads":torch.get_num_threads(),
          "warmup":15,"repeats":50,"TF32":False,"input_length":3000,"time":stamp(),
          "note":"batch latency excludes host transfers; packaged baseline also computes readout features; direct backbone baseline is reported explicitly"})
    print("COST_BENCHMARK_COMPLETE",len(rows),flush=True)


if __name__=="__main__":
    main()
