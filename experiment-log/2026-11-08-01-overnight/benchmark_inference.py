"""Local inference cost for the final architecture widths."""
import argparse
from pathlib import Path

import torch
import numpy as np

import training


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    rows = []
    for width in (16, 32):
        model = training.base.build_backbone("dfms_iq", num_classes=6, width=width, dropout=.2).cuda().eval()
        for batch in (1, 32):
            x = torch.randn(batch, 2, 3000, device="cuda")
            with torch.inference_mode():
                for _ in range(20):
                    model(x)
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                timings = []
                for _ in range(100):
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start.record()
                    model(x)
                    end.record()
                    torch.cuda.synchronize()
                    timings.append(start.elapsed_time(end))
            rows.append({"width": width, "batch": batch, "parameters": sum(p.numel() for p in model.parameters()),
                "median_batch_ms": float(np.median(timings)), "p95_batch_ms": float(np.quantile(timings, .95)),
                "median_per_sample_ms": float(np.median(timings) / batch),
                "peak_allocated_MiB": torch.cuda.max_memory_allocated() / 2**20})
            del x
        del model
        torch.cuda.empty_cache()
    training.base.write(args.output, {"device": torch.cuda.get_device_name(), "precision": "FP32 TF32 disabled",
        "timing": "100 CUDA-event measurements after20 warmups, synthetic resident inputs, inference_mode",
        "source_sha256": training.base.sha(Path(__file__)), "rows": rows})
    print(rows, flush=True)


if __name__ == "__main__":
    main()
