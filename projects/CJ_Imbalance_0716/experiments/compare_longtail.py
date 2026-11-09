"""Run matched backbone/readout or frozen-fusion comparisons without batch scripts."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import json
import multiprocessing
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cjlt.data.ama_hdf5 import load_roles
from cjlt.engine.longtail_evaluation import compare_predictions
from cjlt.engine.longtail_workflow import read_config, train_job, verify_frozen
from cjlt.engine.model_bundle import file_sha256, load_bundle
from cjlt.engine.readout_trainer import collect_records, metrics


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def comparison_stages(config, output, mode):
    """Each list is trained completely before its dependent stage starts."""
    if mode == "fusion":
        stages = []
        for name, kind, targets in [
            ("R_calibrated", "reference", []), ("affine_all", "affine", list(config["data"]["classes"])),
            ("saw_gate", "selective", ["SawFM"]), ("tail_gate", "selective", ["SawFM", "MFT", "ISRJ"]),
        ]:
            arm = deepcopy(config)
            arm["fusion"].update(kind=kind, targets=targets)
            stages.append((name, "fusion", arm, output / name))
        return [stages], [(name, "R_calibrated") for name in ("affine_all", "saw_gate", "tail_gate")]
    if not config["data"].get("weak_views"):
        raise ValueError("The shared comparison needs a weak_views configuration")
    backbones = []
    for name in ("A", "B", "W"):
        arm = deepcopy(config)
        arm["data"].pop("pairs", None)
        if name != "W":
            arm["data"].pop("weak_views")
        arm["training"]["schedule"] = "constant" if name == "A" else "endpoint_cosine"
        backbones.append((name, "baseline", arm, output / name))
    first, final = [], []
    for backbone in ("B", "W"):
        for source in ("global", "segment"):
            name = backbone + "_" + source
            arm = deepcopy(config)
            for key in ("weak_views", "pairs", "cache"):
                arm["data"].pop(key, None)
            arm["reference"] = str(output / backbone / "seed{seed}/model.pt")
            arm["training"] = {"epochs": 24, "steps_per_epoch": 250, "batch_size": 32,
                               "lr": 3e-4, "weight_decay": 1e-4, **config.get("readout_training", {})}
            arm["readout"] = {"dimensions": 128, "hidden": 32, "head_threads": 1,
                              **config.get("readout", {}), "name": "sawmft", "targets": ["SawFM", "MFT"],
                              "source": source, "pair_views": 0}
            first.append((name + "_sawmft", "readout", arm, output / name / "sawmft"))
            last = deepcopy(arm)
            last["reference"] = str(output / name / "sawmft/seed{seed}/model.pt")
            last["readout"].update(name="isrj", targets=["ISRJ"])
            final.append((name, "readout", last, output / name / "isrj"))
    pairs = [("B", "A"), ("W", "B"), ("B_segment", "B_global"),
             ("W_segment", "W_global"), ("W_segment", "B_segment")]
    return [backbones, first, final], pairs


def evaluate(job):
    name, seed, config, model_path, output, threads, device = job
    torch.set_num_threads(threads)
    roles = load_roles(config["data"])
    try:
        verify_frozen(Path(model_path).with_name("frozen.json"), model_path, config, roles)
        model, thresholds, _ = load_bundle(model_path, seed)
        records = collect_records(model, roles["report"], device=device)
        measured = metrics(records, records["logits"], thresholds)
        destination = Path(output) / "evaluation" / name / ("seed{}".format(seed))
        destination.mkdir(parents=True, exist_ok=True)
        write(destination / "metrics.json", measured)
        np.savez_compressed(destination / "predictions.npz", **records, thresholds=np.asarray(thresholds))
        return dict(method=name, seed=seed, **measured)
    finally:
        for dataset in roles.values():
            dataset.close()


def run_comparison(config, output, mode, seeds, jobs=1, threads=4, device="cpu", report=False):
    if jobs < 1 or threads < 1 or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Use positive concurrency and distinct model seeds")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    plan = dict(config=config, mode=mode, seeds=seeds)
    plan_path = output / "plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError("Comparison plan changed; use a new output directory")
    write(plan_path, plan)
    stages, pairs = comparison_stages(config, output, mode)
    records = []
    pool = ProcessPoolExecutor(max_workers=jobs, mp_context=multiprocessing.get_context("spawn")) if jobs > 1 else None
    try:
        write(output / "status.json", dict(status="training", report_used=False))
        for stage in stages:
            tasks = [(kind, arm, seed, str(path), threads, device)
                     for _, kind, arm, path in stage for seed in seeds]
            results = list(pool.map(train_job, tasks)) if pool else list(map(train_job, tasks))
            for (name, _, arm, _), selected in zip(stage,
                                                  [results[i:i + len(seeds)] for i in range(0, len(results), len(seeds))]):
                # Keep the final stage and the three backbone controls for reporting.
                if not name.endswith("_sawmft"):
                    records.extend(dict(method=name, config=arm, **row) for row in selected)
        write(output / "all_models_frozen.json", [dict(method=row["method"], seed=row["seed"],
              path=row["model"], sha256=file_sha256(row["model"])) for row in records])
        if not report:
            write(output / "status.json", dict(status="frozen", report_used=False, models=len(records)))
            return records
        write(output / "status.json", dict(status="report_inference", report_used=True))
        tasks = [(row["method"], row["seed"], row["config"], row["model"], str(output), threads, device) for row in records]
        rows = list(pool.map(evaluate, tasks)) if pool else list(map(evaluate, tasks))
        write(output / "metrics.json", rows)
        summary = dict(methods={}, comparisons={})
        for name in sorted({row["method"] for row in rows}):
            selected = [row for row in rows if row["method"] == name]
            summary["methods"][name] = {key: float(np.mean([row[key] for row in selected]))
                                        for key in ("exact", "raw_bce", "macro_f1", "macro_aupr")}
        for candidate, reference in pairs:
            paths = lambda name: {seed: output / "evaluation" / name / ("seed{}".format(seed)) / "predictions.npz" for seed in seeds}
            summary["comparisons"][candidate + "_vs_" + reference] = compare_predictions(paths(candidate), paths(reference))
        write(output / "summary.json", summary)
        write(output / "status.json", dict(status="completed", report_used=True, rows=len(rows)))
        return summary
    except Exception as error:
        write(output / "status.json", dict(status="failed", error=str(error)))
        raise
    finally:
        if pool:
            pool.shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("shared", "fusion"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(2050, 2055)))
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--report", action="store_true", help="Evaluate only after all predefined candidates are frozen")
    args = parser.parse_args()
    run_comparison(read_config(args.config), args.output, args.mode, args.seeds,
                   args.jobs, args.threads, args.device, args.report)


if __name__ == "__main__":
    main()
