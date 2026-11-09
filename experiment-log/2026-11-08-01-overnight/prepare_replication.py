"""Audit and assign the frozen fresh confirmation roles without reading scores."""
import argparse
import importlib.util
import json
from pathlib import Path

import torch
import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("bce_goal", HERE.parent/"2026-11-06-04-validation-bce-goal/run_bce_goal.py")
goal = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(goal)
base = goal.base


def component_ids(pool, indices):
    identifiers = set()
    for index in indices:
        labels = np.flatnonzero(pool["y"][index])
        identifiers.update(zip(labels.tolist(), pool["components"][index]))
    return identifiers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt",type=Path,required=True)
    parser.add_argument("--original",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    assert args.prompt.is_file()
    quotas = json.loads((HERE.parent/"2026-11-06-01-lt-abc/2026-11-06-lt-abc-quotas.json").read_text(encoding="utf-8"))
    original_manifest = json.loads((args.original/"manifest.json").read_text(encoding="utf-8"))
    original_preflight = json.loads((args.original/"preflight.json").read_text(encoding="utf-8"))
    fit = base.candidates(args.output,"fit_pool",goal.LABELS)
    dev = base.candidates(args.output,"dev_pool",goal.LABELS)
    for role in ("fit_pool","dev_pool"):
        new_config = json.loads((args.output/role/"generation_config.json").read_text(encoding="utf-8"))
        old_config = json.loads((args.original/role/"generation_config.json").read_text(encoding="utf-8"))
        for key in ("radar","params","types","dataset"):
            assert new_config[key] == old_config[key], (role,key)
    assert not set(fit["seeds"]) & set(dev["seeds"])
    assert not set(fit["noise"]) & set(dev["noise"])
    assert not component_ids(fit,range(len(fit["y"]))) & component_ids(dev,range(len(dev["y"])))
    new_noise = set(fit["noise"]) | set(dev["noise"])
    new_samples = set(fit["seeds"]) | set(dev["seeds"])
    registry_ids = np.array([1,2,3,4,6,8])
    new_components = {
        (int(registry_ids[label]),seed)
        for label,seed in component_ids(fit,range(len(fit["y"]))) | component_ids(dev,range(len(dev["y"])))
    }
    history_audit = []
    historical_paths = [Path(original_manifest["fit_path"]),Path(original_manifest["dev_path"])]
    historical_paths += [Path(record["path"]) for record in original_manifest["historical_sources"]]
    previous = Path("E:/LJQ_code/LJQ_code_experiment_storage/weak_component_v1/20261106_01/data")
    previous_manifest = json.loads((previous/"manifest.json").read_text(encoding="utf-8"))
    historical_paths += [Path(previous_manifest["fit_path"]),Path(previous_manifest["dev_path"])]
    first_manifest = json.loads(Path("E:/LJQ_code/LJQ_code_experiment_storage/overnight_v1/20261108_01/data/manifest.json").read_text(encoding="utf-8"))
    historical_paths += [Path(first_manifest["fit_path"]),Path(first_manifest["dev_path"])]
    for path in historical_paths:
        with h5py.File(path,"r") as data:
            noise = np.asarray(data["receiver_noise_seed"]).reshape(-1).astype(np.int64)
            overlap = len(new_noise & set(noise))
            sample_overlap = None
            if "sample_seed" in data:
                sample_overlap = len(new_samples & set(np.asarray(data["sample_seed"]).reshape(-1).astype(np.int64)))
                assert sample_overlap == 0
            derived_samples = (noise - 2*2246822519) % (1<<32)
            derived_overlap = len(new_samples & set(derived_samples))
            assert derived_overlap == 0
            active = np.asarray(data["label_all"],dtype=bool)
            seed_references = np.asarray(data["component_seeds"]).reshape(-1)
            component_overlap = 0
            for index,reference in enumerate(seed_references):
                ids = np.flatnonzero(active[:,index])+1
                seeds = np.asarray(data[reference]).reshape(-1).astype(np.int64)
                expected = (derived_samples[index] + 2246822519 + ids*2654435761) % (1<<32)
                np.testing.assert_array_equal(seeds,expected)
                component_overlap += sum((int(label),int(seed)) in new_components for label,seed in zip(ids,seeds))
            assert component_overlap == 0
        assert overlap == 0
        history_audit.append({"path":str(path),"noise_overlap":overlap,"sample_overlap":sample_overlap,
            "derived_sample_overlap":derived_overlap,"component_overlap":component_overlap,
            "derived_sample_formula_verified_against_all_component_seeds":True,"rows":len(noise)})
        print("HISTORICAL_IDENTITY_AUDITED",str(path),len(noise),flush=True)
    old_components = set()
    for name in ("fit_A","fit_B","early_stop","calibration","development"):
        records = json.loads((args.original/f"{name}_identities.json").read_text(encoding="utf-8"))
        for record in records:
            indices = [j for j in range(6) if record["code"] & (1<<j)]
            old_components.update(zip(indices,record["component_seeds"]))
    assert not old_components & (component_ids(fit,range(len(fit["y"]))) | component_ids(dev,range(len(dev["y"]))))
    rng = np.random.default_rng(20261108)
    fit_indices = []
    for code,count in enumerate(quotas["A_counts"],1):
        available = np.flatnonzero(fit["codes"]==code)
        fit_indices.extend(rng.permutation(available)[:count].tolist())
    rng = np.random.default_rng(20261109)
    roles = {"early_stop":[],"calibration":[],"development":[]}
    for code in range(1,64):
        available = rng.permutation(np.flatnonzero(dev["codes"]==code))
        assert len(available)==96
        roles["early_stop"].extend(available[:16].tolist())
        roles["calibration"].extend(available[16:32].tolist())
        roles["development"].extend(available[32:].tolist())
    assert len(set(fit_indices))==8000
    assert np.bincount(fit["codes"][fit_indices],minlength=64)[1:].tolist()==quotas["A_counts"]
    manifest = {"fit_A":fit_indices,**roles,"fit_path":str(fit["path"]),"dev_path":str(dev["path"]),"fit_sha256":fit["sha256"],"dev_sha256":dev["sha256"],"evaluation_role":"adaptive_development","goal_prompt_sha256":base.sha(args.prompt),"locked_test_used":False}
    base.write(args.output/"manifest.json",manifest)
    for name,pool,indices in [("fit_A",fit,fit_indices)]+[(name,dev,indices) for name,indices in roles.items()]:
        records=[]
        for index in indices:
            records.append({"raw_row":int(index),"source_file_hash":pool["sha256"],"sample_seed":int(pool["seeds"][index]),"receiver_noise_seed":int(pool["noise"][index]),"component_seeds":pool["components"][index],"code":int(pool["codes"][index]),"component_JNR_dB":[None if not np.isfinite(v) else float(v) for v in pool["power"][index]]})
        base.write(args.output/f"{name}_identities.json",records)
        y=pool["y"][indices]
        base.write(args.output/f"{name}_distribution.json",{"n":len(indices),"unique":len(set(indices)),"positive":y.sum(0).tolist(),"counts":np.bincount(pool["codes"][indices],minlength=64)[1:].tolist()})
    base.write(args.output/"preflight.json",{"init_path":original_preflight["init_path"],"init_sha256":original_preflight["init_sha256"],"goal_prompt_sha256":base.sha(args.prompt),"generator_sha256":base.sha(HERE/"generate_replication_candidates.m"),"preparation_sha256":base.sha(Path(__file__)),"historical_overlap_audit":history_audit,"fresh_component_overlap":0,"component_overlap_with_original_selected_roles":0,"role":"adaptive_development","locked_test_used":False})
    print("CONFIRMATION_DATA_AUDIT_PASSED",flush=True)


if __name__=="__main__":
    main()
