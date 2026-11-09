"""Audit the reserved confirmation identities and bind them to the final freeze."""
import argparse
from pathlib import Path

import torch
import h5py
import numpy as np

import training
from run_stage2 import read

helper = training.previous
base = training.base


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--data",type=Path,required=True)
    parser.add_argument("--freeze",type=Path,required=True)
    args = parser.parse_args()
    frozen = read(args.freeze)
    assert frozen["all_models_and_calibrations_frozen"]
    assert not (args.data/"evaluation_access.json").exists()
    pool = base.candidates(args.data,"confirmation_pool",helper.LABELS)
    np.testing.assert_array_equal(np.bincount(pool["codes"],minlength=64)[1:],np.full(63,64))
    np.testing.assert_array_equal(pool["noise"],(pool["seeds"]+2*2246822519)%(1<<32))
    registry_ids = np.array([1,2,3,4,6,8])
    for i,components in enumerate(pool["components"]):
        ids = registry_ids[np.flatnonzero(pool["y"][i])]
        np.testing.assert_array_equal(components,(pool["seeds"][i]+2246822519+ids*2654435761)%(1<<32))
    new_samples = set(pool["seeds"])
    new_noise = set(pool["noise"])
    history = [Path(row["path"]) for row in read(args.root/"data/preflight.json")["historical_overlap_audit"]]
    for role in ("data","replication_data"):
        manifest = read(args.root/role/"manifest.json")
        history += [Path(manifest["fit_path"]),Path(manifest["dev_path"])]
    history = sorted(set(history))
    audits = []
    for path in history:
        with h5py.File(path,"r") as f:
            noise = np.asarray(f["receiver_noise_seed"]).reshape(-1).astype(np.int64)
            sample = (noise-2*2246822519)%(1<<32)
            assert not new_noise & set(noise)
            assert not new_samples & set(sample)
            active = np.asarray(f["label_all"],dtype=bool)
            references = np.asarray(f["component_seeds"]).reshape(-1)
            for i,reference in enumerate(references):
                ids = np.flatnonzero(active[:,i])+1
                actual = np.asarray(f[reference]).reshape(-1).astype(np.int64)
                expected = (sample[i]+2246822519+ids*2654435761)%(1<<32)
                np.testing.assert_array_equal(actual,expected)
        audits.append({"path":str(path),"rows":len(noise),"sample_overlap":0,"noise_overlap":0,
            "per_class_component_overlap":0,"seed_mapping_verified":True})
        print("FINAL_IDENTITY_AUDITED",str(path),flush=True)
    indices = np.concatenate([np.flatnonzero(pool["codes"]==code) for code in range(1,64)]).tolist()
    manifest = {"dev_path":str(pool["path"]),"dev_sha256":pool["sha256"],"development":indices,
        "evaluation_role":"reserved_final_confirmation","candidate_freeze_sha256":base.sha(args.freeze),
        "candidate_freeze_path":str(args.freeze),"locked_test_used":False}
    base.write(args.data/"manifest.json",manifest)
    records = [{"raw_row":int(i),"source_file_hash":pool["sha256"],"sample_seed":int(pool["seeds"][i]),
        "receiver_noise_seed":int(pool["noise"][i]),"component_seeds":pool["components"][i],
        "code":int(pool["codes"][i]),"component_JNR_dB":[None if not np.isfinite(v) else float(v) for v in pool["power"][i]]} for i in indices]
    base.write(args.data/"development_identities.json",records)
    base.write(args.data/"preflight.json",{"status":"passed","identities":4032,"identities_per_combo":64,
        "history_audit":audits,"source_sha256":base.sha(Path(__file__)),"candidate_freeze_sha256":base.sha(args.freeze),
        "role":"reserved_final_confirmation","locked_test_used":False})
    print("FINAL_CONFIRMATION_AUDIT_PASSED",flush=True)


if __name__=="__main__":
    main()
