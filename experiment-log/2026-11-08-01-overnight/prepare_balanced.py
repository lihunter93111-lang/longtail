"""Create a cardinality-matched near-balanced-label training control."""
import argparse
from pathlib import Path

import torch
import numpy as np
from scipy.optimize import milp,Bounds,LinearConstraint

import training
from run_stage2 import read

helper = training.previous
base = training.base
HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    original = read(args.source/"manifest.json")
    distribution = read(args.source/"fit_A_distribution.json")
    original_counts = np.array(distribution["counts"])
    codes = np.arange(1,64)
    labels = ((codes[:,None]>>np.arange(6))&1).astype(int)
    cardinality = labels.sum(1)
    cardinality_counts = np.array([original_counts[cardinality==k].sum() for k in range(1,7)])
    target_margins = np.array([4150,4150,4150,4149,4149,4149])
    assert target_margins.sum()==int((cardinality_counts*np.arange(1,7)).sum())
    target_counts = np.array([cardinality_counts[k-1]/sum(cardinality==k) for k in cardinality])
    equalities = np.vstack([labels.T,*[(cardinality==k).astype(int) for k in range(1,7)]])
    equal_targets = np.r_[target_margins,cardinality_counts]
    identity = np.eye(63)
    constraints = [LinearConstraint(np.c_[equalities,np.zeros((12,63))],equal_targets,equal_targets),
        LinearConstraint(np.c_[identity,-identity],-np.inf,target_counts),
        LinearConstraint(np.c_[-identity,-identity],-np.inf,-target_counts)]
    result = milp(c=np.r_[np.zeros(63),np.ones(63)],integrality=np.r_[np.ones(63),np.zeros(63)],
        bounds=Bounds(np.r_[np.full(63,41),np.zeros(63)],np.r_[np.full(63,400),np.full(63,np.inf)]),
        constraints=constraints,options={"time_limit":60,"mip_rel_gap":0})
    assert result.success,result.message
    counts = np.rint(result.x[:63]).astype(int)
    np.testing.assert_array_equal(labels.T@counts,target_margins)
    assert counts.sum()==8000 and counts.min()>=41 and counts.max()<=400
    pool = base.candidates(args.source,"fit_pool",helper.LABELS)
    rng = np.random.default_rng(20261108)
    selected = []
    for code,count in zip(codes,counts):
        selected.extend(rng.permutation(np.flatnonzero(pool["codes"]==code))[:count].tolist())
    assert len(selected)==len(set(selected))==8000
    manifest = {**original,"fit_A":selected,"distribution_role":"cardinality-matched near-balanced label control"}
    base.write(args.output/"manifest.json",manifest)
    records = []
    for index in selected:
        records.append({"raw_row":int(index),"source_file_hash":pool["sha256"],"sample_seed":int(pool["seeds"][index]),
            "receiver_noise_seed":int(pool["noise"][index]),"component_seeds":pool["components"][index],
            "code":int(pool["codes"][index]),"component_JNR_dB":[None if not np.isfinite(v) else float(v) for v in pool["power"][index]]})
    base.write(args.output/"fit_A_identities.json",records)
    for role in ("early_stop","calibration","development"):
        base.write(args.output/f"{role}_identities.json",read(args.source/f"{role}_identities.json"))
    y = pool["y"][selected].astype(int)
    base.write(args.output/"fit_A_distribution.json",{"n":8000,"unique":8000,"positive":y.sum(0).tolist(),
        "counts":counts.tolist(),"cardinality_counts":cardinality_counts.tolist(),"cooccurrence":(y.T@y).tolist(),
        "label_IF":float(y.sum(0).max()/y.sum(0).min())})
    base.write(args.output/"preflight.json",{**read(args.source/"preflight.json"),
        "role":"matched distribution control; same audited candidate pools",
        "source_manifest_sha256":base.sha(args.source/"manifest.json"),"preparation_sha256":base.sha(Path(__file__)),
        "quota_solver_success":bool(result.success),"quota_objective":float(result.fun),
        "quota_counts":counts.tolist(),"shared_fit_identities_with_IF3":len(set(selected)&set(original["fit_A"]))})
    power_rows = []
    for j,label in enumerate(helper.LABELS):
        for name,indices in (("IF3",original["fit_A"]),("balanced",selected)):
            values = pool["power"][indices,j]
            values = values[np.isfinite(values)]
            power_rows.append({"distribution":name,"label":label,"n":len(values),"mean_JNR":float(values.mean()),
                "bin10_15":int(((values>=10)&(values<15)).sum()),"bin15_20":int(((values>=15)&(values<20)).sum()),
                "bin20_25":int(((values>=20)&(values<25)).sum()),"bin25_30":int(((values>=25)&(values<=30)).sum())})
    helper.table(args.output/"power_distribution.csv",power_rows)
    print("BALANCED_CONTROL_READY",counts.min(),counts.max(),target_margins.tolist(),flush=True)


if __name__=="__main__":
    main()
