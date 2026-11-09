"""Calibration-only operating points, evaluated on consumed development logits."""
import argparse
import json
from pathlib import Path

import torch
import numpy as np
from scipy.special import expit

import training

helper = training.previous
base = training.base
HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--models",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--reference",type=Path,required=True)
    parser.add_argument("--arms",nargs="+",default=["WB","MH","MB"])
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    protocol = {"reference":"W F1 calibration FPR per seed and label","grid":[.05,.95,.005],
        "FPR_tolerances":[0,.005,.01],"selection":"maximum calibration recall under reference FPR+tolerance; largest threshold for recall ties",
        "data_role":"adaptive_development","source_sha256":base.sha(Path(__file__)),
        "freeze_sha256":base.sha(args.models/"freeze.json"),"reference_freeze_sha256":base.sha(args.reference/"freeze.json"),"arms":args.arms}
    base.write(args.output/"protocol.json",protocol)
    quotas = read(HERE.parent/"2026-11-06-01-lt-abc/2026-11-06-lt-abc-quotas.json")
    identities = read(args.source/"development_identities.json")
    powers = np.array([[np.nan if v is None else v for v in row["component_JNR_dB"]] for row in identities])
    weak = np.zeros(powers.shape,dtype=bool)
    for label in range(6):
        other = powers.copy()
        other[:,label] = -np.inf
        maxima = np.max(np.where(np.isfinite(other),other,-np.inf),axis=1)
        weak[:,label] = np.isfinite(powers[:,label]) & (maxima-powers[:,label]>=10)
    thresholds_by_model = {}
    calibration_curves = []
    grid = np.linspace(.05,.95,181)
    for seed in range(2080,2085):
        ref = np.load(args.reference/f"W_{seed}/calibration_logits.npz")
        ref_thresholds = np.array(read(args.reference/f"W_{seed}/calibration.json")["thresholds"]["raw"])
        ref_negative = ref["labels"]==0
        ref_prediction = expit(ref["logits"])>=ref_thresholds
        budgets = (ref_prediction & ref_negative).sum(0)/ref_negative.sum(0)
        for arm in args.arms:
            name = f"{arm}_{seed}"
            saved = np.load(args.models/name/"calibration_logits.npz")
            prob = expit(saved["logits"])
            truth = saved["labels"].astype(bool)
            variants = {"F1":read(args.models/name/"calibration.json")["thresholds"]["raw"]}
            for tolerance in (0,.005,.01):
                selected = []
                for label in range(6):
                    predicted = prob[:,label,None]>=grid
                    fpr = (predicted & ~truth[:,label,None]).sum(0)/(~truth[:,label]).sum()
                    recall = (predicted & truth[:,label,None]).sum(0)/truth[:,label].sum()
                    feasible = fpr<=budgets[label]+tolerance+1e-12
                    if feasible.any():
                        best_recall = recall[feasible].max()
                        index = np.flatnonzero(feasible & (recall==best_recall))[-1]
                        selected.append(float(grid[index]))
                    else:
                        selected.append(1.0)
                    if tolerance==0:
                        calibration_curves.extend({"arm":arm,"seed":seed,"label":helper.LABELS[label],
                            "threshold":float(threshold),"FPR":float(fp),"recall":float(re)} for threshold,fp,re in zip(grid,fpr,recall))
                variants[f"FPR_plus_{tolerance}"] = selected
            thresholds_by_model[name] = variants
    base.write(args.output/"thresholds_frozen.json",thresholds_by_model)
    helper.table(args.output/"calibration_curves.csv",calibration_curves)
    rows = []
    label_rows = []
    for name,variants in thresholds_by_model.items():
        arm,seed = name.split("_")
        saved = np.load(args.models/name/"evaluation_logits.npz")
        z,y = saved["logits"],saved["labels"]
        for variant,thresholds in variants.items():
            metrics,labels,_,_ = helper.metrics(z,y,thresholds,quotas["tail_codes"])
            meta = {"arm":arm,"seed":int(seed),"variant":variant}
            rows.append({**meta,**metrics})
            missed = (expit(z)<np.array(thresholds)) & (y==1)
            for j,label in enumerate(labels):
                label_rows.append({**meta,**label,"threshold":thresholds[j],"weak_positive_bits":int(weak[:,j].sum()),
                    "weak_FNR":float(missed[:,j][weak[:,j]].mean())})
    helper.table(args.output/"metrics.csv",rows)
    helper.table(args.output/"labels.csv",label_rows)
    base.write(args.output/"status.json",{"status":"completed","models":len(thresholds_by_model),
        "threshold_variants":4,"calibration_only_fit":True,"confirmation_accessed":False})
    print("DECISION_DIAGNOSTICS_COMPLETE",flush=True)


if __name__=="__main__":
    main()
