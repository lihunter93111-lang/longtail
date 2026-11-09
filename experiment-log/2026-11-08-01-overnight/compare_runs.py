"""Recompute matched raw-logit comparisons and predeclared guardrails."""
import argparse
import json
from pathlib import Path

import torch
import numpy as np
from scipy.special import expit
from scipy.stats import t as student_t

import training

helper = training.previous
base = training.base
HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def compare(reference,candidate,source,seeds,reference_pattern,candidate_pattern):
    manifest_hash = base.sha(source/"manifest.json")
    assert read(reference/"protocol.json")["manifest_sha256"]==manifest_hash
    assert read(candidate/"protocol.json")["manifest_sha256"]==manifest_hash
    quota = read(HERE.parent/"2026-11-06-01-lt-abc/2026-11-06-lt-abc-quotas.json")
    identity = read(source/"development_identities.json")
    power = np.array([[np.nan if v is None else v for v in row["component_JNR_dB"]] for row in identity])
    other = power.copy()
    other[:,5] = -np.inf
    gap = np.max(np.where(np.isfinite(other),other,-np.inf),axis=1)-power[:,5]
    weak_isrj = np.isfinite(power[:,5]) & (gap>=10)
    metrics = {"reference":[],"candidate":[]}
    per_sample = {"reference":[],"candidate":[]}
    raw_labels = []
    for seed in seeds:
        pairs = []
        for role,root,pattern in (("reference",reference,reference_pattern),("candidate",candidate,candidate_pattern)):
            folder = root/pattern.format(seed=seed)
            saved = np.load(folder/"evaluation_logits.npz")
            z,y = saved["logits"],saved["labels"]
            calibration = read(folder/"calibration.json")
            thresholds = calibration["thresholds"]["raw"]
            values,labels,_,loss = helper.metrics(z,y,thresholds,quota["tail_codes"])
            values["FMB_FPR"] = labels[0]["FPR"]
            values["SawFM_FPR"] = labels[2]["FPR"]
            values["ISRJ_weak_FNR"] = float((expit(z[weak_isrj,5])<thresholds[5]).mean())
            metrics[role].append(values)
            per_sample[role].append(loss)
            raw_labels.extend({"role":role,"seed":seed,**row} for row in labels)
            pairs.append(y)
        np.testing.assert_array_equal(*pairs)
    codes = y.astype(int)@(1<<np.arange(6))
    gains = np.array([a["BCE"]-b["BCE"] for a,b in zip(metrics["reference"],metrics["candidate"])])
    differences = {key:float(np.mean([b[key]-a[key] for a,b in zip(metrics["reference"],metrics["candidate"])])) for key in metrics["reference"][0]}
    reference_bce = float(np.mean([row["BCE"] for row in metrics["reference"]]))
    relative = float(gains.mean()/reference_bce)
    half = float(student_t.ppf(.975,len(seeds)-1)*gains.std(ddof=1)/np.sqrt(len(seeds)))
    checks = {"relative_BCE_reduction_ge2pct":relative>=.02,"positive_seeds_ge4":int((gains>0).sum())>=4,
        "AP_guard":differences["macro_AP"]>=-.001,"exact_guard":differences["f1_exact"]>=-.005,
        "tail_guard":differences["tail_exact"]>=-.01,"FMB_FPR_guard":differences["FMB_FPR"]<=.005,
        "SawFM_FPR_guard":differences["SawFM_FPR"]<=.005,"ISRJ_weak_FNR_guard":differences["ISRJ_weak_FNR"]<=.01}
    return {"reference":str(reference),"candidate":str(candidate),"reference_pattern":reference_pattern,
        "candidate_pattern":candidate_pattern,"seeds":seeds,"reference_BCE":reference_bce,
        "candidate_BCE":float(reference_bce-gains.mean()),"relative_BCE_reduction":relative,
        "paired_BCE_gain":gains.tolist(),"seed_t95":[float(gains.mean()-half),float(gains.mean()+half)],
        "identity_bootstrap95":helper.paired_bootstrap(np.mean(np.array(per_sample["reference"])-np.array(per_sample["candidate"]),axis=0),codes),
        "metric_differences":differences,"checks":checks,"passes":all(checks.values()),
        "metrics":metrics,"per_label":raw_labels,"role":"adaptive_development"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference",type=Path,required=True)
    parser.add_argument("--candidate",type=Path,required=True)
    parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--reference-pattern",default="W_{seed}")
    parser.add_argument("--candidate-pattern",default="M_{seed}")
    parser.add_argument("--seeds",nargs="+",type=int,default=list(range(2080,2085)))
    args = parser.parse_args()
    result = compare(args.reference,args.candidate,args.source,args.seeds,args.reference_pattern,args.candidate_pattern)
    base.write(args.output,result)
    print(json.dumps({key:value for key,value in result.items() if key not in ("metrics","per_label")}),flush=True)


if __name__=="__main__":
    main()
