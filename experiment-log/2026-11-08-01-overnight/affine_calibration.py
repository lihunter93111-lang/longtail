"""Regularized classwise calibration fitted exclusively on calibration logits."""
import argparse
from pathlib import Path

import torch
import numpy as np
from scipy.optimize import minimize

import training
from run_stage2 import read

helper = training.previous
base = training.base
HERE = Path(__file__).resolve().parent


def fit_class(z,y,with_bias):
    bounds = [(np.log(.25),np.log(4)),(-2,2) if with_bias else (0,0)]
    def objective(parameters):
        log_a,bias = parameters
        transformed = np.exp(log_a)*z+bias
        return float(helper.binary_loss(transformed,y).mean()+.001*(log_a**2+bias**2))
    result = minimize(objective,np.zeros(2),method="L-BFGS-B",bounds=bounds,
        options={"maxiter":300,"ftol":1e-13,"gtol":1e-8})
    assert result.success,result.message
    return {"scale":float(np.exp(result.x[0])),"bias":float(result.x[1]),
        "objective":float(result.fun),"success":bool(result.success),"iterations":int(result.nit)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    source_freeze = read(args.models/"freeze.json")
    base.write(args.output/"protocol.json",{"regularization":.001,"scale_bounds":[.25,4],"bias_bounds":[-2,2],
        "source_sha256":base.sha(Path(__file__)),"prompt_sha256":base.sha(HERE/"03B_AFFINE_CALIBRATION_GOAL.txt"),
        "model_freeze_sha256":base.sha(args.models/"freeze.json"),"fit_role":"calibration","score_role":"adaptive_development"})
    frozen = {}
    for name in source_freeze["models"]:
        saved = np.load(args.models/name/"calibration_logits.npz")
        z,y = saved["logits"],saved["labels"]
        frozen[name] = {}
        for variant,bias in (("class_temperature",False),("class_affine",True)):
            parameters = [fit_class(z[:,j],y[:,j],bias) for j in range(6)]
            scale = np.array([p["scale"] for p in parameters])
            offset = np.array([p["bias"] for p in parameters])
            values = z*scale+offset
            frozen[name][variant] = {"parameters":parameters,"thresholds":helper.fit_thresholds(values,y),
                "calibration_raw_BCE":float(helper.binary_loss(z,y).mean()),
                "calibration_scaled_BCE":float(helper.binary_loss(values,y).mean())}
    base.write(args.output/"freeze.json",frozen)
    quotas = read(HERE.parent/"2026-11-06-01-lt-abc/2026-11-06-lt-abc-quotas.json")
    rows = []
    labels = []
    for name,variants in frozen.items():
        arm,seed = name.split("_")
        saved = np.load(args.models/name/"evaluation_logits.npz")
        z,y = saved["logits"],saved["labels"]
        for variant,settings in variants.items():
            scale = np.array([p["scale"] for p in settings["parameters"]])
            offset = np.array([p["bias"] for p in settings["parameters"]])
            values = z*scale+offset
            metrics,per_label,_,_ = helper.metrics(values,y,settings["thresholds"],quotas["tail_codes"])
            meta = {"arm":arm,"seed":int(seed),"variant":variant}
            rows.append({**meta,**metrics,"raw_BCE":float(helper.binary_loss(z,y).mean())})
            labels.extend({**meta,**row} for row in per_label)
    helper.table(args.output/"metrics.csv",rows)
    helper.table(args.output/"labels.csv",labels)
    base.write(args.output/"status.json",{"status":"completed","models":len(frozen),"calibration_only_fit":True})
    print("AFFINE_CALIBRATION_COMPLETE",flush=True)


if __name__=="__main__":
    main()
