"""Calibration-only context-score correction with a fixed regularization recipe."""
import argparse
from pathlib import Path

import torch
import numpy as np
from scipy.special import expit
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score

import training
from run_stage2 import read

helper = training.previous
base = training.base
HERE = Path(__file__).resolve().parent


def transform(z,settings):
    context = np.tanh(z/5)-np.array(settings["context_mean"])
    output = np.empty_like(z)
    for j,parameters in enumerate(settings["classes"]):
        other = [k for k in range(6) if k!=j]
        output[:,j] = parameters["scale"]*z[:,j]+parameters["bias"]+context[:,other]@np.array(parameters["context_weights"])
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models",type=Path,required=True)
    parser.add_argument("--affine",type=Path,required=True)
    parser.add_argument("--counterfactual",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    assert read(args.affine/"protocol.json")["model_freeze_sha256"]==base.sha(args.models/"freeze.json")
    affine = read(args.affine/"freeze.json")
    base.write(args.output/"protocol.json",{"source_sha256":base.sha(Path(__file__)),
        "prompt_sha256":base.sha(HERE/"03D_CONTEXT_SCORE_GOAL.txt"),"model_freeze_sha256":base.sha(args.models/"freeze.json"),
        "fit_role":"calibration","score_role":"adaptive_development","self_penalty":.001,"context_penalty":.01,
        "context":"centered tanh(other logits/5)","parameter_count_per_model":42})
    frozen = {}
    for name in affine:
        if name.split("_")[0] not in ("W","M"):
            continue
        saved = np.load(args.models/name/"calibration_logits.npz")
        z,y = saved["logits"],saved["labels"]
        mean = np.tanh(z/5).mean(0)
        context = np.tanh(z/5)-mean
        classes = []
        for j in range(6):
            other = [k for k in range(6) if k!=j]
            features = context[:,other]
            initial = affine[name]["class_affine"]["parameters"][j]
            initial = np.r_[np.log(initial["scale"]),initial["bias"],np.zeros(5)]
            def objective(parameters):
                log_scale,bias = parameters[:2]
                coefficients = parameters[2:]
                scaled = np.exp(log_scale)*z[:,j]
                values = scaled+bias+features@coefficients
                residual = expit(values)-y[:,j]
                loss = helper.binary_loss(values,y[:,j]).mean()+.001*(log_scale**2+bias**2)+.01*(coefficients**2).sum()
                gradient = np.r_[np.mean(residual*scaled)+.002*log_scale,np.mean(residual)+.002*bias,
                    features.T@residual/len(z)+.02*coefficients]
                return float(loss),gradient
            result = minimize(objective,initial,jac=True,method="L-BFGS-B",
                bounds=[(np.log(.25),np.log(4)),(-2,2)]+[(-2,2)]*5,
                options={"maxiter":400,"ftol":1e-13,"gtol":1e-8})
            assert result.success,result.message
            classes.append({"scale":float(np.exp(result.x[0])),"bias":float(result.x[1]),
                "context_weights":result.x[2:].tolist(),"iterations":int(result.nit),"objective":float(result.fun)})
        settings = {"classes":classes,"context_mean":mean.tolist()}
        settings["thresholds"] = helper.fit_thresholds(transform(z,settings),y)
        frozen[name] = settings
    base.write(args.output/"freeze.json",frozen)
    quotas = read(HERE.parent/"2026-11-06-01-lt-abc/2026-11-06-lt-abc-quotas.json")
    rows,labels,pairs = [],[],[]
    for name,settings in frozen.items():
        arm,seed = name.split("_")
        saved = np.load(args.models/name/"evaluation_logits.npz")
        z,y = saved["logits"],saved["labels"]
        metrics,per_label,_,_ = helper.metrics(transform(z,settings),y,settings["thresholds"],quotas["tail_codes"])
        meta = {"arm":arm,"seed":int(seed),"variant":"context_affine"}
        rows.append({**meta,**metrics,"raw_BCE":float(helper.binary_loss(z,y).mean())})
        labels.extend({**meta,**row} for row in per_label)
        negative = np.load(args.counterfactual/f"{name}_absent_logits.npz")
        neg_z = transform(negative["logits"],settings)
        for gap in (10,15):
            positive = np.load(args.models/name/f"paired_gap{gap}_logits.npz")
            pos_z = transform(positive["logits"],settings)
            lookup = np.searchsorted(negative["positions"],positive["positions"])
            np.testing.assert_array_equal(negative["positions"][lookup],positive["positions"])
            target = positive["targets"]
            p = pos_z[np.arange(len(target)),target]
            n = neg_z[lookup,target]
            threshold = np.array(settings["thresholds"])[target]
            for label in [None,2,4,5]:
                mask = np.ones(len(target),dtype=bool) if label is None else target==label
                pairs.append({**meta,"gap":gap,"label":"all_targets" if label is None else helper.LABELS[label],
                    "identities":int(mask.sum()),"TPR":float((expit(p[mask])>=threshold[mask]).mean()),
                    "matched_absent_FPR":float((expit(n[mask])>=threshold[mask]).mean()),
                    "paired_ranking":float((p[mask]>n[mask]).mean()),
                    "AUC":float(roc_auc_score(np.r_[np.ones(mask.sum()),np.zeros(mask.sum())],np.r_[p[mask],n[mask]]))})
    helper.table(args.output/"metrics.csv",rows)
    helper.table(args.output/"labels.csv",labels)
    helper.table(args.output/"paired_metrics.csv",pairs)
    base.write(args.output/"status.json",{"status":"completed","models":len(frozen),"calibration_only_fit":True})
    print("CONTEXT_CALIBRATION_COMPLETE",flush=True)


if __name__=="__main__":
    main()
