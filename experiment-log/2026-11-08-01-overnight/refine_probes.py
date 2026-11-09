"""Finish unconverged probe fits without changing data, loss, or regularization."""
import argparse
import pickle
import warnings
from pathlib import Path

import torch
import h5py
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

import training
from run_stage2 import read

helper = training.previous
base = training.base
HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--probes",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    manifest = read(args.source/"manifest.json")
    labels_by_role = {}
    for role in ("fit_A","early_stop","calibration","development"):
        index = np.array(manifest[role])
        order = np.argsort(index)
        path = manifest["fit_path"] if role=="fit_A" else manifest["dev_path"]
        with h5py.File(path,"r") as f:
            labels_by_role[role] = np.asarray(f["known_label"][:,index[order]],dtype=np.float32).T[np.argsort(order)]
    base.write(args.output/"protocol.json",{"source_sha256":base.sha(Path(__file__)),"prior_protocol_sha256":base.sha(args.probes/"protocol.json"),
        "reason":"8 of 210 class probes reached max_iter300","new_limit":1500,"warm_start":True,
        "C":1.0,"tol":1e-5,"data_and_regularization_changed":False})
    quotas = read(HERE.parent/"2026-11-06-01-lt-abc/2026-11-06-lt-abc-quotas.json")
    rows,label_rows,repairs = [],[],[]
    for path in sorted(args.probes.glob("*/*.pkl")):
        arm,seed_text = path.parent.name.split("_")
        readout = path.stem
        with path.open("rb") as f:
            model = pickle.load(f)
        normalized = {}
        for role in labels_by_role:
            saved = np.load(path.parent/f"{role}_features.npz")
            normalized[role] = model["scaler"].transform(saved[readout])
        for label,classifier in enumerate(model["classifiers"]):
            if int(classifier.n_iter_[0])>=300:
                classifier.set_params(max_iter=1500,warm_start=True)
                with threadpool_limits(limits=4),warnings.catch_warnings():
                    warnings.simplefilter("error",ConvergenceWarning)
                    classifier.fit(normalized["fit_A"],labels_by_role["fit_A"][:,label])
                assert int(classifier.n_iter_[0])<1500
                repairs.append({"model":path.parent.name,"readout":readout,"label":helper.LABELS[label],
                    "additional_iterations":int(classifier.n_iter_[0])})
        z = {role:np.column_stack([c.decision_function(x) for c in model["classifiers"]]) for role,x in normalized.items()}
        thresholds = helper.fit_thresholds(z["calibration"],labels_by_role["calibration"])
        model["thresholds"] = thresholds
        folder = args.output/path.parent.name
        folder.mkdir(exist_ok=True)
        output_path = folder/path.name
        with output_path.open("wb") as f:
            pickle.dump(model,f)
        base.write(folder/f"{readout}_freeze.json",{"model_sha256":base.sha(output_path),"thresholds":thresholds,
            "all_class_probes_converged":True,"source_model_sha256":base.sha(path)})
        values,labels,_,_ = helper.metrics(z["development"],labels_by_role["development"],thresholds,quotas["tail_codes"])
        meta = {"arm":arm,"seed":int(seed_text),"readout":readout,"features":normalized["fit_A"].shape[1],
            "fit_BCE":float(helper.binary_loss(z["fit_A"],labels_by_role["fit_A"]).mean()),
            "early_BCE":float(helper.binary_loss(z["early_stop"],labels_by_role["early_stop"]).mean())}
        rows.append({**meta,**values})
        label_rows.extend({**meta,**row} for row in labels)
        np.savez_compressed(folder/f"{readout}_development_logits.npz",logits=z["development"],labels=labels_by_role["development"])
    helper.table(args.output/"metrics.csv",rows)
    helper.table(args.output/"labels.csv",label_rows)
    base.write(args.output/"status.json",{"status":"completed","refined_class_fits":repairs,"all210_class_probes_converged":True})
    print("PROBE_CONVERGENCE_REFINED",len(repairs),flush=True)


if __name__=="__main__":
    main()
