"""Fixed linear probes of frozen stage statistics and local temporal evidence."""
import argparse
import json
import pickle
from pathlib import Path

import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

import training
from run_stage2 import load_model,read

helper = training.previous
base = training.base
HERE = Path(__file__).resolve().parent


def statistics(values):
    return torch.cat([values.mean(-1),values.std(-1,unbiased=False)],dim=1)


def encode(model,x):
    collected = {}
    model.eval()
    with torch.no_grad():
        for batch in x.split(64):
            maps = model.forward_structure_maps(batch.cuda())
            outputs = {"global128":maps["global"],"temporal_mean_std":statistics(maps["temporal"])}
            top = maps["temporal"].topk(8,dim=-1).values.mean(-1)
            outputs["temporal_with_top8"] = torch.cat([outputs["temporal_mean_std"],top],dim=1)
            for stage in (1,2,3,4):
                real,imag = maps[f"stage{stage}_real"],maps[f"stage{stage}_imag"]
                magnitude = torch.sqrt(real.square()+imag.square()+1e-8)
                outputs[f"stage{stage}_magnitude"] = statistics(magnitude)
            for name,value in outputs.items():
                collected.setdefault(name,[]).append(value.cpu().numpy())
    return {name:np.concatenate(values) for name,values in collected.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--models",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--arm",choices=["W","M"],default="W")
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    manifest = read(args.source/"manifest.json")
    protocol = {"arm":args.arm,"seeds":list(range(2080,2085)),"probe":"six independent logistic regressions",
        "C":1.0,"max_iter":300,"tolerance":1e-5,"standardization":"fit only",
        "readouts":["stage1-4 magnitude mean/std","temporal mean/std","temporal mean/std plus top8 mean","global128"],
        "interpretation":"linear separability conditional on these statistics; feature dimensions differ",
        "source_sha256":base.sha(Path(__file__)),"model_freeze_sha256":base.sha(args.models/"freeze.json")}
    base.write(args.output/"protocol.json",protocol)
    data = {role:base.load(manifest["fit_path"] if role=="fit_A" else manifest["dev_path"],manifest[role])
        for role in ("fit_A","early_stop","calibration","development")}
    quotas = read(HERE.parent/"2026-11-06-01-lt-abc/2026-11-06-lt-abc-quotas.json")
    rows = []
    label_rows = []
    for seed in range(2080,2085):
        folder = args.output/f"{args.arm}_{seed}"
        folder.mkdir(exist_ok=True)
        model = load_model(args.models/f"{args.arm}_{seed}")
        encoded = {}
        for role,(x,y) in data.items():
            path = folder/f"{role}_features.npz"
            if path.exists():
                saved = np.load(path)
                encoded[role] = {key:saved[key] for key in saved.files}
            else:
                encoded[role] = encode(model,x)
                np.savez_compressed(path,**encoded[role])
        del model
        for readout in encoded["fit_A"]:
            scaler = StandardScaler().fit(encoded["fit_A"][readout])
            normalized = {role:scaler.transform(features[readout]) for role,features in encoded.items()}
            classifiers = []
            with threadpool_limits(limits=4):
                for label in range(6):
                    classifier = LogisticRegression(C=1.0,max_iter=300,tol=1e-5,solver="lbfgs",random_state=seed)
                    classifier.fit(normalized["fit_A"],data["fit_A"][1][:,label])
                    classifiers.append(classifier)
            z = {role:np.column_stack([classifier.decision_function(features) for classifier in classifiers])
                for role,features in normalized.items()}
            thresholds = helper.fit_thresholds(z["calibration"],data["calibration"][1])
            model_path = folder/f"{readout}.pkl"
            with model_path.open("wb") as stream:
                pickle.dump({"scaler":scaler,"classifiers":classifiers,"thresholds":thresholds},stream)
            base.write(folder/f"{readout}_freeze.json",{"model_sha256":base.sha(model_path),
                "thresholds":thresholds,"iterations":[int(c.n_iter_[0]) for c in classifiers]})
            values,labels,_,_ = helper.metrics(z["development"],data["development"][1],thresholds,quotas["tail_codes"])
            meta = {"arm":args.arm,"seed":seed,"readout":readout,"features":normalized["fit_A"].shape[1],
                "fit_BCE":float(helper.binary_loss(z["fit_A"],data["fit_A"][1]).mean()),
                "early_BCE":float(helper.binary_loss(z["early_stop"],data["early_stop"][1]).mean())}
            rows.append({**meta,**values})
            label_rows.extend({**meta,**row} for row in labels)
            np.savez_compressed(folder/f"{readout}_development_logits.npz",logits=z["development"],labels=data["development"][1])
            print("PROBE_COMPLETE",args.arm,seed,readout,values["BCE"],flush=True)
    helper.table(args.output/"metrics.csv",rows)
    helper.table(args.output/"labels.csv",label_rows)
    base.write(args.output/"status.json",{"status":"completed","probes":len(rows),"role":"adaptive_development"})


if __name__=="__main__":
    main()
