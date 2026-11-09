"""Replay balanced-head models at calibration-defined FPR operating points."""
import argparse
import importlib.util
import shutil
from pathlib import Path

import torch
import numpy as np
from scipy.special import expit

import training
from run_stage2 import read

helper = training.previous
base = training.base
HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--models",type=Path,required=True)
    parser.add_argument("--thresholds",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    args.arms = ["MB"]
    args.seeds = list(range(2080,2085))
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    manifest = read(args.source/"manifest.json")
    choices = read(args.thresholds)
    protocol = {"manifest_sha256":base.sha(args.source/"manifest.json"),
        "model_freeze_sha256":base.sha(args.models/"freeze.json"),"thresholds_sha256":base.sha(args.thresholds),
        "operating_point":"FPR_plus_0","source_sha256":base.sha(Path(__file__)),"role":"adaptive_development"}
    base.write(args.output/"protocol.json",protocol)
    models = {}
    for arm in args.arms:
        for seed in args.seeds:
            name = f"{arm}_{seed}"
            source = args.models/name
            folder = args.output/name
            folder.mkdir(exist_ok=True)
            for filename in ("best_BCE.pth","evaluation_logits.npz","calibration_logits.npz","composition_receipt.json"):
                shutil.copy2(source/filename,folder/filename)
            calibration = read(source/"calibration.json")
            thresholds = np.array(choices[name]["FPR_plus_0"])
            calibration["thresholds"]["raw"] = thresholds.tolist()
            with np.errstate(divide="ignore"):
                calibration["thresholds"]["temperature"] = expit(np.log(thresholds/(1-thresholds))/calibration["T"]).tolist()
            base.write(folder/"calibration.json",calibration)
            models[name] = calibration
    freeze = {"models":models,"protocol_sha256":base.sha(args.output/"protocol.json"),
        "all_models_complete":True,"evaluation_role":"adaptive_development"}
    base.write(args.output/"freeze.json",freeze)
    quota = read(HERE.parent/"2026-11-06-01-lt-abc/2026-11-06-lt-abc-quotas.json")
    rows,label_rows,combo_rows = [],[],[]
    for name,calibration in models.items():
        arm,seed = name.split("_")
        saved = np.load(args.output/name/"evaluation_logits.npz")
        for variant in ("raw","temperature"):
            z = saved["logits"]/(calibration["T"] if variant=="temperature" else 1)
            values,labels,combos,_ = helper.metrics(z,saved["labels"],calibration["thresholds"][variant],quota["tail_codes"])
            meta = {"arm":arm,"seed":int(seed),"variant":variant}
            rows.append({**meta,**values})
            label_rows.extend({**meta,**r} for r in labels)
            combo_rows.extend({**meta,**r} for r in combos)
    helper.table(args.output/"evaluation_metrics.csv",rows)
    helper.table(args.output/"evaluation_labels.csv",label_rows)
    helper.table(args.output/"evaluation_combinations.csv",combo_rows)
    spec = importlib.util.spec_from_file_location("head_power_analysis",helper.HERE/"analyze_goal.py")
    analysis = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(analysis)
    analysis.natural_slices(args,manifest,freeze,helper)
    analysis.paired_evaluation(args,manifest,freeze,helper)
    helper.table(args.output/"summary_metrics.csv",analysis.summarize(rows,("arm","variant"),
        ("BCE","macro_AP","f1_exact","tail_exact","ECE","macro_F1")))
    base.write(args.output/"status.json",{"status":"completed","models":len(models),"thresholds_fit_on_calibration":True})
    print("HEAD_OPERATING_POINT_POWER_EVALUATION_COMPLETE",flush=True)


if __name__=="__main__":
    main()
