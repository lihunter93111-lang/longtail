"""Frozen target-present versus target-removed comparisons on paired identities."""
import argparse
from pathlib import Path

import torch
import h5py
import numpy as np
from scipy.special import expit
from sklearn.metrics import roc_auc_score,average_precision_score

import training
from run_stage2 import read

helper = training.previous
base = training.base
HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--models",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--arms",nargs="+",default=["W","M"])
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    source_protocol = read(args.models/"protocol.json")
    model_name = source_protocol.get("model","dfms_iq")
    width = source_protocol.get("width",16)
    absent_path = args.source/"diagnostic_absent.mat"
    assert read(args.source/"absent_pairing_receipt.json")["status"]=="verified"
    x,positions,targets = helper.load_views(absent_path)
    x = x.cuda()
    with h5py.File(absent_path,"r") as data:
        absent_labels = np.asarray(data["known_label"]).T.astype(bool)
    assert not absent_labels[np.arange(len(targets)),targets].any()
    assert (absent_labels.sum(1)>=1).all()
    freeze = read(args.models/"freeze.json")
    chosen = {name:settings for name,settings in freeze["models"].items() if name.split("_")[0] in args.arms}
    base.write(args.output/"protocol.json",{"source_sha256":base.sha(Path(__file__)),
        "prompt_sha256":base.sha(HERE/"05B_COUNTERFACTUAL_DIAGNOSTIC_GOAL.txt"),
        "model_freeze_sha256":base.sha(args.models/"freeze.json"),"absent_views_sha256":base.sha(absent_path),
        "arms":args.arms,"model":model_name,"width":width,"role":"adaptive_development_counterfactual",
        "thresholds":"existing calibration raw thresholds; no refitting on paired diagnostic"})
    rows = []
    for name,calibration in chosen.items():
        arm,seed = name.split("_")
        folder = args.models/name
        assert base.sha(folder/"best_BCE.pth")==calibration["checkpoint_sha256"]
        if model_name=="dfms_iq_global_statistics_bypass":
            from run_global_control import build_control
            model = build_control(model_name,num_classes=6,width=width,dropout=.2).cuda()
        else:
            model = base.build_backbone(model_name,num_classes=6,width=width,dropout=.2).cuda()
        model.load_state_dict(torch.load(folder/"best_BCE.pth",map_location="cuda",weights_only=False)["model_state_dict"])
        negative = helper.logits(model,x).cpu().numpy().astype(np.float64)
        target_negative = negative[np.arange(len(targets)),targets]
        np.savez_compressed(args.output/f"{name}_absent_logits.npz",logits=negative,labels=absent_labels,
            positions=positions,targets=targets)
        for gap in (0,5,10,15):
            positive = np.load(folder/f"paired_gap{gap}_logits.npz")
            lookup = np.searchsorted(positions,positive["positions"])
            np.testing.assert_array_equal(positions[lookup],positive["positions"])
            np.testing.assert_array_equal(targets[lookup],positive["targets"])
            pos = positive["logits"][np.arange(len(lookup)),positive["targets"]]
            neg = target_negative[lookup]
            labels = positive["targets"]
            thresholds = np.array(calibration["thresholds"]["raw"])[labels]
            for label in [None,2,4,5]:
                mask = np.ones(len(labels),dtype=bool) if label is None else labels==label
                positive_z,negative_z = pos[mask],neg[mask]
                threshold = thresholds[mask]
                truth = np.r_[np.ones(mask.sum()),np.zeros(mask.sum())]
                scores = np.r_[positive_z,negative_z]
                tpr = float((expit(positive_z)>=threshold).mean())
                fpr = float((expit(negative_z)>=threshold).mean())
                rows.append({"arm":arm,"seed":int(seed),"gap":gap,
                    "label":"all_targets" if label is None else helper.LABELS[label],"identities":int(mask.sum()),
                    "TPR":tpr,"matched_absent_FPR":fpr,"balanced_accuracy":.5*(tpr+1-fpr),
                    "AUC":float(roc_auc_score(truth,scores)),"AP":float(average_precision_score(truth,scores)),
                    "paired_ranking":float((positive_z>negative_z).mean()),
                    "mean_paired_logit_difference":float((positive_z-negative_z).mean()),
                    "positive_BCE":float(np.logaddexp(0,-positive_z).mean()),
                    "absent_BCE":float(np.logaddexp(0,negative_z).mean()),
                    "fixed05_TPR":float((positive_z>=0).mean()),"fixed05_absent_FPR":float((negative_z>=0).mean())})
        print("ABSENT_MODEL_COMPLETE",name,flush=True)
    helper.table(args.output/"metrics.csv",rows)
    base.write(args.output/"status.json",{"status":"completed","models":len(chosen),"positive_negative_identity_paired":True,
        "interpretation":"paired ranking requires both controlled observations; per-class AUC/AP use matched positive and negative views"})


if __name__=="__main__":
    main()
