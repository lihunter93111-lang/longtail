"""Compose frozen backbone and frozen classifier checkpoints for replay."""
import argparse
import json
import shutil
from pathlib import Path

import torch
import numpy as np

import training
from run_stage2 import read

helper = training.previous
base = training.base
HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--backbones",type=Path,required=True)
    parser.add_argument("--heads",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    manifest = read(args.source/"manifest.json")
    protocol = {"operation":"compose already frozen backbone and head","manifest_sha256":base.sha(args.source/"manifest.json"),
        "backbone_freeze_sha256":base.sha(args.backbones/"freeze.json"),"head_freeze_sha256":base.sha(args.heads/"freeze.json"),
        "source_sha256":base.sha(Path(__file__)),"arms":{"WH":["W","hard"],"WB":["W","balanced"],"MH":["M","hard"],"MB":["M","balanced"]}}
    base.write(args.output/"protocol.json",protocol)
    models = {}
    for alias,(arm,variant) in protocol["arms"].items():
        for seed in range(2080,2085):
            original = args.backbones/f"{arm}_{seed}"
            adapted = args.heads/f"{arm}_{seed}_{variant}"
            folder = args.output/f"{alias}_{seed}"
            folder.mkdir(exist_ok=True)
            backbone = torch.load(original/"best_BCE.pth",map_location="cpu",weights_only=False)
            head = torch.load(adapted/"best_head.pth",map_location="cpu",weights_only=False)
            state = backbone["model_state_dict"]
            for key,value in head["head_state_dict"].items():
                full_key = "classifier."+key
                assert full_key in state and state[full_key].shape==value.shape
                state[full_key] = value
            helper.atomic_torch_save({"model_state_dict":state,"backbone_epoch":backbone["epoch"],"head_epoch":head["epoch"]},folder/"best_BCE.pth")
            calibration = read(adapted/"calibration.json")
            calibration["checkpoint_sha256"] = base.sha(folder/"best_BCE.pth")
            saved = np.load(adapted/"calibration_logits.npz")
            calibration["thresholds"]["temperature"] = helper.fit_thresholds(saved["logits"]/calibration["T"],saved["labels"])
            base.write(folder/"calibration.json",calibration)
            shutil.copy2(adapted/"calibration_logits.npz",folder/"calibration_logits.npz")
            saved = np.load(adapted/"evaluation_logits.npz")
            np.savez_compressed(folder/"evaluation_logits.npz",logits=saved["logits"],labels=saved["labels"],
                indices=np.array(manifest["development"]),codes=saved["labels"].astype(int)@(1<<np.arange(6)))
            base.write(folder/"composition_receipt.json",{"backbone_path":str(original/"best_BCE.pth"),
                "backbone_sha256":base.sha(original/"best_BCE.pth"),"head_path":str(adapted/"best_head.pth"),
                "head_sha256":base.sha(adapted/"best_head.pth"),"composed_sha256":calibration["checkpoint_sha256"]})
            models[folder.name] = calibration
    base.write(args.output/"freeze.json",{"models":models,"protocol_sha256":base.sha(args.output/"protocol.json"),
        "all_models_complete":True,"evaluation_role":"adaptive_development"})
    print("FROZEN_HEADS_COMPOSED",len(models),flush=True)


if __name__=="__main__":
    main()
