"""Matched frozen-feature head adaptation on natural fit identities."""
import argparse
import copy
import json
import random
from pathlib import Path

import torch
import numpy as np
from torch import nn
from torch.nn import functional as F

import training
from run_stage2 import load_model, read

helper = training.previous
base = training.base
HERE = Path(__file__).resolve().parent


def features(model, x):
    model.eval()
    with torch.no_grad():
        return torch.cat([model.forward_features(batch.cuda()).cpu() for batch in x.split(64)])


def train_head(folder, original_head, inputs, labels, early_x, early_y, seed, balanced, weak10, pos10, weak15, pos15):
    receipt_path = folder/"receipt.json"
    if receipt_path.exists():
        return read(receipt_path)
    folder.mkdir(parents=True,exist_ok=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    head = copy.deepcopy(original_head).cuda()
    optimizer = torch.optim.AdamW(head.parameters(),lr=1e-3,weight_decay=1e-4)
    generator = torch.Generator().manual_seed(seed)
    positive = labels.sum(0)
    positive_weight = len(labels)/(2*positive)
    negative_weight = len(labels)/(2*(len(labels)-positive))
    weights = labels*positive_weight+(1-labels)*negative_weight if balanced else torch.ones_like(labels)
    best = float("inf")
    history = []
    initial_hash = helper.model_hash(head)
    orders = []
    for epoch in range(1,41):
        head.train()
        epoch_inputs = inputs.clone()
        selected10 = (pos10+epoch)%2==0
        selected15 = ((pos15+epoch)%2==0) & (((epoch+pos15%2)//2)%2==0)
        epoch_inputs[pos10[selected10]] = weak10[selected10]
        epoch_inputs[pos15[selected15]] = weak15[selected15]
        order = torch.randperm(len(inputs),generator=generator)
        orders.append(helper.hashlib.sha256(order.numpy().tobytes()).hexdigest())
        losses = []
        for index in order.cuda().split(32):
            optimizer.zero_grad(set_to_none=True)
            z = head(epoch_inputs[index])
            loss = (F.binary_cross_entropy_with_logits(z,labels[index],reduction="none")*weights[index]).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(head.parameters(),5)
            optimizer.step()
            losses.append(float(loss.detach()))
        head.eval()
        with torch.no_grad():
            early_loss = float(F.binary_cross_entropy_with_logits(head(early_x),early_y))
        history.append({"epoch":epoch,"train_objective":float(np.mean(losses)),"early_BCE":early_loss,"weak_presentations":int(selected10.sum()),"gap15_presentations":int(selected15.sum())})
        if early_loss<best:
            best = early_loss
            helper.atomic_torch_save({"head_state_dict":head.state_dict(),"epoch":epoch,"early_BCE":best},folder/"best_head.pth")
    base.write(folder/"history.json",history)
    receipt = {"seed":seed,"balanced":balanced,"epochs":40,"updates":10000,"initial_state_sha256":initial_hash,
        "order_sha256":orders,"best_early_BCE":best,"checkpoint_sha256":base.sha(folder/"best_head.pth"),
        "positive_weights":positive_weight.cpu().tolist(),"negative_weights":negative_weight.cpu().tolist()}
    assert sum(row["weak_presentations"] for row in history)==20*len(pos10)
    assert sum(row["gap15_presentations"] for row in history)==10*len(pos15)
    base.write(receipt_path,receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--models",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--arms",nargs="+",default=["M"])
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    manifest = read(args.source/"manifest.json")
    protocol = {"arms":args.arms,"seeds":list(range(2080,2085)),"epochs":40,"lr":1e-3,"batch":32,
        "weight_decay":1e-4,"clip":5,"input":"frozen eval-mode128d features with matched M original/gap10/gap15 schedule",
        "initialization":"copy corresponding trained W/M classifier weights","balanced_weights":"N/(2Npos),N/(2Nneg)",
        "selection":"minimum hard unweighted early-stop BCE","calibration":"global T and per-class F1 grid",
        "source_sha256":base.sha(Path(__file__)),"manifest_sha256":base.sha(args.source/"manifest.json"),
        "backbone_freeze_sha256":base.sha(args.models/"freeze.json"),"role":"adaptive_development"}
    path = args.output/"protocol.json"
    if path.exists():
        assert read(path)==protocol
    else:
        base.write(path,protocol)
    raw = {role:base.load(manifest["fit_path"] if role=="fit_A" else manifest["dev_path"],manifest[role])
        for role in ("fit_A","early_stop","calibration","development")}
    weak10,pos10,_ = helper.load_views(args.source/"fit_gap10.mat")
    weak15,pos15,_ = helper.load_views(args.source/"fit_gap15.mat")
    raw["weak10"] = (weak10,None)
    raw["weak15"] = (weak15,None)
    pos10,pos15 = torch.from_numpy(pos10).cuda(),torch.from_numpy(pos15).cuda()
    records = {}
    for arm in args.arms:
        for seed in range(2080,2085):
            name = f"{arm}_{seed}"
            model = load_model(args.models/name)
            feature_path = args.output/f"{name}_features.npz"
            if feature_path.exists():
                saved = np.load(feature_path)
                encoded = {role:torch.from_numpy(saved[role]) for role in raw}
            else:
                encoded = {role:features(model,x) for role,(x,y) in raw.items()}
                np.savez_compressed(feature_path,**{role:x.numpy() for role,x in encoded.items()})
            fit_x,early_x = encoded["fit_A"].cuda(),encoded["early_stop"].cuda()
            fit_y,early_y = torch.from_numpy(raw["fit_A"][1]).cuda(),torch.from_numpy(raw["early_stop"][1]).cuda()
            pair = []
            for variant,balanced in (("hard",False),("balanced",True)):
                folder = args.output/f"{name}_{variant}"
                receipt = train_head(folder,model.classifier,fit_x,fit_y,early_x,early_y,seed,balanced,encoded["weak10"].cuda(),pos10,encoded["weak15"].cuda(),pos15)
                pair.append(receipt)
                head = copy.deepcopy(model.classifier).cuda()
                state = torch.load(folder/"best_head.pth",map_location="cuda",weights_only=False)
                head.load_state_dict(state["head_state_dict"])
                head.eval()
                with torch.no_grad():
                    z = head(encoded["calibration"].cuda()).cpu().numpy().astype(np.float64)
                y = raw["calibration"][1]
                calibration = {**helper.fit_temperature(z,y),"thresholds":{"raw":helper.fit_thresholds(z,y)}}
                base.write(folder/"calibration.json",calibration)
                np.savez_compressed(folder/"calibration_logits.npz",logits=z,labels=y)
                records[folder.name] = {"arm":arm,"seed":seed,"variant":variant,"calibration":calibration,
                    "checkpoint_sha256":receipt["checkpoint_sha256"],"best_epoch":state["epoch"]}
            assert pair[0]["initial_state_sha256"]==pair[1]["initial_state_sha256"]
            assert pair[0]["order_sha256"]==pair[1]["order_sha256"]
            print("HEAD_PAIR_COMPLETE",name,flush=True)
    base.write(args.output/"freeze.json",{"models":records,"protocol_sha256":base.sha(path)})
    quotas = read(HERE.parent/"2026-11-06-01-lt-abc/2026-11-06-lt-abc-quotas.json")
    rows = []
    labels = []
    for name,record in records.items():
        folder = args.output/name
        head = nn.Sequential(nn.Dropout(.2),nn.Linear(128,6)).cuda()
        head.load_state_dict(torch.load(folder/"best_head.pth",map_location="cuda",weights_only=False)["head_state_dict"])
        head.eval()
        saved = np.load(args.output/f'{record["arm"]}_{record["seed"]}_features.npz')
        with torch.no_grad():
            z = head(torch.from_numpy(saved["development"]).cuda()).cpu().numpy().astype(np.float64)
        y = raw["development"][1]
        np.savez_compressed(folder/"evaluation_logits.npz",logits=z,labels=y)
        values,per_label,_,_ = helper.metrics(z,y,record["calibration"]["thresholds"]["raw"],quotas["tail_codes"])
        meta = {key:record[key] for key in ("arm","seed","variant","best_epoch")}
        rows.append({**meta,**values})
        labels.extend({**meta,**row} for row in per_label)
    helper.table(args.output/"metrics.csv",rows)
    helper.table(args.output/"labels.csv",labels)
    base.write(args.output/"status.json",{"status":"completed","heads":len(records),"backbone_frozen":True})
    print("HEAD_STAGE_COMPLETE",flush=True)


if __name__=="__main__":
    main()
