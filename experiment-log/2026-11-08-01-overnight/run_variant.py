"""Matched architecture or initialization diagnostics using frozen stage recipes."""
import argparse
import hashlib
import importlib.util
import json
import random
from datetime import datetime,timezone
from pathlib import Path
from types import SimpleNamespace

import torch
import numpy as np

import training
import run_stage2

training.json = json
helper = training.previous
base = training.base
HERE = Path(__file__).resolve().parent
read = run_stage2.read


def initialize_model(args,seed,init):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if args.initialization=="scratch":
        model = base.build_backbone(args.model,num_classes=6,width=args.width,dropout=.2).cuda()
        return model,{"initialization":"scratch","shared_subset_sha256":None}
    assert args.width==16
    reference = base.build_backbone("dfms_iq",num_classes=6,width=16,dropout=.2).cuda()
    base.initialize_f2(reference,init)
    reference_hash = helper.model_hash(reference)
    reference_rng = torch.get_rng_state()
    torch.manual_seed(seed)
    model = base.build_backbone(args.model,num_classes=6,width=16,dropout=.2).cuda()
    missing,unexpected = model.load_state_dict(reference.state_dict(),strict=False)
    allowed = ("local_statistics_bypass.","cross_stage_pyramid.","evidence_preserving_block.")
    assert not unexpected
    assert all(key.startswith(allowed) for key in missing),missing
    digest = hashlib.sha256()
    for key in reference.state_dict():
        value = model.state_dict()[key]
        torch.testing.assert_close(value,reference.state_dict()[key],rtol=0,atol=0)
        digest.update(key.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    assert digest.hexdigest()==reference_hash
    torch.set_rng_state(reference_rng)
    return model,{"initialization":"shared F2 and matched classifier","shared_subset_sha256":reference_hash,
        "new_module_keys":missing}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--model",default="dfms_iq")
    parser.add_argument("--width",type=int,default=16)
    parser.add_argument("--coverage",choices=["B","W","M"],default="W")
    parser.add_argument("--initialization",choices=["shared","scratch"],default="shared")
    parser.add_argument("--seeds",nargs="+",type=int,default=list(range(2080,2085)))
    parser.add_argument("--hypothesis",required=True)
    args = parser.parse_args()
    args.arms = [args.coverage]
    args.training_deadline = datetime(2026,11,8,23,20,tzinfo=timezone.utc).timestamp()
    args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    manifest = read(args.source/"manifest.json")
    init = Path(read(args.source/"preflight.json")["init_path"])
    assert base.sha(init)==base.EXPECTED_INIT
    assert base.sha(Path(manifest["fit_path"]))==manifest["fit_sha256"]
    assert base.sha(Path(manifest["dev_path"]))==manifest["dev_sha256"]
    source_files = [Path(__file__),HERE/"training.py",HERE/"run_stage2.py",Path(helper.__file__),helper.HERE/"analyze_goal.py",
        HERE/"GOAL_PROMPT.txt",HERE/"04_STRUCTURE_GOAL.txt"]
    source_files += list((base.REPO/"projects/CJ_Imbalance_0716/cjlt/models").glob("*.py"))
    protocol = {"hypothesis":args.hypothesis,"model":args.model,"width":args.width,"coverage":args.coverage,
        "initialization":args.initialization,"seeds":args.seeds,"epochs":40,"batch":32,"updates":10000,
        "lr":[1e-4,1e-6],"weight_decay":1e-4,"dropout":.2,"clip":5,
        "manifest_sha256":base.sha(args.source/"manifest.json"),"source_sha256":{str(p):base.sha(p) for p in source_files},
        "view_sha256":{p.name:base.sha(p) for p in args.source.glob("*gap*.mat")},"role":"adaptive_development"}
    protocol_path = args.output/"protocol.json"
    if protocol_path.exists():
        assert read(protocol_path)==protocol
    else:
        base.write(protocol_path,protocol)
    fingerprint = base.sha(protocol_path)
    x,y = base.load(manifest["fit_path"],manifest["fit_A"])
    ex,ey = base.load(manifest["dev_path"],manifest["early_stop"])
    w10,p10,_ = helper.load_views(args.source/"fit_gap10.mat")
    w15,p15,_ = helper.load_views(args.source/"fit_gap15.mat")
    x,y,ex,ey,w10,w15 = x.cuda(),torch.from_numpy(y).cuda(),ex.cuda(),torch.from_numpy(ey).cuda(),w10.cuda(),w15.cuda()
    p10,p15 = torch.from_numpy(p10).cuda(),torch.from_numpy(p15).cuda()
    receipts = []
    for seed in args.seeds:
        model,initialization = initialize_model(args,seed,init)
        folder = args.output/f"{args.coverage}_{seed}"
        folder.mkdir(exist_ok=True)
        base.write(folder/"initialization.json",{**initialization,"parameters":sum(p.numel() for p in model.parameters())})
        receipt = training.train_one(args,model,x,y,ex,ey,seed,args.coverage,helper.model_hash(model),fingerprint,w10,p10,w15,p15)
        history = read(folder/"history.json")
        assert sum(row["augmented_presentations"] for row in history)==(0 if args.coverage=="B" else 20*len(p10))
        assert sum(row["gap15_presentations"] for row in history)==(10*len(p15) if args.coverage=="M" else 0)
        receipts.append(receipt)
    base.write(args.output/"runs.json",receipts)
    del x,y,ex,ey,w10,w15,model

    def factory(model_name,num_classes=6,width=16,dropout=.2):
        return base.build_backbone(args.model,num_classes=num_classes,width=args.width,dropout=dropout)

    def load_variant(folder):
        model = factory(args.model).cuda()
        model.load_state_dict(torch.load(folder/"best_BCE.pth",map_location="cuda",weights_only=False)["model_state_dict"])
        return model

    # The reused calibration routine resolves this loader in its own module.
    run_stage2.load_model = load_variant
    freeze = run_stage2.calibrate(args,manifest,receipts,fingerprint)
    base.write(args.output/"evaluation_access.json",{"freeze_sha256":base.sha(args.output/"freeze.json"),
        "role":"adaptive_development","locked_test_used":False})
    x,y = base.load(manifest["dev_path"],manifest["development"])
    x = x.cuda()
    quotas = read(HERE.parent/"2026-11-06-01-lt-abc/2026-11-06-lt-abc-quotas.json")
    codes = y.astype(int)@(1<<np.arange(6))
    rows,labels,combos = [],[],[]
    for name,calibration in freeze["models"].items():
        z = helper.logits(load_variant(args.output/name),x).cpu().numpy().astype(np.float64)
        np.savez_compressed(args.output/name/"evaluation_logits.npz",logits=z,labels=y,codes=codes,
            indices=np.array(manifest["development"]))
        for variant,values in (("raw",z),("temperature",z/calibration["T"])):
            metrics,per_label,per_combo,_ = helper.metrics(values,y,calibration["thresholds"][variant],quotas["tail_codes"])
            meta = {"arm":args.coverage,"seed":int(name.split("_")[1]),"variant":variant}
            rows.append({**meta,**metrics})
            labels.extend({**meta,**row} for row in per_label)
            combos.extend({**meta,**row} for row in per_combo)
    helper.table(args.output/"evaluation_metrics.csv",rows)
    helper.table(args.output/"evaluation_labels.csv",labels)
    helper.table(args.output/"evaluation_combinations.csv",combos)
    spec = importlib.util.spec_from_file_location("prior_analysis_variant",helper.HERE/"analyze_goal.py")
    analysis = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(analysis)
    # The prior analysis accepts a helper object; supply the declared model factory.
    analysis_base = SimpleNamespace(**{**vars(base),"build_backbone":factory})
    analysis_helper = SimpleNamespace(**{**vars(helper),"base":analysis_base})
    analysis.natural_slices(args,manifest,freeze,analysis_helper)
    analysis.paired_evaluation(args,manifest,freeze,analysis_helper)
    helper.table(args.output/"summary_metrics.csv",analysis.summarize(rows,("arm","variant"),
        ("BCE","macro_AP","f1_exact","tail_exact","ECE","macro_F1")))
    base.write(args.output/"status.json",{"status":"completed","model":args.model,"width":args.width,
        "coverage":args.coverage,"initialization":args.initialization,"runs":len(receipts)})
    print("VARIANT_COMPLETE",str(args.output),flush=True)


if __name__=="__main__":
    main()
