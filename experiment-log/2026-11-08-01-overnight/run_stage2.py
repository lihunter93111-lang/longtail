"""Five-seed coverage development study with frozen calibration and power tests."""
import argparse
import json
import random
import sys
import time
import traceback
from datetime import datetime, timezone
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


def load_model(folder):
    model = base.build_backbone("dfms_iq",num_classes=6,width=16,dropout=.2).cuda()
    model.load_state_dict(torch.load(folder/"best_BCE.pth",map_location="cuda",weights_only=False)["model_state_dict"])
    return model


def calibrate(args, manifest, receipts, fingerprint):
    x,y = base.load(manifest["dev_path"],manifest["calibration"])
    x = x.cuda()
    models = {}
    for receipt in receipts:
        name = f'{receipt["arm"]}_{receipt["seed"]}'
        folder = args.output/name
        model = load_model(folder)
        z = helper.logits(model,x).cpu().numpy().astype(np.float64)
        temperature = helper.fit_temperature(z,y)
        calibration = {**temperature,"thresholds":{"raw":helper.fit_thresholds(z,y),
            "temperature":helper.fit_thresholds(z/temperature["T"],y)},
            "checkpoint_sha256":base.sha(folder/"best_BCE.pth")}
        base.write(folder/"calibration.json",calibration)
        np.savez_compressed(folder/"calibration_logits.npz",logits=z,labels=y)
        models[name] = calibration
    freeze = {"protocol_sha256":fingerprint,"models":models,"all_models_complete":True,
        "evaluation_role":"adaptive_development","frozen_utc":datetime.now(timezone.utc).isoformat()}
    base.write(args.output/"freeze.json",freeze)
    return freeze


def evaluate(args, manifest, freeze, tail_codes):
    base.write(args.output/"evaluation_access.json",{"freeze_sha256":base.sha(args.output/"freeze.json"),
        "role":"adaptive_development","locked_test_used":False,"time":time.time()})
    x,y = base.load(manifest["dev_path"],manifest["development"])
    x = x.cuda()
    codes = y.astype(int)@(1<<np.arange(6))
    rows = []
    label_rows = []
    combo_rows = []
    sample_losses = {}
    for name,calibration in freeze["models"].items():
        folder = args.output/name
        arm,seed = name.split("_")
        assert base.sha(folder/"best_BCE.pth")==calibration["checkpoint_sha256"]
        z = helper.logits(load_model(folder),x).cpu().numpy().astype(np.float64)
        np.savez_compressed(folder/"evaluation_logits.npz",logits=z,labels=y,codes=codes,
            indices=np.array(manifest["development"]))
        for variant,values in (("raw",z),("temperature",z/calibration["T"])):
            metrics,labels,combos,loss = helper.metrics(values,y,calibration["thresholds"][variant],tail_codes)
            meta = {"arm":arm,"seed":int(seed),"variant":variant}
            rows.append({**meta,**metrics})
            label_rows.extend({**meta,**row} for row in labels)
            combo_rows.extend({**meta,**row} for row in combos)
            if variant=="raw":
                sample_losses[arm,int(seed)] = loss
    helper.table(args.output/"evaluation_metrics.csv",rows)
    helper.table(args.output/"evaluation_labels.csv",label_rows)
    helper.table(args.output/"evaluation_combinations.csv",combo_rows)
    analysis_spec = helper.importlib.util.spec_from_file_location("prior_analysis",helper.HERE/"analyze_goal.py")
    analysis = helper.importlib.util.module_from_spec(analysis_spec)
    analysis_spec.loader.exec_module(analysis)
    slices,weak = analysis.natural_slices(args,manifest,freeze,helper)
    paired,effects = analysis.paired_evaluation(args,manifest,freeze,helper)
    summary = analysis.summarize(rows,("arm","variant"),("BCE","macro_AP","f1_exact","tail_exact","ECE","macro_F1"))
    helper.table(args.output/"summary_metrics.csv",summary)
    comparisons = {}
    for candidate,reference in (("W","B"),("M","W")):
        selected = {arm:[row for row in rows if row["arm"]==arm and row["variant"]=="raw"] for arm in (candidate,reference)}
        gains = np.array([a["BCE"]-b["BCE"] for a,b in zip(selected[reference],selected[candidate])])
        difference = {key:float(np.mean([b[key]-a[key] for a,b in zip(selected[reference],selected[candidate])]))
            for key in ("BCE","macro_AP","f1_exact","tail_exact")}
        relative = float(gains.mean()/np.mean([row["BCE"] for row in selected[reference]]))
        fpr = {label:float(np.mean([row["FPR"] for row in label_rows if row["arm"]==candidate and row["label"]==label and row["variant"]=="raw"])-
            np.mean([row["FPR"] for row in label_rows if row["arm"]==reference and row["label"]==label and row["variant"]=="raw"])) for label in ("FMB","SawFM")}
        weak_isrj = {arm:float(np.mean([row["FNR"] for row in slices if row["arm"]==arm and row["slice"]=="gap" and row["group"]=="10+" and row["label"]=="ISRJ"])) for arm in (candidate,reference)}
        checks = {"BCE_reduction_ge2pct":relative>=.02,"positive_seeds_ge4":int((gains>0).sum())>=4,
            "AP_guard":difference["macro_AP"]>=-.001,"exact_guard":difference["f1_exact"]>=-.005,
            "tail_guard":difference["tail_exact"]>=-.01,"FMB_FPR_guard":fpr["FMB"]<=.005,
            "SawFM_FPR_guard":fpr["SawFM"]<=.005,"ISRJ_weak_FNR_guard":weak_isrj[candidate]-weak_isrj[reference]<=.01}
        half = float(student_t.ppf(.975,4)*gains.std(ddof=1)/np.sqrt(5))
        interval = helper.paired_bootstrap(np.mean([sample_losses[reference,seed]-sample_losses[candidate,seed] for seed in args.seeds],axis=0),codes)
        comparisons[f"{candidate}_vs_{reference}"] = {"relative_BCE_reduction":relative,"paired_gains":gains.tolist(),
            "seed_t95":[float(gains.mean()-half),float(gains.mean()+half)],"identity_bootstrap95":interval,
            "metric_differences":difference,"FPR_increases":fpr,"ISRJ_weak_FNR":weak_isrj,
            "checks":checks,"passes":all(checks.values())}
    decision = {"phase":"coverage_development","comparisons":comparisons,"data_role":"adaptive_development",
        "provisional_candidate":"M" if comparisons["M_vs_W"]["passes"] else "W","time":time.time()}
    base.write(args.output/"decision.json",decision)
    base.write(args.output/"status.json",{"status":"completed","decision":decision})
    print("STAGE2_COMPLETE",json.dumps(decision),flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    args.arms = ["B","W","M"]
    args.seeds = list(range(2080,2085))
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
    for role in ("fit","diagnostic"):
        assert read(args.source/f"{role}_pairing_receipt.json")["status"]=="verified"
    sources = [Path(__file__),HERE/"training.py",Path(helper.__file__),Path(base.__file__),HERE/"GOAL_PROMPT.txt",HERE/"02_COVERAGE_GOAL.txt"]
    sources += list((base.REPO/"projects/CJ_Imbalance_0716/cjlt/models").glob("*.py"))
    protocol = {"arms":args.arms,"seeds":args.seeds,"epochs":40,"batch":32,"lr_max":1e-4,"lr_min":1e-6,
        "source_sha256":{str(path):base.sha(path) for path in sources},"manifest_sha256":base.sha(args.source/"manifest.json"),
        "view_sha256":{p.name:base.sha(p) for p in args.source.glob("*gap*.mat")},"init_sha256":base.sha(init),
        "data_role":"adaptive_development","M_schedule":"20 augmented epochs; feasible identities alternate 10 gap10 and 10 gap15; others 20 gap10"}
    protocol_path = args.output/"protocol.json"
    if protocol_path.exists():
        assert read(protocol_path)==protocol
    else:
        base.write(protocol_path,protocol)
    fingerprint = base.sha(protocol_path)
    x,y = base.load(manifest["fit_path"],manifest["fit_A"])
    ex,ey = base.load(manifest["dev_path"],manifest["early_stop"])
    w10,p10,t10 = helper.load_views(args.source/"fit_gap10.mat")
    w15,p15,t15 = helper.load_views(args.source/"fit_gap15.mat")
    assert set(p15)<=set(p10)
    np.testing.assert_array_equal(t10[np.searchsorted(p10,p15)],t15)
    epochs = np.arange(1,41)[:,None]
    assert np.all((((p15+epochs)%2==0)&(((epochs+p15%2)//2)%2==0)).sum(0)==10)
    x,y,ex,ey,w10,w15 = x.cuda(),torch.from_numpy(y).cuda(),ex.cuda(),torch.from_numpy(ey).cuda(),w10.cuda(),w15.cuda()
    p10,p15 = torch.from_numpy(p10).cuda(),torch.from_numpy(p15).cuda()
    receipts = []
    for seed in args.seeds:
        pair = []
        for arm in args.arms:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            model = base.build_backbone("dfms_iq",num_classes=6,width=16,dropout=.2).cuda()
            base.initialize_f2(model,init)
            initial_hash = helper.model_hash(model)
            receipt = training.train_one(args,model,x,y,ex,ey,seed,arm,initial_hash,fingerprint,w10,p10,w15,p15)
            history = read(args.output/f"{arm}_{seed}"/"history.json")
            assert sum(row["augmented_presentations"] for row in history)==(0 if arm=="B" else 20*len(p10))
            assert sum(row["gap15_presentations"] for row in history)==(10*len(p15) if arm=="M" else 0)
            pair.append(receipt)
            receipts.append(receipt)
        assert len({r["initial_state_sha256"] for r in pair})==1
        assert all(r["order_sha256"]==pair[0]["order_sha256"] for r in pair)
    base.write(args.output/"runs.json",receipts)
    del x,y,ex,ey,w10,w15,model
    freeze = calibrate(args,manifest,receipts,fingerprint)
    quotas = read(HERE.parent/"2026-11-06-01-lt-abc/2026-11-06-lt-abc-quotas.json")
    evaluate(args,manifest,freeze,quotas["tail_codes"])


if __name__=="__main__":
    try:
        main()
    except Exception:
        print(traceback.format_exc(),flush=True)
        raise
