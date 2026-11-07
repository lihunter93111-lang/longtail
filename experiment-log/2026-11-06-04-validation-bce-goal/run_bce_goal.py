"""Matched BCE experiments with resumable training and isolated evaluation."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import platform
import random
import shutil
import sys
import time
import traceback
from pathlib import Path

import torch
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit
from sklearn.metrics import average_precision_score
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("lt_base", HERE.parent / "2026-11-06-01-lt-abc/run_lt_abc.py")
base = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(base)
LABELS = ["FMB", "AMA", "SawFM", "SinFM", "MFT", "ISRJ"]
ARMS = {
    "A": {"smoothing": 0.0, "cosine": False},
    "B": {"smoothing": 0.0, "cosine": True},
    "C": {"smoothing": 0.05, "cosine": False},
    "D": {"smoothing": 0.05, "cosine": True},
}


def table(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def atomic_torch_save(value, path):
    temporary = path.with_suffix(".tmp.pth")
    torch.save(value, temporary)
    temporary.replace(path)


def model_hash(model):
    digest = hashlib.sha256()
    for key, value in model.state_dict().items():
        digest.update(key.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def source_hashes():
    paths = [Path(__file__), Path(base.__file__), HERE / "GOAL_PROMPT.txt"]
    paths += sorted((base.REPO / "projects/CJ_Imbalance_0716/cjlt/models").glob("*.py"))
    paths.append(base.REPO / "projects/CJ_Imbalance_0716/experiments/run_route_a_c1_baseline.py")
    return {str(path.relative_to(base.REPO)): base.sha(path) for path in paths}


def logits(model, inputs):
    model.eval()
    with torch.no_grad():
        return torch.cat([model(batch) for batch in inputs.split(64)])


def binary_loss(z, y):
    z = np.asarray(z, dtype=np.float64)
    return np.logaddexp(0.0, z) - np.asarray(y, dtype=np.float64) * z


def fit_temperature(z, y):
    objective = lambda log_t: float(binary_loss(z / np.exp(log_t), y).mean())
    result = minimize_scalar(objective, bounds=(math.log(.1), math.log(20)), method="bounded", options={"xatol": 1e-10})
    assert result.success
    choices = [0.0, math.log(.1), math.log(20), float(result.x)]
    best = min(choices, key=objective)
    return {"T": math.exp(best), "calibration_BCE_raw": objective(0), "calibration_BCE_scaled": objective(best), "at_boundary": best in choices[1:3]}


def fit_thresholds(z, y):
    probabilities = expit(np.asarray(z, dtype=np.float64))
    grid = np.linspace(.05, .95, 181)
    thresholds = []
    for label in range(6):
        prediction = probabilities[:, label, None] >= grid[None, :]
        truth = y[:, label, None].astype(bool)
        tp = (prediction & truth).sum(axis=0)
        fp = (prediction & ~truth).sum(axis=0)
        fn = (~prediction & truth).sum(axis=0)
        f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
        thresholds.append(float(grid[np.argmax(f1)]))
    return thresholds


def metrics(z, y, thresholds, tail_codes):
    z = np.asarray(z, dtype=np.float64)
    truth = y.astype(bool)
    probabilities = expit(z)
    losses = binary_loss(z, y)
    fixed = z >= 0
    calibrated = probabilities >= np.asarray(thresholds)
    errors = fixed != truth
    confident_wrong = np.where(truth, z, -z) < -math.log(99)
    ap = average_precision_score(y, probabilities, average=None)
    tp = (calibrated & truth).sum(axis=0)
    fp = (calibrated & ~truth).sum(axis=0)
    fn = (~calibrated & truth).sum(axis=0)
    codes = y.astype(np.int64) @ (1 << np.arange(6))
    exact = (calibrated == truth).all(axis=1)
    fixed_exact = (fixed == truth).all(axis=1)
    ece = []
    for label in range(6):
        bins = np.minimum((probabilities[:, label] * 15).astype(int), 14)
        value = 0.0
        for index in range(15):
            mask = bins == index
            if mask.any():
                value += mask.mean() * abs(probabilities[mask, label].mean() - y[mask, label].mean())
        ece.append(float(value))
    result = {
        "BCE": float(losses.mean()), "Brier": float(((probabilities-y)**2).mean()),
        "ECE": float(np.mean(ece)), "macro_AP": float(ap.mean()),
        "fixed05_exact": float(fixed_exact.mean()), "f1_exact": float(exact.mean()),
        "macro_F1": float((2*tp / np.maximum(2*tp+fp+fn, 1)).mean()),
        "tail_exact": float(np.mean([exact[codes == code].mean() for code in tail_codes])),
        "label_error_rate": float(errors.mean()),
        "confident_wrong_fraction": float(confident_wrong.mean()),
        "confident_wrong_loss_share": float(losses[confident_wrong].sum() / max(losses.sum(), 1e-30)),
    }
    per_label = []
    for j, name in enumerate(LABELS):
        per_label.append({"label": name, "BCE": float(losses[:,j].mean()), "AP": float(ap[j]), "ECE": ece[j], "FP": int(fp[j]), "FN": int(fn[j]), "recall": float(tp[j]/max(y[:,j].sum(),1)), "FPR": float(fp[j]/max((1-y[:,j]).sum(),1))})
    per_combo = [{"code": code, "n": int((codes==code).sum()), "BCE": float(losses[codes==code].mean()), "exact": float(exact[codes==code].mean())} for code in range(1,64)]
    return result, per_label, per_combo, losses.mean(axis=1)


def selfcheck(init):
    torch.manual_seed(12345)
    z = torch.randn(12, 6)
    y = torch.randint(0, 2, (12, 6)).float()
    np.testing.assert_allclose(binary_loss(z.numpy(), y.numpy()), F.binary_cross_entropy_with_logits(z, y, reduction="none").numpy(), rtol=2e-6, atol=1e-7)
    torch.testing.assert_close(F.binary_cross_entropy_with_logits(z, y), F.binary_cross_entropy_with_logits(z, (1-0)*y+0/2))
    np.testing.assert_array_equal(z.numpy() >= 0, z.numpy()/3 >= 0)
    temp = fit_temperature(z.numpy(), y.numpy())
    assert temp["calibration_BCE_scaled"] <= temp["calibration_BCE_raw"] + 1e-12
    model = base.build_backbone("dfms_iq", num_classes=6, width=16, dropout=.2).cuda()
    base.initialize_f2(model, init)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    loss = F.binary_cross_entropy_with_logits(model(torch.randn(2,2,3000,device="cuda")), y[:2].cuda())
    loss.backward(); optimizer.step()
    assert torch.isfinite(loss)
    return {"passed": True, "smoke_loss": float(loss.detach()), "parameters": sum(p.numel() for p in model.parameters())}


def initialize(args):
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.source / "manifest.json").read_text(encoding="utf-8"))
    quotas = json.loads((HERE.parent / "2026-11-06-01-lt-abc/2026-11-06-lt-abc-quotas.json").read_text(encoding="utf-8"))
    source_preflight = json.loads((args.source / "preflight.json").read_text(encoding="utf-8"))
    init = Path(source_preflight["init_path"])
    assert base.sha(init) == base.EXPECTED_INIT
    assert len(manifest["fit_A"]) == len(set(manifest["fit_A"])) == 8000
    roles = [set(manifest[name]) for name in ("early_stop", "calibration", "development")]
    assert [len(role) for role in roles] == [1008,1008,4032]
    assert all(not roles[i]&roles[j] for i in range(3) for j in range(i))
    identity_names = ("fit_A", "early_stop", "calibration", "development")
    identities = [json.loads((args.source/f"{name}_identities.json").read_text(encoding="utf-8")) for name in identity_names]
    identity_audit = {}
    for key in ("sample_seed", "receiver_noise_seed"):
        sets = [set(row[key] for row in records) for records in identities]
        assert all(len(values)==len(records) for values,records in zip(sets,identities))
        assert all(not sets[i]&sets[j] for i in range(4) for j in range(i))
        identity_audit[key] = "all roles unique and disjoint"
    sources = source_hashes()
    config = {
        "stage": args.phase, "source": str(args.source.resolve()), "arms": args.arms,
        "seeds": args.seeds, "epochs": 40, "batch": 32, "updates_per_epoch": 250,
        "lr_max": 1e-4, "lr_min": 1e-6, "weight_decay": 1e-4, "clip": 5,
        "arm_settings": {arm: ARMS[arm] for arm in args.arms},
        "checkpoint_selection": "minimum unweighted hard-label early-stop BCE; earliest tie",
        "manifest_sha256": base.sha(args.source/"manifest.json"),
        "init_sha256": base.sha(init), "source_sha256": sources,
        "data_role": "reused exploratory development" if args.phase=="screen" else "fresh identity confirmation",
        "temperature_bounds": [.1,20], "tail_codes": quotas["tail_codes"],
        "locked_test_used": False,
    }
    config_path = args.output / "protocol.json"
    if config_path.exists():
        assert json.loads(config_path.read_text(encoding="utf-8")) == config, "Protocol/source changed; use a new run version"
    else:
        base.write(config_path, config)
        shutil.copy2(HERE/"GOAL_PROMPT.txt", args.output/"GOAL_PROMPT.txt")
        shutil.copy2(Path(__file__), args.output/"run_bce_goal_snapshot.py")
    fingerprint = base.sha(config_path)
    assert base.sha(Path(manifest["fit_path"])) == manifest["fit_sha256"]
    assert base.sha(Path(manifest["dev_path"])) == manifest["dev_sha256"]
    preflight_path = args.output/"preflight.json"
    if not preflight_path.exists():
        checks = selfcheck(init)
        base.write(preflight_path, {"protocol_sha256": fingerprint, "python": sys.version, "platform": platform.platform(), "torch": torch.__version__, "gpu": torch.cuda.get_device_name(), "precision": "FP32 TF32 off", "identity_audit": identity_audit, "selfcheck": checks, "command": sys.argv})
    return manifest, init, config, fingerprint


def train_one(args, model, inputs, labels, early_x, early_y, seed, arm, initial_hash, fingerprint):
    folder = args.output / f"{arm}_{seed}"
    folder.mkdir(exist_ok=True)
    receipt_path = folder/"receipt.json"
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        assert receipt["protocol_sha256"] == fingerprint
        assert base.sha(folder/"best_BCE.pth") == receipt["checkpoint_sha256"]
        return receipt
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(seed)
    history = []; orders = []; best_bce = math.inf; best_ap = -math.inf
    best_epoch = 0; best_ap_epoch = 0; elapsed = 0.0
    recovery_path = folder/"recovery.pth"
    if recovery_path.exists():
        recovery = torch.load(recovery_path, map_location="cpu", weights_only=False)
        assert recovery["protocol_sha256"] == fingerprint
        assert recovery["initial_state_sha256"] == initial_hash
        model.load_state_dict(recovery["model_state_dict"])
        optimizer.load_state_dict(recovery["optimizer"])
        history = recovery["history"]; orders = recovery["orders"]
        best_bce = recovery["best_bce"]; best_ap = recovery["best_ap"]
        best_epoch = recovery["best_epoch"]; best_ap_epoch = recovery["best_ap_epoch"]
        elapsed = recovery["elapsed"]
        restored_bce = float(F.binary_cross_entropy_with_logits(logits(model, early_x), early_y).cpu())
        assert abs(restored_bce-history[-1]["early_BCE"]) < 1e-6
        generator.set_state(recovery["order_rng"])
        random.setstate(recovery["python_rng"]); np.random.set_state(recovery["numpy_rng"])
        torch.set_rng_state(recovery["torch_rng"]); torch.cuda.set_rng_state_all(recovery["cuda_rng"])
    started = time.time()
    for epoch in range(len(history)+1,41):
        settings = ARMS[arm]
        learning_rate = 1e-4
        if settings["cosine"]:
            learning_rate = 1e-6 + (1e-4-1e-6)*(1+math.cos(math.pi*(epoch-1)/39))/2
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        model.train()
        order = torch.randperm(8000, generator=generator)
        assert len(torch.unique(order)) == 8000
        orders.append(hashlib.sha256(order.numpy().tobytes()).hexdigest())
        objective_losses = []; hard_losses = []
        for batch in order.cuda().split(32):
            optimizer.zero_grad(set_to_none=True)
            prediction = model(inputs[batch]); truth = labels[batch]
            target = (1-settings["smoothing"])*truth + settings["smoothing"]/2
            loss = F.binary_cross_entropy_with_logits(prediction, target)
            hard = F.binary_cross_entropy_with_logits(prediction.detach(), truth)
            assert torch.isfinite(loss) and torch.isfinite(hard)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5); optimizer.step()
            objective_losses.append(loss.detach()); hard_losses.append(hard)
        early_z = logits(model, early_x)
        early_bce = float(F.binary_cross_entropy_with_logits(early_z, early_y).cpu())
        early_ap = float(average_precision_score(early_y.cpu().numpy(), expit(early_z.cpu().numpy().astype(np.float64)), average="macro"))
        fit_eval = None
        if epoch % 5 == 0:
            fit_eval = float(F.binary_cross_entropy_with_logits(logits(model,inputs),labels).cpu())
        row = {"seed":seed,"arm":arm,"epoch":epoch,"lr":learning_rate,"train_objective":float(torch.stack(objective_losses).double().mean().cpu()),"train_hard_BCE":float(torch.stack(hard_losses).double().mean().cpu()),"fit_eval_BCE":fit_eval,"early_BCE":early_bce,"early_macro_AP":early_ap}
        history.append(row)
        if early_bce < best_bce:
            best_bce = early_bce; best_epoch = epoch
            atomic_torch_save({"model_state_dict":model.state_dict(),"epoch":epoch,"early_BCE":early_bce},folder/"best_BCE.pth")
        if early_ap > best_ap:
            best_ap = early_ap; best_ap_epoch = epoch
            atomic_torch_save({"model_state_dict":model.state_dict(),"epoch":epoch,"early_macro_AP":early_ap},folder/"best_AP.pth")
        base.write(folder/"history.json",history)
        atomic_torch_save({"protocol_sha256":fingerprint,"initial_state_sha256":initial_hash,"model_state_dict":model.state_dict(),"optimizer":optimizer.state_dict(),"history":history,"orders":orders,"best_bce":best_bce,"best_ap":best_ap,"best_epoch":best_epoch,"best_ap_epoch":best_ap_epoch,"elapsed":elapsed+time.time()-started,"order_rng":generator.get_state(),"python_rng":random.getstate(),"numpy_rng":np.random.get_state(),"torch_rng":torch.get_rng_state(),"cuda_rng":torch.cuda.get_rng_state_all()},recovery_path)
        base.write(args.output/"status.json",{"status":"training",**row})
        if epoch % 5 == 0:
            print(json.dumps(row),flush=True)
    state = torch.load(folder/"best_BCE.pth",map_location="cuda",weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    check = float(F.binary_cross_entropy_with_logits(logits(model,early_x),early_y).cpu())
    assert abs(check-best_bce)<1e-6
    receipt = {"arm":arm,"seed":seed,"epochs":40,"updates":10000,"unique_fit_per_epoch":8000,"initial_state_sha256":initial_hash,"order_sha256":orders,"best_BCE_epoch":best_epoch,"best_AP_epoch":best_ap_epoch,"best_early_BCE":best_bce,"best_early_AP":best_ap,"checkpoint_sha256":base.sha(folder/"best_BCE.pth"),"protocol_sha256":fingerprint,"elapsed_seconds":elapsed+time.time()-started}
    base.write(receipt_path,receipt)
    return receipt


def train_all(args, manifest, init, config, fingerprint):
    x,y = base.load(manifest["fit_path"],manifest["fit_A"])
    ex,ey = base.load(manifest["dev_path"],manifest["early_stop"])
    x=x.cuda();y=torch.from_numpy(y).cuda();ex=ex.cuda();ey=torch.from_numpy(ey).cuda()
    receipts=[]
    for seed in args.seeds:
        paired_hash=None;paired_order=None
        for arm in args.arms:
            random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
            model=base.build_backbone("dfms_iq",num_classes=6,width=16,dropout=.2).cuda()
            base.initialize_f2(model,init)
            initial_hash=model_hash(model)
            paired_hash=paired_hash or initial_hash
            assert initial_hash==paired_hash
            receipt=train_one(args,model,x,y,ex,ey,seed,arm,initial_hash,fingerprint)
            paired_order=paired_order or receipt["order_sha256"]
            assert receipt["order_sha256"]==paired_order
            receipts.append(receipt)
    base.write(args.output/"runs.json",receipts)
    table(args.output/"runs.csv",[{k:v for k,v in r.items() if k!="order_sha256"} for r in receipts])
    del x,y,ex,ey,model
    kx,ky=base.load(manifest["dev_path"],manifest["calibration"]);kx=kx.cuda()
    calibrations={}
    for receipt in receipts:
        name=f'{receipt["arm"]}_{receipt["seed"]}';folder=args.output/name
        model=base.build_backbone("dfms_iq",num_classes=6,width=16,dropout=.2).cuda()
        model.load_state_dict(torch.load(folder/"best_BCE.pth",map_location="cuda",weights_only=False)["model_state_dict"])
        z=logits(model,kx).cpu().numpy().astype(np.float64)
        temperature=fit_temperature(z,ky)
        thresholds={"raw":fit_thresholds(z,ky),"temperature":fit_thresholds(z/temperature["T"],ky)}
        calibration={**temperature,"thresholds":thresholds,"checkpoint_sha256":receipt["checkpoint_sha256"]}
        base.write(folder/"calibration.json",calibration)
        np.savez_compressed(folder/"calibration_logits.npz",logits=z,labels=ky)
        calibrations[name]=calibration
    freeze={"protocol_sha256":fingerprint,"models":calibrations,"all_models_complete":True,"evaluation_role": "development" if args.phase=="screen" else "confirmation"}
    base.write(args.output/"freeze.json",freeze)
    return freeze


def paired_bootstrap(differences, codes):
    rng=np.random.default_rng(20261126)
    groups=np.stack([np.flatnonzero(codes==code) for code in range(1,64)])
    estimates=[]
    for _ in range(100):
        draws=rng.integers(0,groups.shape[1],size=(100,*groups.shape))
        indices=groups[np.arange(63)[None,:,None],draws]
        estimates.extend(differences[indices].mean(axis=(1,2)).tolist())
    return np.percentile(estimates,[2.5,97.5]).tolist()


def score(args, manifest, config, freeze):
    if (args.output/"decision.json").exists():
        return json.loads((args.output/"decision.json").read_text())
    base.write(args.output/"evaluation_access.json",{"freeze_sha256":base.sha(args.output/"freeze.json"),"role":freeze["evaluation_role"],"all_models_and_calibrations_frozen":True,"locked_test_used":False})
    x,y=base.load(manifest["dev_path"],manifest["development"]);x=x.cuda()
    codes=y.astype(np.int64)@(1<<np.arange(6))
    rows=[];label_rows=[];combo_rows=[];per_sample={};results={}
    for name,calibration in freeze["models"].items():
        arm,seed_text=name.split("_");seed=int(seed_text);folder=args.output/name
        assert base.sha(folder/"best_BCE.pth")==calibration["checkpoint_sha256"]
        model=base.build_backbone("dfms_iq",num_classes=6,width=16,dropout=.2).cuda()
        model.load_state_dict(torch.load(folder/"best_BCE.pth",map_location="cuda",weights_only=False)["model_state_dict"])
        raw=logits(model,x).cpu().numpy().astype(np.float64)
        np.testing.assert_array_equal(raw>=0,raw/calibration["T"]>=0)
        np.savez_compressed(folder/"evaluation_logits.npz",logits=raw,labels=y,indices=np.array(manifest["development"]),codes=codes)
        for variant,z in (("raw",raw),("temperature",raw/calibration["T"])):
            values,labels,combos,sample_loss=metrics(z,y,calibration["thresholds"][variant],config["tail_codes"])
            metadata={"arm":arm,"seed":seed,"variant":variant}
            rows.append({**metadata,**values});label_rows.extend([{**metadata,**r} for r in labels]);combo_rows.extend([{**metadata,**r} for r in combos])
            results[arm,seed,variant]=values;per_sample[arm,seed,variant]=sample_loss
    table(args.output/"evaluation_metrics.csv",rows);table(args.output/"evaluation_labels.csv",label_rows);table(args.output/"evaluation_combinations.csv",combo_rows)
    decisions={}
    comparisons=[(arm,"raw","A","raw") for arm in args.arms if arm!="A"]
    comparisons.append(("A","temperature","A","raw"))
    if args.phase=="confirm":
        comparisons += [(arm,"temperature",arm,"raw") for arm in args.arms if arm!="A"]
    for candidate,variant,reference,reference_variant in comparisons:
        candidate_rows=[results[candidate,seed,variant] for seed in args.seeds]
        reference_rows=[results[reference,seed,reference_variant] for seed in args.seeds]
        candidate_mean=float(np.mean([r["BCE"] for r in candidate_rows]));reference_mean=float(np.mean([r["BCE"] for r in reference_rows]))
        gains=[a["BCE"]-b["BCE"] for a,b in zip(reference_rows,candidate_rows)]
        differences={key:float(np.mean([b[key]-a[key] for a,b in zip(reference_rows,candidate_rows)])) for key in candidate_rows[0]}
        checks={"relative_BCE_reduction":1-candidate_mean/reference_mean>=.05,"positive_seeds":sum(value>0 for value in gains)>=(2 if args.phase=="screen" else 4)}
        if variant=="raw":
            checks.update(macro_AP=differences["macro_AP"]>=-.001,f1_exact=differences["f1_exact"]>=-.005,tail_exact=differences["tail_exact"]>=-.01)
        sample_gain=np.mean([per_sample[reference,seed,reference_variant]-per_sample[candidate,seed,variant] for seed in args.seeds],axis=0)
        interval=paired_bootstrap(sample_gain,codes)
        if args.phase=="confirm":
            checks["identity_CI_excludes_zero"]=interval[0]>0
        key=f"{candidate}_{variant}_vs_{reference}_{reference_variant}"
        decisions[key]={"candidate_BCE":candidate_mean,"reference_BCE":reference_mean,"relative_reduction":1-candidate_mean/reference_mean,"paired_BCE_gain":gains,"mean_BCE_gain":float(np.mean(gains)),"sd_BCE_gain":float(np.std(gains,ddof=1)),"identity_bootstrap95_BCE_gain":interval,"metric_differences":differences,"checks":checks,"passes":all(checks.values())}
    selected=None
    if args.phase=="screen":
        eligible=[arm for arm in args.arms if arm!="A" and decisions[f"{arm}_raw_vs_A_raw"]["passes"]]
        if eligible:
            selected=min(eligible,key=lambda arm:decisions[f"{arm}_raw_vs_A_raw"]["candidate_BCE"])
    decision={"phase":args.phase,"comparisons":decisions,"selected_training_candidate":selected,"temperature_candidate":decisions["A_temperature_vs_A_raw"]["passes"],"confirmation_required":args.phase=="screen" and (selected is not None or decisions["A_temperature_vs_A_raw"]["passes"]),"evidence_role":config["data_role"],"locked_test_used":False}
    base.write(args.output/"decision.json",decision)
    base.write(args.output/"status.json",{"status":"completed","decision":decision})
    print(json.dumps(decision),flush=True)
    return decision


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--phase",choices=["screen","confirm"],default="screen")
    parser.add_argument("--arms",nargs="+",choices=list(ARMS),default=list(ARMS))
    parser.add_argument("--seeds",nargs="+",type=int,default=[2050,2051,2052])
    parser.add_argument("--preflight-only",action="store_true")
    args=parser.parse_args()
    assert args.arms[0]=="A" and len(set(args.arms))==len(args.arms)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assert torch.cuda.is_available()
    try:
        manifest,init,config,fingerprint=initialize(args)
        if args.preflight_only:
            print("PREFLIGHT_PASSED",flush=True)
            return
        freeze=train_all(args,manifest,init,config,fingerprint)
        score(args,manifest,config,freeze)
    except Exception:
        failure={"status":"failed","error":traceback.format_exc(),"time":time.time()}
        args.output.mkdir(parents=True,exist_ok=True)
        base.write(args.output/f"failure_{time.time_ns()}.json",failure)
        base.write(args.output/"status.json",failure)
        raise


if __name__=="__main__":
    main()
