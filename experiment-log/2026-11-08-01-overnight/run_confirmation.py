"""One-time evaluation of the frozen final reference and candidate."""
import argparse
from datetime import datetime,timezone
from pathlib import Path

import torch
import numpy as np
from scipy.special import expit,logit
from scipy.stats import t as student_t

import training
from run_stage2 import read

helper = training.previous
base = training.base
HERE = Path(__file__).resolve().parent


def load_declared(record):
    if record["model"]=="dfms_iq_global_statistics_bypass":
        from run_global_control import build_control
        model = build_control(record["model"],num_classes=6,width=record["width"],dropout=.2).cuda()
    else:
        model = base.build_backbone(record["model"],num_classes=6,width=record["width"],dropout=.2).cuda()
    path = Path(record["checkpoint"])
    assert base.sha(path)==record["checkpoint_sha256"]
    model.load_state_dict(torch.load(path,map_location="cuda",weights_only=False)["model_state_dict"])
    return model


def probability_variants(z,record):
    calibration = record["calibration"]
    threshold = np.asarray(calibration["thresholds"]["raw"],dtype=float)
    yield "raw",z,threshold
    cut = logit(threshold)
    temperature = calibration["T"]
    yield "temperature",z/temperature,expit(cut/temperature)
    if record.get("affine") is not None:
        parameters = record["affine"]["parameters"]
        scale = np.array([p["scale"] for p in parameters])
        bias = np.array([p["bias"] for p in parameters])
        assert (scale>0).all()
        yield "class_affine",z*scale+bias,expit(cut*scale+bias)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",type=Path,required=True)
    parser.add_argument("--freeze",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    freeze = read(args.freeze)
    manifest = read(args.data/"manifest.json")
    assert freeze["all_models_and_calibrations_frozen"]
    assert manifest["candidate_freeze_sha256"]==base.sha(args.freeze)
    assert base.sha(Path(manifest["dev_path"]))==manifest["dev_sha256"]
    for source,digest in freeze["source_sha256"].items():
        assert base.sha(Path(source))==digest,source
    access_path = args.data/"evaluation_access.json"
    access = {"candidate_freeze_sha256":base.sha(args.freeze),"manifest_sha256":base.sha(args.data/"manifest.json"),
        "evaluator_sha256":base.sha(Path(__file__)),"role":"reserved_final_confirmation","locked_test_used":False}
    if access_path.exists():
        old = read(access_path)
        assert all(old[key]==value for key,value in access.items())
    else:
        base.write(access_path,{**access,"first_access_utc":datetime.now(timezone.utc).isoformat()})
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    x,y = base.load(manifest["dev_path"],manifest["development"])
    x = x.cuda()
    codes = y.astype(int)@(1<<np.arange(6))
    quotas = read(HERE.parent/"2026-11-06-01-lt-abc/2026-11-06-lt-abc-quotas.json")
    identity = read(args.data/"development_identities.json")
    powers = np.array([[np.nan if v is None else v for v in row["component_JNR_dB"]] for row in identity])
    gap = np.full_like(powers,np.nan)
    for j in range(6):
        others = powers.copy()
        others[:,j] = -np.inf
        maxima = np.max(np.where(np.isfinite(others),others,-np.inf),axis=1)
        mask = np.isfinite(powers[:,j]) & np.isfinite(maxima)
        gap[mask,j] = maxima[mask]-powers[mask,j]
    paired_views = {g:helper.load_views(args.data/f"diagnostic_gap{g}.mat") for g in (0,5,10,15)}
    common15 = paired_views[15][1]
    rows,label_rows,combo_rows,slices,pairs = [],[],[],[],[]
    raw_results = {}
    per_sample = {}
    for record in freeze["models"]:
        name = record["id"]
        folder = args.output/name
        folder.mkdir(exist_ok=True)
        model = load_declared(record)
        path = folder/"natural_logits.npz"
        if path.exists():
            saved = np.load(path)
            np.testing.assert_array_equal(saved["labels"],y)
            np.testing.assert_array_equal(saved["indices"],manifest["development"])
            raw = saved["logits"]
        else:
            raw = helper.logits(model,x).cpu().numpy().astype(np.float64)
            np.savez_compressed(path,logits=raw,labels=y,indices=np.array(manifest["development"]),codes=codes)
        meta = {"group":record["group"],"seed":record["seed"],"model":record["model"],"width":record["width"]}
        raw_prediction = expit(raw)>=np.asarray(record["calibration"]["thresholds"]["raw"])
        for variant,z,thresholds in probability_variants(raw,record):
            metrics,labels,combos,loss = helper.metrics(z,y,thresholds,quotas["tail_codes"])
            np.testing.assert_array_equal(expit(z)>=thresholds,raw_prediction)
            tags = {**meta,"variant":variant}
            rows.append({**tags,**metrics})
            label_rows.extend({**tags,**r} for r in labels)
            combo_rows.extend({**tags,**r} for r in combos)
            if variant=="raw":
                mask = gap[:,5]>=10
                metrics.update(FMB_FPR=labels[0]["FPR"],SawFM_FPR=labels[2]["FPR"],
                    ISRJ_weak_FNR=float((~raw_prediction[mask,5]).mean()))
                raw_results[record["group"],record["seed"]] = metrics
                per_sample[record["group"],record["seed"]] = loss
        losses = helper.binary_loss(raw,y)
        for cardinality in range(1,7):
            mask = y.sum(1)==cardinality
            slices.append({**meta,"slice":"cardinality","group_value":cardinality,"label":"all","n":int(mask.sum()),
                "BCE":float(losses[mask].mean()),"exact":float((raw_prediction[mask]==y[mask]).all(1).mean())})
        for label in range(6):
            for low in (10,15,20,25):
                mask = np.isfinite(powers[:,label]) & (powers[:,label]>=low) & (powers[:,label]<low+5)
                slices.append({**meta,"slice":"positive_JNR","group_value":f"{low}-{low+5}","label":helper.LABELS[label],
                    "n":int(mask.sum()),"BCE":float(losses[mask,label].mean()),"FNR":float((~raw_prediction[mask,label]).mean())})
            mask = gap[:,label]>=10
            slices.append({**meta,"slice":"positive_gap","group_value":"10+","label":helper.LABELS[label],
                "n":int(mask.sum()),"BCE":float(losses[mask,label].mean()),"FNR":float((~raw_prediction[mask,label]).mean())})
        for level,(px,positions,targets) in paired_views.items():
            path = folder/f"paired_gap{level}_logits.npz"
            if path.exists():
                saved = np.load(path)
                np.testing.assert_array_equal(saved["positions"],positions)
                np.testing.assert_array_equal(saved["targets"],targets)
                z = saved["logits"]
            else:
                z = helper.logits(model,px.cuda()).cpu().numpy().astype(np.float64)
                np.savez_compressed(path,logits=z,positions=positions,targets=targets)
            population_masks = {"gap15_feasible":np.isin(positions,common15)}
            if level!=15:
                population_masks["gap0_5_10_feasible"] = np.ones(len(positions),dtype=bool)
            target_z = z[np.arange(len(z)),targets]
            thresholds = np.array(record["calibration"]["thresholds"]["raw"])[targets]
            for population,mask in population_masks.items():
                for label in [None,2,4,5]:
                    selected = mask if label is None else mask & (targets==label)
                    pairs.append({**meta,"population":population,"gap":level,
                        "label":"all_targets" if label is None else helper.LABELS[label],"n":int(selected.sum()),
                        "BCE":float(np.logaddexp(0,-target_z[selected]).mean()),
                        "FNR":float((expit(target_z[selected])<thresholds[selected]).mean()),
                        "fixed05_FNR":float((target_z[selected]<0).mean())})
        print("FINAL_MODEL_SCORED",name,flush=True)
    helper.table(args.output/"metrics.csv",rows)
    helper.table(args.output/"labels.csv",label_rows)
    helper.table(args.output/"combinations.csv",combo_rows)
    helper.table(args.output/"natural_slices.csv",slices)
    helper.table(args.output/"paired_power_metrics.csv",pairs)
    seeds = freeze["seeds"]
    gains = np.array([raw_results["reference",seed]["BCE"]-raw_results["candidate",seed]["BCE"] for seed in seeds])
    differences = {key:float(np.mean([raw_results["candidate",seed][key]-raw_results["reference",seed][key] for seed in seeds]))
        for key in raw_results["reference",seeds[0]]}
    reference_mean = float(np.mean([raw_results["reference",seed]["BCE"] for seed in seeds]))
    relative = float(gains.mean()/reference_mean)
    half = float(student_t.ppf(.975,len(seeds)-1)*gains.std(ddof=1)/np.sqrt(len(seeds)))
    checks = {"raw_BCE_reduction_ge2pct":relative>=.02,"positive_seeds_ge4":int((gains>0).sum())>=4,
        "macro_AP_guard":differences["macro_AP"]>=-.001,"exact_guard":differences["f1_exact"]>=-.005,
        "tail_guard":differences["tail_exact"]>=-.01,"FMB_FPR_guard":differences["FMB_FPR"]<=.005,
        "SawFM_FPR_guard":differences["SawFM_FPR"]<=.005,"ISRJ_weak_FNR_guard":differences["ISRJ_weak_FNR"]<=.01}
    decision = {"reference_BCE":reference_mean,"candidate_BCE":float(reference_mean-gains.mean()),
        "relative_BCE_reduction":relative,"paired_BCE_gain":gains.tolist(),
        "seed_t95":[float(gains.mean()-half),float(gains.mean()+half)],
        "identity_bootstrap95":helper.paired_bootstrap(np.mean([per_sample["reference",s]-per_sample["candidate",s] for s in seeds],axis=0),codes),
        "metric_differences":differences,"checks":checks,"passes":all(checks.values()),
        "role":"reserved_final_confirmation","confirmation_consumed":True,"locked_test_used":False,
        "freeze_sha256":base.sha(args.freeze),"manifest_sha256":base.sha(args.data/"manifest.json")}
    base.write(args.output/"decision.json",decision)
    base.write(args.output/"status.json",{"status":"completed","models":len(freeze["models"]),"decision":decision})
    print("FINAL_CONFIRMATION_COMPLETE",decision,flush=True)


if __name__=="__main__":
    main()
