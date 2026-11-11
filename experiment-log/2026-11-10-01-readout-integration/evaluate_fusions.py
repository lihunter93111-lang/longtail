"""Evaluate frozen score-fusion candidates through their normal I/Q entry point."""
import argparse
from common import *
from fusion_runtime import load_fusion
from evaluate_stage import pair_comparison


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=CONFIG["splits"], required=True)
    parser.add_argument("--role", choices=["development", "confirmation"], default="development")
    parser.add_argument("--methods", nargs="+")
    parser.add_argument("--tag", default="")
    args = parser.parse_args()
    setup()
    suffix = "_" + args.tag if args.tag else ""
    model_root = ROOT / args.split / ("fusions" + suffix)
    status = read(model_root / "status.json")
    assert status["status"] == "completed"
    methods = args.methods or status["methods"]
    global_freeze = None
    if args.role == "confirmation":
        global_freeze = read(ROOT / "final_freeze.json")
        assert global_freeze["all_models_and_calibrations_frozen"]
        assert methods == global_freeze["fusion_methods"]
    destination = ROOT / args.split / f"fusion_{args.role}_results{suffix}"
    destination.mkdir(exist_ok=True)
    manifest = read(ROOT / args.split / args.role / "manifest.json")
    assert base.sha(manifest["path"]) == manifest["sha256"]
    model_records = []
    for name in ["baseline",*methods]:
        for seed in CONFIG["seeds"]:
            parent = ROOT / args.split / "readouts" if name == "baseline" else model_root
            path = parent / f"{name}_{seed}" / "model.pth"
            digest = base.sha(path)
            if global_freeze is not None:
                assert global_freeze["models"][f"{args.split}/{name}_{seed}"]["sha256"] == digest
            model_records.append({"name":name,"seed":seed,"path":str(path),"sha256":digest})
    write(destination / "freeze.json", {"models":model_records,"time":stamp(),
          "evaluator_sha256":base.sha(Path(__file__)),"manifest_sha256":base.sha(ROOT/args.split/args.role/"manifest.json")})
    write(destination / "evaluation_access.json", {"time":stamp(),"freeze_sha256":base.sha(destination/"freeze.json"),"role":args.role})
    inputs,labels = base.load(manifest["path"],manifest["indices"])
    records = metadata(manifest["path"],manifest["indices"])
    rows = []
    for item in model_records:
        check_deadline("evaluation_cutoff")
        name = item["name"]
        loader = load_bundle if name == "baseline" else load_fusion
        model,state = loader(item["path"])
        calibration = read(Path(item["path"]).parent/"calibration.json")
        logits = base.predict(model,inputs,"cuda").astype(float)
        output = destination / f'{name}_{item["seed"]}'
        output.mkdir(exist_ok=True)
        metrics = {"calibrated":evaluate(records,logits,calibration["thresholds"]),
                   "fixed_reference":evaluate(records,logits,calibration["original_thresholds"])}
        write(output/"metrics.json",metrics)
        np.savez(output/"predictions.npz",logits=logits,target=labels,identity=records["identity"],
                 thresholds=calibration["thresholds"],original_thresholds=calibration["original_thresholds"])
        row={"group":name,"seed":item["seed"],**{key:metrics["calibrated"][key]
             for key in ("exact","raw_bce","macro_aupr","macro_f1","tail_exact","ECE")}}
        rows.append(row)
        print("FUSION_EVALUATED",args.split,args.role,name,item["seed"],row["exact"],flush=True)
        del model
    helper.table(destination/"metrics.csv",rows)
    pairs=[(name,"baseline") for name in methods]
    if "reference_affine" in methods:
        pairs += [(name,"reference_affine") for name in methods if name != "reference_affine"]
    for suffix in ("mean","affine"):
        if f"combo_{suffix}" in methods and f"ordinary_{suffix}" in methods:
            pairs.append((f"combo_{suffix}",f"ordinary_{suffix}"))
        for group in ("bootstrap", "weighted"):
            if f"{group}_{suffix}" in methods and f"ordinary_{suffix}" in methods:
                pairs.append((f"{group}_{suffix}", f"ordinary_{suffix}"))
    comparisons=[pair_comparison(candidate,reference,destination,records) for candidate,reference in pairs]
    means={group:{key:float(np.mean([row[key] for row in rows if row["group"]==group]))
                  for key in ("exact","raw_bce","macro_aupr","macro_f1","tail_exact","ECE")}
           for group in ["baseline",*methods]}
    write(destination/"summary.json",{"means":means,"comparisons":comparisons,"role":args.role,"completed_utc":stamp()})
    print("FUSION_COMPARISONS_COMPLETE",args.split,args.role,json.dumps(means),flush=True)


if __name__ == "__main__":
    main()
