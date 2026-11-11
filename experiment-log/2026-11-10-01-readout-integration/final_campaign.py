"""Freeze explicit candidates and evaluate a fresh confirmation cohort once."""
import argparse
from common import *
from adaptive_readouts import load_adaptive
from combined_readout import load_combined
from fusion_runtime import load_fusion
from evaluate_stage import pair_comparison

LOADERS = {"readout": load_bundle, "adaptive": load_adaptive, "fusion": load_fusion, "combined": load_combined}


def freeze(selection_path):
    check_deadline("freeze_cutoff")
    destination = ROOT/"final_freeze.json"
    assert not destination.exists(), "Final freeze already exists"
    selection = read(selection_path)
    assert selection["candidates"][0]["name"]=="baseline"
    assert len({item["name"] for item in selection["candidates"]})==len(selection["candidates"])
    names = {item["name"] for item in selection["candidates"]}
    assert all(a in names and b in names for a,b in selection["comparisons"])
    assert set(selection.get("physical_methods",[]))<=names
    assert all(a in selection["physical_methods"] and b in selection["physical_methods"]
               for a,b in selection.get("physical_comparisons",[]))
    models = {}
    for split in CONFIG["splits"]:
        assert not (ROOT/split/"confirmation/evaluation_pool/generation_receipt.json").exists()
        for item in selection["candidates"]:
            for seed in CONFIG["seeds"]:
                folder = ROOT/split/item["parent"]/f'{item["name"]}_{seed}'
                model_path, calibration_path = folder/"model.pth", folder/"calibration.json"
                calibration = read(calibration_path)
                model,state = LOADERS[item["loader"]](model_path)
                np.testing.assert_array_equal(state["thresholds"],calibration["thresholds"])
                key = f'{split}/{item["name"]}_{seed}'
                models[key] = {**item,"split":split,"seed":seed,"path":str(model_path),
                    "sha256":base.sha(model_path),"calibration_path":str(calibration_path),
                    "calibration_sha256":base.sha(calibration_path),"thresholds":calibration["thresholds"],
                    "parameters":sum(p.numel() for p in model.parameters())}
                del model,state
    sources = {str(p):base.sha(p) for p in HERE.iterdir() if p.suffix in (".py",".m")
               or p.name=="protocol.json" or p.name.endswith("GOAL.txt") or p.name=="GOAL_PROMPT.txt"}
    for relative in ("experiment-log/2026-11-09-01-wide-followup/generate_confirmation_pairs.m",
                     "experiment-log/2026-11-08-01-overnight/generate_absent_views.m"):
        path = SOURCE/relative
        sources[str(path)] = base.sha(path)
    evidence = {str(p):base.sha(p) for split in CONFIG["splits"] for p in (ROOT/split).glob("*results*/summary.json")}
    write(destination,{"all_models_and_calibrations_frozen":True,"selection":selection,"models":models,
          "selection_path":str(selection_path),"selection_sha256":base.sha(selection_path),
          "confirmation_seeds":CONFIG["confirmation_seeds"],"sources":sources,"development_evidence":evidence,
          "time":stamp(),"confirmation_not_generated_at_freeze":True})
    print("FINAL_FREEZE_COMPLETE",len(models),base.sha(destination),flush=True)


def evaluate_confirmation(split):
    frozen = read(ROOT/"final_freeze.json")
    assert frozen["all_models_and_calibrations_frozen"]
    assert read(ROOT/split/"confirmation/preflight.json")["passed"]
    manifest_path = ROOT/split/"confirmation/manifest.json"
    manifest = read(manifest_path)
    assert manifest["freeze_sha256"]==base.sha(ROOT/"final_freeze.json")
    assert manifest["sha256"]==base.sha(manifest["path"])
    destination = ROOT/split/"final_confirmation_results"
    destination.mkdir(exist_ok=True)
    assert not (destination/"summary.json").exists(), "Confirmation already evaluated"
    for path,digest in frozen["sources"].items():
        assert base.sha(path)==digest,(path,"source changed after final freeze")
    write(destination/"evaluation_access.json",{"time":stamp(),"freeze_sha256":base.sha(ROOT/"final_freeze.json"),
          "manifest_sha256":base.sha(manifest_path),"role":"confirmation"})
    inputs,labels = base.load(manifest["path"],manifest["indices"])
    records = metadata(manifest["path"],manifest["indices"])
    np.testing.assert_array_equal(labels,records["target"])
    rows = []
    for key,item in frozen["models"].items():
        if item["split"]!=split:
            continue
        check_deadline("evaluation_cutoff")
        assert base.sha(item["path"])==item["sha256"]
        assert base.sha(item["calibration_path"])==item["calibration_sha256"]
        calibration = read(item["calibration_path"])
        model,state = LOADERS[item["loader"]](item["path"])
        np.testing.assert_array_equal(state["thresholds"],calibration["thresholds"])
        logits = base.predict(model,inputs,"cuda").astype(float)
        metrics = {"calibrated":evaluate(records,logits,calibration["thresholds"]),
                   "fixed_reference":evaluate(records,logits,calibration["original_thresholds"])}
        output = destination/key.split("/")[1]
        output.mkdir(exist_ok=True)
        write(output/"metrics.json",metrics)
        np.savez(output/"predictions.npz",logits=logits,target=labels,identity=records["identity"],
                 thresholds=calibration["thresholds"],original_thresholds=calibration["original_thresholds"])
        write(output/"receipt.json",{"checkpoint_sha256":item["sha256"],"calibration_sha256":item["calibration_sha256"],
              "prediction_sha256":base.sha(output/"predictions.npz"),"metric_sha256":base.sha(output/"metrics.json"),"time":stamp()})
        rows.append({"group":item["name"],"seed":item["seed"],**{k:metrics["calibrated"][k] for k in
                    ("exact","raw_bce","macro_aupr","macro_f1","tail_exact","ECE")}})
        print("FINAL_CONFIRMATION_EVALUATED",split,item["name"],item["seed"],metrics["calibrated"]["exact"],flush=True)
        del model,state
    helper.table(destination/"metrics.csv",rows)
    comparisons = [pair_comparison(a,b,destination,records) for a,b in frozen["selection"]["comparisons"]]
    means = {item["name"]:{k:float(np.mean([r[k] for r in rows if r["group"]==item["name"]])) for k in
             ("exact","raw_bce","macro_aupr","macro_f1","tail_exact","ECE")} for item in frozen["selection"]["candidates"]}
    write(destination/"summary.json",{"means":means,"comparisons":comparisons,"role":"confirmation",
          "freeze_sha256":base.sha(ROOT/"final_freeze.json"),"completed_utc":stamp()})
    print("FINAL_CONFIRMATION_COMPLETE",split,json.dumps(means),flush=True)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action",choices=["freeze","evaluate"])
    parser.add_argument("--selection",type=Path)
    parser.add_argument("--split",choices=CONFIG["splits"])
    options = parser.parse_args()
    setup()
    if options.action=="freeze":
        assert options.selection is not None
        freeze(options.selection)
    else:
        assert options.split is not None
        evaluate_confirmation(options.split)
