"""Label-local ablations using already independently fitted affine coefficients."""
import argparse
import copy
from common import *
from combined_readout import load_combined
from fusion_runtime import load_fusion
from evaluate_stage import pair_comparison


def build(split,prefix="combo"):
    output_name = "combined_ablations" if prefix=="combo" else f"combined_{prefix}_ablations"
    source_name = "combined" if prefix=="combo" else f"combined_{prefix}"
    reference_parent = "fusions_controls" if prefix=="weighted" else "fusions"
    output_root = ROOT/split/output_name
    output_root.mkdir(exist_ok=True)
    designs = {f"{prefix}_segment_isrj":[5],f"{prefix}_segment_saw_mft":[2,4]}
    write(output_root/"protocol.json",{"designs":designs,"source_sha256":base.sha(Path(__file__)),
          "goal_sha256":base.sha(HERE/"05E_LABEL_ABLATION_GOAL.txt"),"prefix":prefix,
          "extension_goal_sha256":base.sha(HERE/("05G_WEIGHTED_LOCAL_GOAL.txt" if prefix=="weighted" else "05E_LABEL_ABLATION_GOAL.txt")),"time":stamp()})
    path,indices = role_paths(split)["calibration"]
    inputs,_ = base.load(path,indices)
    for seed in CONFIG["seeds"]:
        full_folder = ROOT/split/source_name/f"{prefix}_segment_local_{seed}"
        reference_folder = ROOT/split/reference_parent/f"{prefix}_affine_{seed}"
        full,_ = load_combined(full_folder/"model.pth")
        reference,_ = load_fusion(reference_folder/"model.pth")
        full_z = base.predict(full,inputs,"cuda")
        reference_z = base.predict(reference,inputs,"cuda")
        full_cal,reference_cal = read(full_folder/"calibration.json"),read(reference_folder/"calibration.json")
        for name,keep in designs.items():
            check_deadline()
            model = copy.deepcopy(full)
            thresholds = np.asarray(reference_cal["thresholds"]).copy()
            thresholds[keep] = np.asarray(full_cal["thresholds"])[keep]
            expected = reference_z.copy()
            expected[:,keep] = full_z[:,keep]
            model.fallback = [column for column in full.fallback if column in keep]
            for column in set(range(6))-set(keep):
                if column in reference.fusion_state.get("fallback_columns",[]):
                    weights,bias = [1.,0.,0.],0.
                else:
                    row = reference.fusion_state["targets"].index(LABELS[column])
                    weights = [*reference.fusion_state["weights"][2*row:2*row+2],0.]
                    bias = reference.fusion_state["biases"][row]
                model.coefficients["weights"][column] = weights
                model.coefficients["biases"][column] = bias
            folder = output_root/f"{name}_{seed}"
            folder.mkdir(exist_ok=True)
            info = {"name":name,"seed":seed,"kept_local_columns":keep,
                    "source_model_sha256":base.sha(full_folder/"model.pth"),
                    "reference_sha256":base.sha(reference_folder/"model.pth"),
                    "protocol_sha256":base.sha(output_root/"protocol.json"),"time":stamp()}
            model.save(folder/"model.pth",thresholds,info)
            restored,_ = load_combined(folder/"model.pth")
            actual = base.predict(restored,inputs,"cuda")
            np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-12)
            np.testing.assert_array_equal(expit(actual)>=thresholds,expit(expected)>=thresholds)
            write(folder/"calibration.json",{"thresholds":thresholds.tolist(),
                  "original_thresholds":reference_cal["original_thresholds"]})
            write(folder/"receipt.json",{**info,"sha256":base.sha(folder/"model.pth"),"replay_max_error":float(abs(actual-expected).max())})
            print("ABLATION_FROZEN",split,name,seed,flush=True)
            del model,restored
        del full,reference
    write(output_root/"status.json",{"status":"completed","methods":list(designs),"time":stamp()})


def evaluate_ablations(split,prefix="combo"):
    output_name = "combined_ablations" if prefix=="combo" else f"combined_{prefix}_ablations"
    source_name = "combined" if prefix=="combo" else f"combined_{prefix}"
    reference_parent = "fusions_controls" if prefix=="weighted" else "fusions"
    methods = read(ROOT/split/output_name/"status.json")["methods"]
    specification = {"baseline":("readouts",load_bundle),f"{prefix}_affine":(reference_parent,load_fusion),
                     f"{prefix}_segment_local":(source_name,load_combined),f"{prefix}_global_local":(source_name,load_combined)}
    specification.update({name:(output_name,load_combined) for name in methods})
    manifest = read(ROOT/split/"development/manifest.json")
    inputs,_ = base.load(manifest["path"],manifest["indices"])
    records = metadata(manifest["path"],manifest["indices"])
    destination = ROOT/split/("ablation_development_results" if prefix=="combo" else f"{prefix}_ablation_development_results")
    destination.mkdir(exist_ok=True)
    rows = []
    for name,(parent,loader) in specification.items():
        for seed in CONFIG["seeds"]:
            check_deadline("evaluation_cutoff")
            folder = ROOT/split/parent/f"{name}_{seed}"
            model,_ = loader(folder/"model.pth")
            calibration = read(folder/"calibration.json")
            logits = base.predict(model,inputs,"cuda").astype(float)
            metrics = {"calibrated":evaluate(records,logits,calibration["thresholds"]),
                       "fixed_reference":evaluate(records,logits,calibration["original_thresholds"])}
            output = destination/f"{name}_{seed}"
            output.mkdir(exist_ok=True)
            write(output/"metrics.json",metrics)
            np.savez(output/"predictions.npz",logits=logits,target=records["target"],identity=records["identity"],
                     thresholds=calibration["thresholds"],original_thresholds=calibration["original_thresholds"])
            rows.append({"group":name,"seed":seed,**{k:metrics["calibrated"][k] for k in
                        ("exact","raw_bce","macro_aupr","macro_f1","tail_exact","ECE")}})
            del model
    pairs = [(name,reference) for name in methods for reference in
             ("baseline",f"{prefix}_affine",f"{prefix}_segment_local",f"{prefix}_global_local")]
    comparisons = [pair_comparison(a,b,destination,records) for a,b in pairs]
    means = {name:{k:float(np.mean([r[k] for r in rows if r["group"]==name])) for k in
             ("exact","raw_bce","macro_aupr","macro_f1","tail_exact","ECE")} for name in specification}
    helper.table(destination/"metrics.csv",rows)
    write(destination/"summary.json",{"means":means,"comparisons":comparisons,"role":"development","time":stamp()})
    print("ABLATIONS_EVALUATED",split,json.dumps(means),flush=True)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split",choices=CONFIG["splits"],required=True)
    parser.add_argument("--prefix",choices=["combo","weighted"],default="combo")
    options = parser.parse_args()
    setup()
    build(options.split,options.prefix)
    evaluate_ablations(options.split,options.prefix)
