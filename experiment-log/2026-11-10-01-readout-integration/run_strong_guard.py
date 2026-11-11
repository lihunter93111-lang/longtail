"""Calibrate local corrections against the stronger weighted-ensemble reference."""
import argparse
import copy
import shutil
from common import *
from fit_fusions import fit_guarded_thresholds
from combined_readout import load_combined
from fusion_runtime import load_fusion
from evaluate_stage import evaluate_split, pair_comparison


def load_any(path):
    state = torch.load(path,map_location="cpu",weights_only=False)
    loaders = {"w32-readout-integration-v1":load_bundle,"w32_static_fusion_v1":load_fusion,
               "w32_combined_readout_v1":load_combined}
    return loaders[state["format"]](path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split",choices=CONFIG["splits"],required=True)
    args = parser.parse_args()
    setup()
    designs = {"weighted_segment_guarded":[2,4,5],"weighted_isrj_guarded":[5]}
    borrowed = {"weighted_affine":"fusions_controls","weighted_segment_local":"combined_weighted",
                "weighted_segment_isrj":"combined_weighted_ablations"}
    protocol_path = ROOT/args.split/"strong_guard_protocol.json"
    write(protocol_path,{"designs":designs,"reference":"weighted_affine","max_calibration_weak_recall_drop":0,
          "source_sha256":base.sha(Path(__file__)),"calibrator_sha256":base.sha(HERE/"fit_fusions.py"),
          "goal_sha256":base.sha(HERE/"05H_STRONG_REFERENCE_GUARD_GOAL.txt"),"time":stamp()})
    path,indices = role_paths(args.split)["calibration"]
    inputs,_ = base.load(path,indices)
    records = metadata(path,indices)
    for seed in CONFIG["seeds"]:
        for name,parent in borrowed.items():
            source = ROOT/args.split/parent/f"{name}_{seed}"
            destination = ROOT/args.split/"readouts"/source.name
            destination.mkdir(exist_ok=True)
            for filename in ("model.pth","calibration.json","receipt.json"):
                shutil.copy2(source/filename,destination/filename)
                assert base.sha(source/filename)==base.sha(destination/filename)
        full_path = ROOT/args.split/"combined_weighted"/f"weighted_segment_local_{seed}"
        ref_path = ROOT/args.split/"fusions_controls"/f"weighted_affine_{seed}"
        full,_ = load_combined(full_path/"model.pth")
        reference,_ = load_fusion(ref_path/"model.pth")
        full_z,ref_z = base.predict(full,inputs,"cuda"),base.predict(reference,inputs,"cuda")
        ref_cal = read(ref_path/"calibration.json")
        selected,fallback,guards = fit_guarded_thresholds(records,full_z,ref_z,ref_cal["thresholds"],max_weak_recall_drop=0)
        for name,targets in designs.items():
            check_deadline()
            model = copy.deepcopy(full)
            keep = [column for column in targets if column not in fallback]
            model.fallback = [column for column in full.fallback if column in keep]
            thresholds = np.asarray(ref_cal["thresholds"]).copy()
            thresholds[keep] = np.asarray(selected)[keep]
            expected = ref_z.copy()
            expected[:,keep] = full_z[:,keep]
            for column in set(range(6))-set(keep):
                if column in reference.fusion_state.get("fallback_columns",[]):
                    weights,bias = [1.,0.,0.],0.
                else:
                    row = reference.fusion_state["targets"].index(LABELS[column])
                    weights = [*reference.fusion_state["weights"][2*row:2*row+2],0.]
                    bias = reference.fusion_state["biases"][row]
                model.coefficients["weights"][column] = weights
                model.coefficients["biases"][column] = bias
            folder = ROOT/args.split/"readouts"/f"{name}_{seed}"
            folder.mkdir(exist_ok=True)
            info = {"targets":targets,"kept_local_columns":keep,"calibration_fallback_columns":fallback,
                "reference_sha256":base.sha(ref_path/"model.pth"),"local_source_sha256":base.sha(full_path/"model.pth"),
                "protocol_sha256":base.sha(protocol_path),"time":stamp()}
            model.save(folder/"model.pth",thresholds,info)
            restored,_ = load_combined(folder/"model.pth")
            actual = base.predict(restored,inputs,"cuda")
            np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-12)
            np.testing.assert_array_equal(expit(actual)>=thresholds,expit(expected)>=thresholds)
            prediction,ref_prediction = expit(actual)>=thresholds,expit(ref_z)>=ref_cal["thresholds"]
            for column in targets:
                weak = records["weak_mask"][:,column]
                negative = records["target"][:,column]==0
                assert prediction[weak,column].mean()>=ref_prediction[weak,column].mean()-1e-12
                assert prediction[negative,column].mean()<=ref_prediction[negative,column].mean()+.005+1e-12
            write(folder/"calibration.json",{"thresholds":thresholds.tolist(),"original_thresholds":ref_cal["original_thresholds"],
                  "guards":guards,"max_weak_recall_drop":0})
            write(folder/"receipt.json",{**info,"checkpoint_sha256":base.sha(folder/"model.pth"),
                  "normal_forward_max_error":float(abs(actual-expected).max())})
            print("STRONG_GUARD_FROZEN",args.split,name,seed,keep,flush=True)
            del model,restored
        del full,reference
    variants = [*borrowed,*designs]
    evaluate_split(args.split,"development",variants,"strong_guard",load_any)
    destination = ROOT/args.split/"development_results_strong_guard"
    summary = read(destination/"summary.json")
    manifest = read(ROOT/args.split/"development/manifest.json")
    development = metadata(manifest["path"],manifest["indices"])
    for candidate in designs:
        for reference_name in borrowed:
            summary["comparisons"].append(pair_comparison(candidate,reference_name,destination,development))
    write(destination/"summary.json",summary)
    write(ROOT/args.split/"strong_guard_status.json",{"status":"completed","time":stamp()})
    print("STRONG_GUARD_COMPLETE",args.split,json.dumps(summary["means"]),flush=True)


if __name__=="__main__":
    main()
