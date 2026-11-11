"""Apply the common affine fit and guarded calibration to architecture controls."""
import argparse
import shutil
from common import *
from adaptive_readouts import load_adaptive
from fit_fusions import cached_scores, fit_guarded_thresholds
from fusion_runtime import StaticFusion, load_fusion
from cjlt.engine.selective_fusion import fit_affine_fusion, apply_fusion
from evaluate_stage import pair_comparison


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split",choices=CONFIG["splits"],required=True)
    args = parser.parse_args()
    setup()
    assert read(ROOT/args.split/"adaptive_status.json")["status"]=="completed"
    manifest_path = ROOT/args.split/"fusion_fit/manifest.json"
    manifest = read(manifest_path)
    fit_path,fit_indices = Path(manifest["path"]),manifest["indices"]
    cal_path,cal_indices = role_paths(args.split)["calibration"]
    fit_records,cal_records = metadata(fit_path,fit_indices),metadata(cal_path,cal_indices)
    inputs,_ = base.load(cal_path,cal_indices)
    groups = {"global_pca256_affine":"global_stats_pca256","segment_pca256_affine":"segment_pca256",
              "global_head64_affine":"global_stats_head64","segment_head64_affine":"segment_head64"}
    borrowed = ["reference_affine","coverage_affine"]
    output_root = ROOT/args.split/"fusions_adaptive"
    output_root.mkdir(exist_ok=True)
    write(output_root/"protocol.json",{"groups":groups,"borrowed_controls":borrowed,"epochs":300,"lr":.04,
          "fit_manifest_sha256":base.sha(manifest_path),"source_sha256":base.sha(Path(__file__)),
          "runtime_sha256":base.sha(HERE/"fusion_runtime.py"),"goal_sha256":base.sha(HERE/"04B_ADAPTIVE_CALIBRATION_GOAL.txt"),"time":stamp()})
    for seed in CONFIG["seeds"]:
        reference_path = ROOT/args.split/"readouts"/f"baseline_{seed}"
        reference,_ = load_bundle(reference_path/"model.pth")
        thresholds_reference = read(reference_path/"calibration.json")["thresholds"]
        reference_cal = cached_scores(args.split,seed,"baseline","calibration",reference,cal_path,cal_indices)
        for name in borrowed:
            source = ROOT/args.split/"fusions"/f"{name}_{seed}"
            destination = output_root/source.name
            destination.mkdir(exist_ok=True)
            for filename in ("model.pth","calibration.json","receipt.json"):
                shutil.copy2(source/filename,destination/filename)
                assert base.sha(source/filename)==base.sha(destination/filename)
        for name,group in groups.items():
            check_deadline()
            specialist_path = ROOT/args.split/"readouts"/f"{group}_{seed}/model.pth"
            specialist,_ = load_adaptive(specialist_path)
            assert helper.model_hash(reference.backbone)==helper.model_hash(specialist.backbone)
            fit_z = cached_scores(args.split,seed,group,"fit",specialist,fit_path,fit_indices)
            cal_z = cached_scores(args.split,seed,group,"calibration",specialist,cal_path,cal_indices)
            state = fit_affine_fusion(fit_z,np.zeros_like(fit_z),fit_records["target"],seed=seed,epochs=300,learning_rate=.04)
            candidate,_ = apply_fusion(cal_z,np.zeros_like(cal_z),state)
            thresholds,fallback,guards = fit_guarded_thresholds(cal_records,candidate,reference_cal,thresholds_reference)
            state["fallback_columns"] = fallback
            candidate[:,fallback] = reference_cal[:,fallback]
            model = StaticFusion(reference,specialist,state,"specialist_only",True).cuda().eval().requires_grad_(False)
            folder = output_root/f"{name}_{seed}"
            folder.mkdir(exist_ok=True)
            info = {"name":name,"seed":seed,"adaptive_source_sha256":base.sha(specialist_path),
                    "protocol_sha256":base.sha(output_root/"protocol.json"),"time":stamp()}
            model.save(folder/"model.pth",thresholds,info)
            restored,_ = load_fusion(folder/"model.pth")
            actual = base.predict(restored,inputs,"cuda")
            np.testing.assert_allclose(actual,candidate,rtol=1e-5,atol=5e-5)
            np.testing.assert_array_equal(expit(actual)>=thresholds,expit(candidate)>=thresholds)
            write(folder/"calibration.json",{"thresholds":thresholds,"original_thresholds":thresholds_reference,"guards":guards})
            write(folder/"receipt.json",{**info,"sha256":base.sha(folder/"model.pth"),"fallback_columns":fallback,
                  "parameters":sum(p.numel() for p in model.parameters()),"shared_backbone":True,
                  "normal_forward_max_error":float(abs(actual-candidate).max())})
            print("ADAPTIVE_AFFINE_FROZEN",args.split,name,seed,flush=True)
            del model,restored,specialist
        del reference
    write(output_root/"status.json",{"status":"completed","methods":[*borrowed,*groups],"time":stamp()})


if __name__=="__main__":
    main()
