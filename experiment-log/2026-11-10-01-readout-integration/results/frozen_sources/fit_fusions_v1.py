"""Matched supervised score fusion, guarded calibration, and normal-forward replay."""
import argparse
from common import *
from fusion_runtime import StaticFusion, load_fusion
from cjlt.engine.selective_fusion import fit_affine_fusion, apply_fusion


def fit_guarded_thresholds(records, candidate, reference, reference_thresholds):
    targets = records["target"].astype(bool)
    weak = records["weak_mask"]
    reference_prediction = expit(reference) >= reference_thresholds
    probabilities = expit(candidate)
    grid = np.linspace(.05, .95, 181)
    thresholds = []
    fallback = []
    guard_rows = []
    for column, label in enumerate(LABELS):
        truth = targets[:, column]
        ref = reference_prediction[:, column]
        ref_fpr = float(ref[~truth].mean())
        ref_f1 = 2*np.sum(ref & truth)/max(np.sum(ref)+np.sum(truth),1)
        predictions = probabilities[:, column, None] >= grid[None, :]
        tp = (predictions & truth[:, None]).sum(0)
        f1 = 2*tp/np.maximum(predictions.sum(0)+truth.sum(),1)
        fpr = predictions[~truth].mean(0)
        allowed = fpr <= ref_fpr+.005+1e-12
        if column in [2,4,5] and weak[:,column].any():
            allowed &= predictions[weak[:,column]].mean(0) >= ref[weak[:,column]].mean()-.01-1e-12
        if column == 0:
            allowed &= f1 >= ref_f1-.002-1e-12
        if allowed.any():
            index = int(np.argmax(np.where(allowed,f1,-1)))
            threshold = float(grid[index])
        else:
            fallback.append(column)
            threshold = float(reference_thresholds[column])
        thresholds.append(threshold)
        guard_rows.append({"label": label, "fallback": column in fallback, "allowed_grid_points": int(allowed.sum())})
    return thresholds, fallback, guard_rows


def cached_scores(split, seed, group, role, model, path, indices):
    folder = ROOT / split / "fusion_score_cache" / f"{group}_{seed}"
    folder.mkdir(parents=True, exist_ok=True)
    output = folder / f"{role}.npz"
    checkpoint = ROOT / split / "readouts" / f"{group}_{seed}" / "model.pth"
    fingerprint = {"model_sha256": base.sha(checkpoint), "data_sha256": base.sha(path),
                   "indices_sha256": hashlib.sha256(np.asarray(indices,dtype="<i8").tobytes()).hexdigest()}
    if output.exists() and output.with_suffix(".json").exists():
        receipt = read(output.with_suffix(".json"))
        assert receipt["binding"] == fingerprint and receipt["sha256"] == base.sha(output)
        with np.load(output) as values:
            return values["logits"]
    inputs, labels = base.load(path, indices)
    logits = base.predict(model, inputs, "cuda").astype(float)
    np.savez(output, logits=logits, labels=labels)
    write(output.with_suffix(".json"), {"binding": fingerprint, "sha256": base.sha(output), "time": stamp()})
    return logits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=CONFIG["splits"], required=True)
    parser.add_argument("--tag", default="")
    parser.add_argument("--methods", nargs="+")
    args = parser.parse_args()
    setup()
    manifest = read(ROOT / args.split / "fusion_fit/manifest.json")
    fit_path, fit_indices = Path(manifest["path"]), manifest["indices"]
    cal_path, cal_indices = role_paths(args.split)["calibration"]
    fit_records = metadata(fit_path, fit_indices)
    cal_records = metadata(cal_path, cal_indices)
    available_methods = {"reference_affine": (None,"affine","reference_only"),
        "ordinary_mean": ("ordinary_expert","mean","two_model"),
        "combo_mean": ("combo_expert","mean","two_model"),
        "ordinary_affine": ("ordinary_expert","affine","two_model"),
        "combo_affine": ("combo_expert","affine","two_model"),
        "segment_affine": ("segment","affine","specialist_only"),
        "coverage_affine": ("segment_coverage","affine","specialist_only"),
        "bootstrap_mean": ("bootstrap_expert","mean","two_model"),
        "bootstrap_affine": ("bootstrap_expert","affine","two_model"),
        "weighted_mean": ("weighted_expert","mean","two_model"),
        "weighted_affine": ("weighted_expert","affine","two_model")}
    names = args.methods or list(available_methods)[:7]
    methods = {name: available_methods[name] for name in names}
    suffix = "_" + args.tag if args.tag else ""
    output_root = ROOT / args.split / ("fusions" + suffix)
    output_root.mkdir(exist_ok=True)
    write(output_root / "protocol.json", {"methods": methods, "fit_manifest_sha256": base.sha(ROOT / args.split / "fusion_fit/manifest.json"),
        "source_sha256": {str(p):base.sha(p) for p in (Path(__file__), HERE/"fusion_runtime.py", HERE/"05B_FUSION_CONTROL_GOAL.txt")},
        "fit_identities": len(fit_indices), "calibration_identities": len(cal_indices), "time": stamp()})
    for seed in CONFIG["seeds"]:
        reference_path = ROOT / args.split / "readouts" / f"baseline_{seed}"
        reference, _ = load_bundle(reference_path / "model.pth")
        reference_thresholds = read(reference_path / "calibration.json")["thresholds"]
        left_fit = cached_scores(args.split, seed, "baseline", "fit", reference, fit_path, fit_indices)
        left_cal = cached_scores(args.split, seed, "baseline", "calibration", reference, cal_path, cal_indices)
        for name,(specialist_group,kind,mode) in methods.items():
            check_deadline()
            specialist = None
            right_fit, right_cal = np.zeros_like(left_fit), np.zeros_like(left_cal)
            if specialist_group is not None:
                specialist_path = ROOT / args.split / "readouts" / f"{specialist_group}_{seed}" / "model.pth"
                specialist, _ = load_bundle(specialist_path)
                right_fit = cached_scores(args.split, seed, specialist_group, "fit", specialist, fit_path, fit_indices)
                right_cal = cached_scores(args.split, seed, specialist_group, "calibration", specialist, cal_path, cal_indices)
            a_fit,b_fit,a_cal,b_cal = left_fit,right_fit,left_cal,right_cal
            if mode == "specialist_only":
                a_fit,b_fit,a_cal,b_cal = right_fit,np.zeros_like(right_fit),right_cal,np.zeros_like(right_cal)
            if kind == "mean":
                state = {"kind":"affine","targets":LABELS,"weights":[.5,.5]*6,"biases":[0]*6}
            else:
                state = fit_affine_fusion(a_fit,b_fit,fit_records["target"],seed=seed,epochs=300,learning_rate=.04)
            candidate_cal,_ = apply_fusion(a_cal,b_cal,state)
            thresholds,fallback,guards = fit_guarded_thresholds(cal_records,candidate_cal,left_cal,reference_thresholds)
            state["fallback_columns"] = fallback
            candidate_cal[:,fallback] = left_cal[:,fallback]
            shared = specialist is not None and helper.model_hash(specialist.backbone) == helper.model_hash(reference.backbone)
            model = StaticFusion(reference,specialist,state,mode,shared).cuda().eval().requires_grad_(False)
            folder = output_root / f"{name}_{seed}"
            folder.mkdir(exist_ok=True)
            info = {"name":name,"seed":seed,"guards":guards,"fallback_columns":fallback,
                    "fit_manifest_sha256":base.sha(ROOT / args.split / "fusion_fit/manifest.json"),
                    "protocol_sha256":base.sha(output_root/"protocol.json"),"time":stamp()}
            model.save(folder/"model.pth",thresholds,info)
            loaded,_ = load_fusion(folder/"model.pth")
            inputs,_ = base.load(cal_path,cal_indices)
            actual = base.predict(loaded,inputs,"cuda")
            np.testing.assert_allclose(actual,candidate_cal,rtol=1e-5,atol=5e-5)
            np.testing.assert_array_equal(expit(actual)>=thresholds,expit(candidate_cal)>=thresholds)
            write(folder/"calibration.json",{"thresholds":thresholds,"original_thresholds":reference_thresholds,
                    "guards":guards,"max_logit_error":float(np.max(abs(actual-candidate_cal)))})
            write(folder/"receipt.json",{**info,"sha256":base.sha(folder/"model.pth"),
                    "parameters":sum(p.numel() for p in model.parameters()),"shared_backbone":shared})
            print("FUSION_FROZEN",args.split,seed,name,"fallback",fallback,flush=True)
            del model,loaded,specialist
        del reference
    write(output_root/"status.json",{"status":"completed","methods":list(methods),"time":stamp()})


if __name__ == "__main__":
    main()
