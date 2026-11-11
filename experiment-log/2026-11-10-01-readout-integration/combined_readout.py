"""Matched three-score fusion of a shared local readout and a second expert."""
import argparse
from common import *
from fit_fusions import cached_scores, fit_guarded_thresholds
from cjlt.engine.selective_fusion import _fit
from fusion_runtime import architecture, make_readout
from evaluate_stage import pair_comparison


class ThreeScoreAffine(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 1)
        with torch.no_grad():
            self.linear.weight.copy_(torch.tensor([[1., 0., 0.]]))
            self.linear.bias.zero_()

    def forward(self, inputs):
        reference = inputs[:, :1]
        return self.linear(inputs), torch.ones_like(reference), reference


class CombinedReadout(torch.nn.Module):
    def __init__(self, local, expert, coefficients, fallback):
        super().__init__()
        self.local = local
        self.expert = expert
        self.coefficients = coefficients
        self.fallback = fallback

    def forward(self, inputs):
        if self.local.stages:
            reference, features = self.local.feature_branches(inputs)
            local_logits = reference.clone()
            for stage, head, transform in zip(self.local.stages, self.local.heads, self.local.transforms):
                local_logits[:, stage["targets"]] += head(transform(features[self.local.variant]))
        else:
            reference = self.local.backbone(inputs)
            local_logits = reference
        expert = self.expert.backbone(inputs)
        scores = torch.stack((reference.double(), expert.double(), local_logits.double()-reference.double()), dim=-1)
        weights = torch.as_tensor(self.coefficients["weights"], device=scores.device, dtype=scores.dtype)
        biases = torch.as_tensor(self.coefficients["biases"], device=scores.device, dtype=scores.dtype)
        result = (scores*weights).sum(-1)+biases
        result[:, self.fallback] = reference[:, self.fallback].double()
        return result

    def save(self, path, thresholds, info):
        helper.atomic_torch_save({"format": "w32_combined_readout_v1", "local": architecture(self.local),
            "expert": architecture(self.expert), "coefficients": self.coefficients, "fallback": self.fallback,
            "model_state_dict": self.state_dict(), "thresholds": list(thresholds), "info": info}, path)


def load_combined(path):
    state = torch.load(path, map_location="cpu", weights_only=False)
    model = CombinedReadout(make_readout(state["local"]), make_readout(state["expert"]),
                            state["coefficients"], state["fallback"])
    model.load_state_dict(state["model_state_dict"])
    return model.cuda().eval().requires_grad_(False), state


def fit(split, expert_group="combo_expert", prefix="combo", output_name="combined"):
    manifest_path = ROOT/split/"fusion_fit/manifest.json"
    manifest = read(manifest_path)
    fit_path, fit_indices = Path(manifest["path"]), manifest["indices"]
    cal_path, cal_indices = role_paths(split)["calibration"]
    fit_records, cal_records = metadata(fit_path, fit_indices), metadata(cal_path, cal_indices)
    output_root = ROOT/split/output_name
    output_root.mkdir(exist_ok=True)
    groups = {f"{prefix}_residual_zero": "baseline", f"{prefix}_global_local": "global_stats_coverage",
              f"{prefix}_segment_local": "segment_coverage"}
    protocol = {"groups": groups, "epochs": 300, "lr": .04, "weight_decay": 1e-4,
                "fit_manifest_sha256": base.sha(manifest_path), "source_sha256": base.sha(Path(__file__)),
                "goal_sha256": base.sha(HERE/"05D_COMBINED_READOUT_GOAL.txt"),"expert_group":expert_group,
                "extension_goal_sha256":base.sha(HERE/("05G_WEIGHTED_LOCAL_GOAL.txt" if prefix=="weighted" else "05F_ORDINARY_LOCAL_CONTROL_GOAL.txt"))}
    if (output_root/"protocol.json").exists():
        assert read(output_root/"protocol.json")==protocol
    else:
        write(output_root/"protocol.json", protocol)
    cal_inputs, _ = base.load(cal_path, cal_indices)
    for seed in CONFIG["seeds"]:
        source = ROOT/split/"readouts"
        reference, _ = load_bundle(source/f"baseline_{seed}"/"model.pth")
        expert, _ = load_bundle(source/f"{expert_group}_{seed}"/"model.pth")
        thresholds_reference = read(source/f"baseline_{seed}"/"calibration.json")["thresholds"]
        scores = {}
        for role, path, indices in (("fit",fit_path,fit_indices),("calibration",cal_path,cal_indices)):
            scores[role] = [cached_scores(split,seed,"baseline",role,reference,path,indices),
                            cached_scores(split,seed,expert_group,role,expert,path,indices)]
        for name, group in groups.items():
            check_deadline()
            local, _ = load_bundle(source/f"{group}_{seed}"/"model.pth")
            arrays = {}
            for role, path, indices in (("fit",fit_path,fit_indices),("calibration",cal_path,cal_indices)):
                local_scores = cached_scores(split,seed,group,role,local,path,indices)
                arrays[role] = np.stack((*scores[role],local_scores-scores[role][0]),axis=-1)
            weights, biases = [], []
            for column in range(6):
                affine = _fit(ThreeScoreAffine(),arrays["fit"][:,column],fit_records["target"][:,column:column+1],
                              seed+column,300,.04)
                weights.append(affine.linear.weight.detach().numpy()[0].tolist())
                biases.append(float(affine.linear.bias.detach()[0]))
            coefficients = {"weights":weights,"biases":biases}
            predicted = (arrays["calibration"]*np.asarray(weights)).sum(-1)+biases
            thresholds,fallback,guards = fit_guarded_thresholds(cal_records,predicted,scores["calibration"][0],thresholds_reference)
            predicted[:,fallback] = scores["calibration"][0][:,fallback]
            model = CombinedReadout(local,expert,coefficients,fallback).cuda().eval().requires_grad_(False)
            folder = output_root/f"{name}_{seed}"
            folder.mkdir(exist_ok=True)
            info = {"name":name,"seed":seed,"protocol_sha256":base.sha(output_root/"protocol.json"),"time":stamp()}
            model.save(folder/"model.pth",thresholds,info)
            restored, _ = load_combined(folder/"model.pth")
            actual = base.predict(restored,cal_inputs,"cuda")
            np.testing.assert_allclose(actual,predicted,rtol=1e-5,atol=5e-5)
            np.testing.assert_array_equal(expit(actual)>=thresholds,expit(predicted)>=thresholds)
            write(folder/"calibration.json",{"thresholds":thresholds,"original_thresholds":thresholds_reference,"guards":guards})
            write(folder/"receipt.json",{**info,"sha256":base.sha(folder/"model.pth"),"fallback":fallback,
                  "parameters":sum(p.numel() for p in model.parameters()),"replay_max_error":float(abs(actual-predicted).max())})
            print("COMBINED_FROZEN",split,name,seed,flush=True)
            del local,model,restored
        del reference,expert
    write(output_root/"status.json",{"status":"completed","methods":list(groups),"time":stamp()})


def evaluate_combined(split, prefix="combo", output_name="combined"):
    from fusion_runtime import load_fusion
    methods = read(ROOT/split/output_name/"status.json")["methods"]
    reference_name = f"{prefix}_affine"
    reference_parent = "fusions_controls" if prefix=="weighted" else "fusions"
    manifest = read(ROOT/split/"development/manifest.json")
    inputs,_ = base.load(manifest["path"],manifest["indices"])
    records = metadata(manifest["path"],manifest["indices"])
    destination = ROOT/split/f"{output_name}_development_results"
    destination.mkdir(exist_ok=True)
    rows = []
    frozen = []
    for group in ["baseline",reference_name,*methods]:
        parent = "readouts" if group=="baseline" else reference_parent if group==reference_name else output_name
        loader = load_bundle if group=="baseline" else load_fusion if group==reference_name else load_combined
        for seed in CONFIG["seeds"]:
            check_deadline("evaluation_cutoff")
            folder = ROOT/split/parent/f"{group}_{seed}"
            calibration = read(folder/"calibration.json")
            frozen.append({"group":group,"seed":seed,"path":str(folder/"model.pth"),"sha256":base.sha(folder/"model.pth"),
                           "calibration_sha256":base.sha(folder/"calibration.json")})
            model,_ = loader(folder/"model.pth")
            logits = base.predict(model,inputs,"cuda").astype(float)
            output = destination/f"{group}_{seed}"
            output.mkdir(exist_ok=True)
            metrics = {"calibrated":evaluate(records,logits,calibration["thresholds"]),
                       "fixed_reference":evaluate(records,logits,calibration["original_thresholds"])}
            write(output/"metrics.json",metrics)
            np.savez(output/"predictions.npz",logits=logits,target=records["target"],identity=records["identity"],
                     thresholds=calibration["thresholds"],original_thresholds=calibration["original_thresholds"])
            rows.append({"group":group,"seed":seed,**{k:metrics["calibrated"][k] for k in
                         ("exact","raw_bce","macro_aupr","macro_f1","tail_exact","ECE")}})
            del model
    helper.table(destination/"metrics.csv",rows)
    pairs = [(name,"baseline") for name in methods]+[(name,reference_name) for name in methods]
    pairs += [(f"{prefix}_segment_local",f"{prefix}_global_local")]
    comparisons = [pair_comparison(a,b,destination,records) for a,b in pairs]
    means = {name:{k:float(np.mean([r[k] for r in rows if r["group"]==name])) for k in
                  ("exact","raw_bce","macro_aupr","macro_f1","tail_exact","ECE")} for name in ["baseline",reference_name,*methods]}
    write(destination/"summary.json",{"means":means,"comparisons":comparisons,"role":"development","models":frozen,"time":stamp()})
    print("COMBINED_EVALUATED",split,json.dumps(means),flush=True)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split",choices=CONFIG["splits"],required=True)
    parser.add_argument("--action",choices=["fit","evaluate"],required=True)
    parser.add_argument("--expert-group",choices=["combo_expert","ordinary_expert","weighted_expert"],default="combo_expert")
    options = parser.parse_args()
    setup()
    prefix = options.expert_group.removesuffix("_expert")
    output_name = "combined" if prefix=="combo" else f"combined_{prefix}"
    if options.action=="fit":
        fit(options.split,options.expert_group,prefix,output_name)
    else:
        evaluate_combined(options.split,prefix,output_name)
