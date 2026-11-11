"""PCA retention versus head-width controls using the existing M-view trainer."""
import argparse
from common import *
from run_coverage_readouts import train_stage, weak_features


def save_adaptive(model, path, thresholds, info, dimensions, hidden):
    helper.atomic_torch_save({"format":"w32_readout_adaptive_v1","variant":model.variant,
        "stages":model.stages,"width":model.width,"dimensions":dimensions,"hidden":hidden,
        "model_state_dict":model.state_dict(),"thresholds":list(thresholds),"info":info},Path(path))


def load_adaptive(path):
    state=torch.load(path,map_location="cpu",weights_only=False)
    if state["format"] != "w32_readout_adaptive_v1":
        return load_bundle(path)
    model=ReadoutBundle(state["variant"],width=state["width"])
    for stage in state["stages"]:
        model.transforms.append(FeatureTransform(stage["inputs"],state["dimensions"]))
        model.heads.append(ResidualReadout(state["dimensions"],state["hidden"],len(stage["targets"])))
        model.stages.append(stage)
    model.load_state_dict(state["model_state_dict"])
    return model.cuda().eval().requires_grad_(False),state


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--split",choices=CONFIG["splits"],required=True)
    args=parser.parse_args()
    setup()
    designs=[("pca256",256,32),("head64",128,64)]
    protocol={"designs":designs,"feature_sources":["global_stats","segment"],"views":"M",
              "epochs":24,"selection":"same guarded exact rule with epoch0 fallback",
              "sources":{str(p):base.sha(p) for p in [Path(__file__),HERE/"run_coverage_readouts.py",HERE/"04A_PCA_CAPACITY_GOAL.txt"]}}
    write(ROOT/args.split/"adaptive_protocol.json",protocol)
    for seed in CONFIG["seeds"]:
        original,calibration,_=load_baseline(args.split,seed)
        thresholds=calibration["thresholds"]["raw"]
        records={}
        for role in ("fit","check","calibration"):
            with np.load(ROOT/args.split/"cache"/f"M_{seed}"/f"{role}.npz") as data:
                records[role]=dict(data)
        weak10=weak_features(args.split,seed,10,original,records["fit"])
        weak15=weak_features(args.split,seed,15,original,records["fit"])
        for design,dimensions,hidden in designs:
            # Process-local settings supply the explicitly declared branch dimensions to the shared trainer.
            CONFIG["pca_components"]=dimensions
            CONFIG["hidden"]=hidden
            for variant in ("global_stats","segment"):
                check_deadline()
                output=ROOT/args.split/"readouts"/f"{variant}_{design}_{seed}"
                output.mkdir(exist_ok=True)
                if (output/"receipt.json").exists():
                    assert read(output/"receipt.json")["protocol_sha256"]==base.sha(ROOT/args.split/"adaptive_protocol.json")
                    continue
                model=ReadoutBundle(variant).cuda().eval().requires_grad_(False)
                model.backbone.load_state_dict(original.backbone.state_dict())
                backbone_hash=helper.model_hash(model.backbone)
                stages=[train_stage(model,records["fit"],records["check"],weak10,weak15,thresholds,targets,seed,output)
                        for targets in CONFIG["stages"]]
                assert helper.model_hash(model.backbone)==backbone_hash
                z=cached_logits(model,records["calibration"])
                new_thresholds=helper.fit_thresholds(z,records["calibration"]["target"])
                info={"dimensions":dimensions,"hidden":hidden,"coverage":"M","stages":stages,
                      "head_parameters":sum(p.numel() for head in model.heads for p in head.parameters()),
                      "protocol_sha256":base.sha(ROOT/args.split/"adaptive_protocol.json"),"time":stamp()}
                save_adaptive(model,output/"model.pth",new_thresholds,info,dimensions,hidden)
                write(output/"calibration.json",{"thresholds":new_thresholds,"original_thresholds":thresholds})
                restored,_=load_adaptive(output/"model.pth")
                path,indices=role_paths(args.split)["calibration"]
                x,_=base.load(path,indices[:126])
                actual=base.predict(restored,x,"cuda")
                np.testing.assert_allclose(actual,z[:126],rtol=1e-5,atol=3e-5)
                np.testing.assert_array_equal(expit(actual)>=new_thresholds,expit(z[:126])>=new_thresholds)
                write(output/"receipt.json",{**info,"checkpoint_sha256":base.sha(output/"model.pth")})
                print("ADAPTIVE_READOUT_COMPLETE",args.split,seed,variant,design,flush=True)
                del model,restored
        del original,records,weak10,weak15
        torch.cuda.empty_cache()
    write(ROOT/args.split/"adaptive_status.json",{"status":"completed","time":stamp()})


if __name__=="__main__":
    main()
