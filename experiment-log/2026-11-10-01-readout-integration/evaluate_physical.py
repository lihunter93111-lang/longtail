"""Evaluate frozen candidates on identity-paired target-present/removed views."""
import argparse
from common import *
from final_campaign import LOADERS
from evaluate_stage import interval
from sklearn.metrics import roc_auc_score, average_precision_score

METRICS = ["TPR","matched_absent_FPR","paired_ranking","mean_paired_logit_difference",
           "positive_BCE","absent_BCE","fixed05_TPR","fixed05_absent_FPR"]


def sample_values(positive,negative,thresholds,targets):
    rows = np.arange(len(targets))
    pos,neg = positive[rows,targets],negative[rows,targets]
    threshold = np.asarray(thresholds)[targets]
    return np.column_stack((expit(pos)>=threshold,expit(neg)>=threshold,pos>neg,pos-neg,
                            np.logaddexp(0,-pos),np.logaddexp(0,neg),pos>=0,neg>=0))


def bootstrap(values, targets):
    rng = np.random.default_rng(2026111006)
    groups = [np.flatnonzero(targets==label) for label in np.unique(targets)]
    means = []
    for _ in range(40):
        indices = np.concatenate([rng.choice(group,size=(50,len(group)),replace=True) for group in groups],axis=1)
        means.append(values[indices].mean(1))
    return np.quantile(np.concatenate(means),[.025,.975],axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split",choices=CONFIG["splits"],required=True)
    args = parser.parse_args()
    setup()
    frozen = read(ROOT/"final_freeze.json")
    for path,digest in frozen["sources"].items():
        assert base.sha(path)==digest,(path,"source changed after final freeze")
    methods = frozen["selection"]["physical_methods"]
    data_root = ROOT/args.split/"confirmation_pairs"
    assert read(data_root/"manifest.json")["freeze_sha256"]==base.sha(ROOT/"final_freeze.json")
    assert read(data_root/"diagnostic_pairing_receipt.json")["single_target_delta_verified"]
    assert read(data_root/"absent_pairing_receipt.json")["receiver_noise_and_other_components_verified"]
    output = ROOT/args.split/"physical_confirmation"
    output.mkdir(exist_ok=True)
    assert not (output/"summary.json").exists()
    paths = [data_root/"diagnostic_absent.mat",*[data_root/f"diagnostic_gap{gap}.mat" for gap in (0,5,10,15)]]
    write(output/"protocol.json",{"role":"frozen_confirmation_physical_diagnostic","methods":methods,
          "freeze_sha256":base.sha(ROOT/"final_freeze.json"),"views":{str(p):base.sha(p) for p in paths},
          "source_sha256":base.sha(Path(__file__)),"time":stamp()})
    write(data_root/"evaluation_access.json",{"time":stamp(),"protocol_sha256":base.sha(output/"protocol.json")})
    absent_x,absent_positions,absent_targets = helper.load_views(paths[0])
    with h5py.File(paths[0],"r") as stream:
        absent_y = np.asarray(stream["known_label"]).T.astype(bool)
    assert not absent_y[np.arange(len(absent_targets)),absent_targets].any()
    assert (absent_y.sum(1)>=1).all()
    views = {gap:helper.load_views(data_root/f"diagnostic_gap{gap}.mat") for gap in (0,5,10,15)}
    sample_cache, rows = {}, []
    for name in methods:
        sample_cache[name] = {}
        for seed in CONFIG["seeds"]:
            check_deadline("evaluation_cutoff")
            item = frozen["models"][f"{args.split}/{name}_{seed}"]
            assert base.sha(item["path"])==item["sha256"]
            assert base.sha(item["calibration_path"])==item["calibration_sha256"]
            model,_ = LOADERS[item["loader"]](item["path"])
            thresholds = item["thresholds"]
            negative = base.predict(model,absent_x,"cuda").astype(float)
            folder = output/f"{name}_{seed}"
            folder.mkdir(exist_ok=True)
            np.savez(folder/"absent.npz",logits=negative,positions=absent_positions,targets=absent_targets,labels=absent_y)
            for gap,(inputs,positions,targets) in views.items():
                lookup = np.searchsorted(absent_positions,positions)
                np.testing.assert_array_equal(absent_positions[lookup],positions)
                np.testing.assert_array_equal(absent_targets[lookup],targets)
                positive = base.predict(model,inputs,"cuda").astype(float)
                values = sample_values(positive,negative[lookup],thresholds,targets)
                sample_cache[name][(seed,gap)] = values
                np.savez(folder/f"gap{gap}.npz",logits=positive,positions=positions,targets=targets,
                         thresholds=thresholds,diagnostic_values=values)
                for label in (None,2,4,5):
                    mask = np.ones(len(targets),dtype=bool) if label is None else targets==label
                    assert mask.any()
                    pos = positive[np.arange(len(targets)),targets][mask]
                    neg = negative[lookup][np.arange(len(targets)),targets][mask]
                    truth,scores = np.r_[np.ones(mask.sum()),np.zeros(mask.sum())],np.r_[pos,neg]
                    rows.append({"model":name,"seed":seed,"gap":gap,"label":"all_targets" if label is None else LABELS[label],
                        "identities":int(mask.sum()),**dict(zip(METRICS,values[mask].mean(0).tolist())),
                        "AUC":float(roc_auc_score(truth,scores)),"AP":float(average_precision_score(truth,scores))})
            del model
            print("PHYSICAL_EVALUATED",args.split,name,seed,flush=True)
    comparisons = []
    for candidate,reference in frozen["selection"]["physical_comparisons"]:
        for gap in (0,5,10,15):
            targets = views[gap][2]
            for label in (None,2,4,5):
                mask = np.ones(len(targets),dtype=bool) if label is None else targets==label
                left = np.stack([sample_cache[candidate][(seed,gap)][mask] for seed in CONFIG["seeds"]])
                right = np.stack([sample_cache[reference][(seed,gap)][mask] for seed in CONFIG["seeds"]])
                differences = left-right
                bounds = bootstrap(differences.mean(0),targets[mask])
                for column,metric in enumerate(METRICS):
                    deltas = differences.mean(1)[:,column]
                    comparisons.append({"candidate":candidate,"reference":reference,"gap":gap,
                        "label":"all_targets" if label is None else LABELS[label],"metric":metric,"identities":int(mask.sum()),
                        "candidate_mean":float(left[:,:,column].mean()),"reference_mean":float(right[:,:,column].mean()),
                        "difference":float(deltas.mean()),"seed_t95":interval(deltas),"identity_bootstrap95":bounds[:,column].tolist()})
    helper.table(output/"metrics.csv",rows)
    helper.table(output/"comparisons.csv",comparisons)
    write(output/"summary.json",{"role":"frozen_confirmation_physical_diagnostic","comparisons":comparisons,
          "models":len(methods)*5,"identity_counts":{str(gap):len(views[gap][1]) for gap in views},
          "bootstrap":"2000 target-label-stratified identity resamples, paired views and five seeds kept together",
          "normalization":"ordinary per-input RMS after physical pairing verification","completed_utc":stamp()})
    print("PHYSICAL_COMPLETE",args.split,len(rows),flush=True)


if __name__=="__main__":
    main()
