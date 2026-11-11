"""Paired interaction of expert sampling and segment residual readout."""
from common import *
from evaluate_stage import interval, identity_interval


def compare(split,role,prefix="combo"):
    candidate_affine,candidate_local = f"{prefix}_affine",f"{prefix}_segment_local"
    candidate_family = "combined_development_results" if prefix=="combo" else f"combined_{prefix}_development_results"
    names = {"ordinary_affine":"combined_ordinary_development_results",
             "ordinary_segment_local":"combined_ordinary_development_results",
             candidate_affine:candidate_family,candidate_local:candidate_family}
    if role=="confirmation":
        names = {name:"final_confirmation_results" for name in names}
    if not all((ROOT/split/family/"summary.json").exists() for family in names.values()):
        return []
    if not all((ROOT/split/family/f"{name}_{CONFIG['seeds'][0]}").exists() for name,family in names.items()):
        return []
    results = []
    exact_rows,bce_rows = [],[]
    for seed in CONFIG["seeds"]:
        exact,bce,identities,labels = {},{},{},{}
        for name,family in names.items():
            with np.load(ROOT/split/family/f"{name}_{seed}"/"predictions.npz") as data:
                z,y = data["logits"].astype(float),data["target"].astype(bool)
                exact[name] = ((expit(z)>=data["thresholds"])==y).all(1).astype(float)
                bce[name] = (np.logaddexp(0,z)-y*z).mean(1)
                identities[name],labels[name] = data["identity"],y
        for name in names:
            np.testing.assert_array_equal(identities[name],identities["ordinary_affine"])
            np.testing.assert_array_equal(labels[name],labels["ordinary_affine"])
        exact_rows.append(exact[candidate_local]-exact[candidate_affine]-exact["ordinary_segment_local"]+exact["ordinary_affine"])
        bce_rows.append(bce[candidate_affine]-bce[candidate_local]-bce["ordinary_affine"]+bce["ordinary_segment_local"])
    for metric,values in (("exact",exact_rows),("BCE_reduction",bce_rows)):
        per_seed = np.asarray(values).mean(1)
        results.append({"split":split,"role":role,"expert_family":prefix,"metric":metric,"interaction":float(per_seed.mean()),"per_seed":per_seed.tolist(),
            "seed_t95":interval(per_seed),"identity_bootstrap95":identity_interval(values,labels["ordinary_affine"])})
    return results


def main():
    results = []
    for split in CONFIG["splits"]:
        for role in ("development","confirmation"):
            for prefix in ("combo","weighted"):
                results.extend(compare(split,role,prefix))
    write(ROOT/"analysis/factorial_interaction.json",{"results":results,"time":stamp(),
          "definition":"local gain in the specified expert ensemble minus local gain in ordinary ensemble; both stages include their fixed calibration policy"})
    print(json.dumps(results),flush=True)


if __name__=="__main__":
    main()
