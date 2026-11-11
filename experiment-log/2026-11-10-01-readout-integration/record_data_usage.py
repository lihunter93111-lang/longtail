"""Account for unique identities, selected rows, data roles, and consumed confirmation."""
from common import *


def main():
    rows,identity_sets,hashes = [],[],{}
    for split in CONFIG["splits"]:
        roles = role_paths(split)
        for role in ("fusion_fit","development","confirmation"):
            manifest = read(ROOT/split/role/"manifest.json")
            roles[role] = (Path(manifest["path"]),manifest["indices"])
        for role,(path,indices) in roles.items():
            key = str(path)
            if key not in hashes:
                hashes[key] = base.sha(path)
            with h5py.File(path,"r") as stream:
                noise = np.asarray(stream["receiver_noise_seed"]).reshape(-1).astype(np.int64)[indices]
                labels = np.asarray(stream["known_label"]).T[indices]
                pool_rows = int(stream["known_label"].shape[1])
            unique = set(noise.tolist())
            assert len(unique)==len(indices)
            assert not any(unique&previous for previous in identity_sets),(split,role,"identity overlap")
            identity_sets.append(unique)
            rows.append({"split":split,"role":role,"path":key,"sha256":hashes[key],"source_pool_rows":pool_rows,
                "selected_rows":len(indices),"unique_identities":len(unique),"label_positives":labels.sum(0).astype(int).tolist(),
                "confirmation_consumed":role=="confirmation"})
    totals = {role:sum(row["unique_identities"] for row in rows if row["role"]==role)
              for role in ("fit","check","calibration","fusion_fit","development","confirmation")}
    physical = {}
    for split in CONFIG["splits"]:
        path = ROOT/split/"physical_confirmation/summary.json"
        if path.exists():
            physical[split] = {"paired_view_identity_counts":read(path)["identity_counts"],"independent_identities_added":0,
                              "source_role":"confirmation"}
    write(ROOT/"data_usage_receipt.json",{"roles":rows,"totals":totals,
        "parameter_fitting_unique_identities":totals["fit"]+totals["fusion_fit"],
        "all_current_roles_unique_identities":sum(totals.values()),"all_roles_pairwise_identity_disjoint":True,
        "five_model_seeds_reuse_each_split_fit_identities":True,"physical_views":physical,
        "final_freeze_sha256":base.sha(ROOT/"final_freeze.json"),"confirmation_status":"consumed after model and calibration freeze",
        "head_training":"cached-feature presentations, physical views preserve source identity",
        "affine_fitting":"independent fusion_fit, 300 full-batch steps per label, reused across matched candidates",
        "time":stamp()})
    print("DATA_USAGE_ACCOUNTED",json.dumps(totals),sum(totals.values()),flush=True)


if __name__=="__main__":
    main()
