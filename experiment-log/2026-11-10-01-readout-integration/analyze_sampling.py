"""Audit realized and objective training exposure for all four expert controls."""
from common import *


def main():
    rows, designs = [], []
    for split in CONFIG["splits"]:
        path,indices = role_paths(split)["fit"]
        with h5py.File(path,"r") as stream:
            labels = np.asarray(stream["known_label"]).T[indices].astype(int)
        codes = labels@(1<<np.arange(6))
        cardinality = labels.sum(1)
        combo_counts = np.bincount(codes,minlength=64)
        np.testing.assert_array_equal(combo_counts[1:],QUOTAS["A_counts"])
        for folder in (ROOT/split/"expert_training").iterdir():
            group = folder.name
            probabilities = np.full(len(labels),1/len(labels))
            design_path = folder/"sampler.json"
            if design_path.exists():
                probabilities = np.asarray(read(design_path)["probabilities"])
            design_path = folder/"exposure_design.json"
            if design_path.exists():
                probabilities = np.asarray(read(design_path)["objective_probabilities"])
            effective_combos = np.bincount(codes,weights=probabilities,minlength=64)[1:]
            label_fraction = probabilities@labels
            designs.append({"split":split,"group":group,"unique_fit_identities":len(labels),
                "training_label_margins":labels.sum(0).tolist(),"objective_label_fraction":label_fraction.tolist(),
                "objective_label_IF":float(label_fraction.max()/label_fraction.min()),
                "objective_combination_IF":float(effective_combos.max()/effective_combos.min()),
                "objective_Kish_effective_sample_size":float(1/np.square(probabilities).sum()),
                "objective_cardinality_fraction":np.bincount(cardinality,weights=probabilities,minlength=7)[1:].tolist()})
            for receipt_path in folder.glob("M_*/receipt.json"):
                receipt = read(receipt_path)
                history = read(receipt_path.parent/"history.json")
                weak_visits = sum(row["augmented_presentations"] for row in history)
                gap15_visits = sum(row["gap15_presentations"] for row in history)
                assert 0<=gap15_visits<=weak_visits<=320000
                count_path = receipt_path.parent/"sample_counts.npz"
                counts = np.ones((40,len(labels)),dtype=int)
                if count_path.exists():
                    with np.load(count_path) as data:
                        counts = data["counts"]
                assert counts.shape==(40,8000) and (counts.sum(1)==8000).all()
                unique = np.count_nonzero(counts,axis=1)
                visits = counts.sum(0)
                realized_labels = visits@labels/visits.sum()
                rows.append({"split":split,"group":group,"seed":int(receipt_path.parent.name[2:]),
                    "updates":receipt["updates"],"identity_visits":int(visits.sum()),
                    "original_view_presentations":320000-weak_visits,
                    "gap10_view_presentations":weak_visits-gap15_visits,
                    "gap15_view_presentations":gap15_visits,
                    "union_unique_identities":int(np.count_nonzero(visits)),
                    "mean_unique_per_epoch":float(unique.mean()),"mean_repetition_fraction":float(1-unique.mean()/8000),
                    "visit_Kish_effective_sample_size":float(visits.sum()**2/np.square(visits.astype(float)).sum()),
                    **{f"{label}_presentation_fraction":float(realized_labels[j]) for j,label in enumerate(LABELS)}})
    output = ROOT/"analysis"
    output.mkdir(exist_ok=True)
    helper.table(output/"sampling_exposure.csv",rows)
    write(output/"sampling_designs.json",{"designs":designs,"time":stamp(),
        "note":"weighted objective probabilities describe loss weights; physical visits remain uniform without replacement"})
    print("SAMPLING_ANALYSIS_COMPLETE",len(rows),flush=True)


if __name__=="__main__":
    main()
