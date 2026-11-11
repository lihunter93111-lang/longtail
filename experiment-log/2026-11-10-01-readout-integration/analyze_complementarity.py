"""Describe corrected errors, new errors, and expert overlap without model fitting."""
from common import *


def predictions(path):
    with np.load(path) as data:
        result = {key:data[key] for key in ("logits","target","identity","thresholds")}
    result["prediction"] = expit(result["logits"])>=result["thresholds"]
    result["correct"] = (result["prediction"]==result["target"]).all(1)
    return result


def main():
    changes, overlaps = [], []
    for split in CONFIG["splits"]:
        for summary_path in (ROOT/split).glob("*results*/summary.json"):
            summary = read(summary_path)
            result_root = summary_path.parent
            pairs = {(p["candidate"],p["reference"]) for p in summary["comparisons"] if p["variant"]=="calibrated"}
            for candidate,reference in sorted(pairs):
                for seed in CONFIG["seeds"]:
                    left = predictions(result_root/f"{candidate}_{seed}"/"predictions.npz")
                    right = predictions(result_root/f"{reference}_{seed}"/"predictions.npz")
                    np.testing.assert_array_equal(left["identity"],right["identity"])
                    np.testing.assert_array_equal(left["target"],right["target"])
                    good,bad = left["correct"]&~right["correct"],~left["correct"]&right["correct"]
                    changes.append({"split":split,"role":summary["role"],"family":result_root.name,
                        "candidate":candidate,"reference":reference,"seed":seed,"n":len(good),
                        "corrected":int(good.sum()),"new_errors":int(bad.sum()),
                        "net_gain":float(good.mean()-bad.mean()),
                        "net_per_correction":float((good.sum()-bad.sum())/max(good.sum(),1)),
                        "both_wrong":int((~left["correct"]&~right["correct"]).sum()),
                        "changed_label_decisions":int((left["prediction"]!=right["prediction"]).sum())})
        expert_root = ROOT/split/"development_results_sampling_controls"
        if not (expert_root/"summary.json").exists():
            expert_root = ROOT/split/"development_results_experts"
        if not (expert_root/"summary.json").exists():
            continue
        groups = read(expert_root/"summary.json")["means"]
        for seed in CONFIG["seeds"]:
            reference = predictions(expert_root/f"baseline_{seed}"/"predictions.npz")
            for name in groups:
                if name=="baseline":
                    continue
                expert = predictions(expert_root/f"{name}_{seed}"/"predictions.npz")
                np.testing.assert_array_equal(reference["identity"],expert["identity"])
                ref_error,exp_error = ~reference["correct"],~expert["correct"]
                overlap = ref_error&exp_error
                overlaps.append({"split":split,"seed":seed,"expert":name,"role":"development",
                    "reference_exact":float(reference["correct"].mean()),"expert_exact":float(expert["correct"].mean()),
                    "oracle_union_exact":float((reference["correct"]|expert["correct"]).mean()),
                    "oracle_recoverable_reference_errors":int((ref_error&~exp_error).sum()),
                    "shared_errors":int(overlap.sum()),"error_jaccard":float(overlap.sum()/(ref_error|exp_error).sum()),
                    "note":"oracle bounds a true-label selector choosing one whole-model prediction per identity; label-wise fusion can construct additional prediction vectors"})
    output = ROOT/"analysis"
    output.mkdir(exist_ok=True)
    helper.table(output/"corrected_and_new_errors.csv",changes)
    helper.table(output/"expert_error_overlap.csv",overlaps)
    write(output/"complementarity_receipt.json",{"comparison_rows":len(changes),"overlap_rows":len(overlaps),"time":stamp()})
    print("COMPLEMENTARITY_COMPLETE",len(changes),len(overlaps),flush=True)


if __name__=="__main__":
    main()
