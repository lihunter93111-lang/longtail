"""Label, combination-frequency, cardinality, and power slices of saved predictions."""
from common import *
from sklearn.metrics import average_precision_score


def main():
    combinations = []
    slices = []
    label_rows = []
    counts = np.asarray(QUOTAS["A_counts"])
    for split in CONFIG["splits"]:
        metadata_cache = {}
        for result_root in (ROOT/split).glob("*results*"):
            role = "confirmation" if "confirmation" in result_root.name else "development"
            if role not in metadata_cache:
                manifest_path = ROOT/split/role/"manifest.json"
                if not manifest_path.exists():
                    continue
                manifest = read(manifest_path)
                metadata_cache[role] = metadata(manifest["path"],manifest["indices"])
            records = metadata_cache[role]
            truth = records["target"].astype(bool)
            codes = truth.astype(int)@(1<<np.arange(6))
            cardinality = truth.sum(1)
            train_frequency = counts[codes-1]
            maximum = np.where(truth,records["power"],-np.inf).max(1)
            minimum = np.where(truth,records["power"],np.inf).min(1)
            spread = maximum-minimum
            groups = {"all":np.ones(len(truth),dtype=bool),
                      "combo_count_le50":train_frequency<=50,
                      "combo_count_51_199":(train_frequency>50)&(train_frequency<200),
                      "combo_count_ge200":train_frequency>=200,
                      "legacy13_tail":np.isin(codes,QUOTAS["tail_codes"])}
            groups.update({f"cardinality_{k}":cardinality==k for k in range(1,7)})
            for lo,hi in ((0,5),(5,10),(10,15),(15,20.000001)):
                groups[f"power_spread_{lo}_{hi}"]=(spread>=lo)&(spread<hi)
                groups[f"lowfreq_power_spread_{lo}_{hi}"]=(spread>=lo)&(spread<hi)&(train_frequency<=50)
            for path in result_root.glob("*/predictions.npz"):
                with np.load(path) as data:
                    np.testing.assert_array_equal(data["identity"],records["identity"])
                    np.testing.assert_array_equal(data["target"],truth)
                    z = data["logits"].astype(float)
                    prediction = expit(z)>=data["thresholds"]
                exact = (prediction==truth).all(1)
                losses = (np.logaddexp(0,z)-truth*z).mean(1)
                info={"split":split,"role":role,"family":result_root.name,"model":path.parent.name}
                for code in range(1,64):
                    mask=codes==code
                    combinations.append({**info,"combination":code,"train_count":int(counts[code-1]),
                        "components":int(code.bit_count()),"n":int(mask.sum()),"exact":float(exact[mask].mean()),"BCE":float(losses[mask].mean())})
                for name,mask in groups.items():
                    if not mask.any():
                        continue
                    slices.append({**info,"slice":name,"n":int(mask.sum()),"exact":float(exact[mask].mean()),
                        "BCE":float(losses[mask].mean()),"empty_prediction_fraction":float((prediction[mask].sum(1)==0).mean())})
                for column,label in enumerate(LABELS):
                    for name in ("all","combo_count_le50","combo_count_51_199","combo_count_ge200"):
                        mask=groups[name]
                        positive=mask&truth[:,column]
                        negative=mask&~truth[:,column]
                        weak=mask&records["weak_mask"][:,column]
                        label_rows.append({**info,"slice":name,"label":label,"training_positive":int(QUOTAS["margins"][column]),
                            "positive_n":int(positive.sum()),"negative_n":int(negative.sum()),"weak_n":int(weak.sum()),
                            "recall":float(prediction[positive,column].mean()) if positive.any() else None,
                            "FPR":float(prediction[negative,column].mean()) if negative.any() else None,
                            "weak_recall":float(prediction[weak,column].mean()) if weak.any() else None,
                            "AP":float(average_precision_score(truth[mask,column],expit(z[mask,column]))) if positive.any() else None})
    output=ROOT/"analysis"
    output.mkdir(exist_ok=True)
    helper.table(output/"combination_metrics.csv",combinations)
    helper.table(output/"slice_metrics.csv",slices)
    helper.table(output/"label_frequency_metrics.csv",label_rows)
    write(output/"slice_contract.json",{"combination_frequency_bins":["<=50","51..199",">=200"],
        "power_spread":"max active nominal JNR minus min active nominal JNR", "legacy_tail_codes":QUOTAS["tail_codes"],
        "note":"descriptive slices; frequency, semantics and cardinality are associated", "time":stamp()})
    print("SLICES_COMPLETE",len(combinations),len(slices),len(label_rows),flush=True)


if __name__=="__main__":
    main()
