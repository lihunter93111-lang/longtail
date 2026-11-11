"""Recompute physical diagnostics from saved positive/negative model outputs."""
import csv
from common import *
from sklearn.metrics import roc_auc_score, average_precision_score


def main():
    count,max_error = 0,0.
    for split in CONFIG["splits"]:
        folder = ROOT/split/"physical_confirmation"
        if not (folder/"summary.json").exists():
            continue
        protocol = read(folder/"protocol.json")
        assert protocol["freeze_sha256"]==base.sha(ROOT/"final_freeze.json")
        for path,digest in protocol["views"].items():
            assert base.sha(path)==digest
        with (folder/"metrics.csv").open(encoding="utf-8-sig",newline="") as stream:
            metrics = list(csv.DictReader(stream))
        for row in metrics:
            name = f'{row["model"]}_{row["seed"]}'
            with np.load(folder/name/"absent.npz") as absent, np.load(folder/name/f'gap{row["gap"]}.npz') as positive:
                lookup = np.searchsorted(absent["positions"],positive["positions"])
                np.testing.assert_array_equal(absent["positions"][lookup],positive["positions"])
                targets = positive["targets"]
                np.testing.assert_array_equal(absent["targets"][lookup],targets)
                mask = np.ones(len(targets),dtype=bool) if row["label"]=="all_targets" else targets==LABELS.index(row["label"])
                pos = positive["logits"][np.arange(len(targets)),targets][mask]
                neg = absent["logits"][lookup][np.arange(len(targets)),targets][mask]
                thresholds = positive["thresholds"][targets][mask]
                truth = np.r_[np.ones(mask.sum()),np.zeros(mask.sum())]
                scores = np.r_[pos,neg]
                values = {"TPR":float((expit(pos)>=thresholds).mean()),"matched_absent_FPR":float((expit(neg)>=thresholds).mean()),
                    "paired_ranking":float((pos>neg).mean()),"mean_paired_logit_difference":float((pos-neg).mean()),
                    "positive_BCE":float(np.logaddexp(0,-pos).mean()),"absent_BCE":float(np.logaddexp(0,neg).mean()),
                    "fixed05_TPR":float((pos>=0).mean()),"fixed05_absent_FPR":float((neg>=0).mean()),
                    "AUC":float(roc_auc_score(truth,scores)),"AP":float(average_precision_score(truth,scores))}
                assert int(row["identities"])==int(mask.sum())
                for metric,value in values.items():
                    error = abs(value-float(row[metric]))
                    assert error<1e-12,(split,name,metric,error)
                    max_error = max(max_error,error)
                    count += 1
    result = {"passed":count>0,"recomputed_metrics":count,"maximum_error":max_error,"time":stamp()}
    write(ROOT/"physical_verification.json",result)
    print(json.dumps(result),flush=True)


if __name__=="__main__":
    main()
