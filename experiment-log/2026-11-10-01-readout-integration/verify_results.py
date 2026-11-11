"""Independent metric, immutable-source, and completed-training checks."""
from common import *


def main():
    errors = []
    metric_count = 0
    max_error = 0.0
    for manifest in (ROOT / "source_lock.json",):
        for filename,digest in read(manifest)["files"].items():
            if not Path(filename).exists() or base.sha(filename) != digest:
                errors.append({"path":filename,"error":"source hash mismatch"})
    for split in CONFIG["splits"]:
        for folder in (ROOT / split).glob("*results*"):
            for metric_path in folder.glob("*/metrics.json"):
                prediction_path = metric_path.parent / "predictions.npz"
                if not prediction_path.exists():
                    continue
                metrics = read(metric_path)
                with np.load(prediction_path) as data:
                    z,y = data["logits"].astype(float),data["target"].astype(bool)
                    assert len(np.unique(data["identity"])) == len(y)
                    bce = float(np.mean(np.maximum(z,0)-y*z+np.log1p(np.exp(-np.abs(z)))))
                    for name,key in (("calibrated","thresholds"),("fixed_reference","original_thresholds")):
                        pred = expit(z) >= data[key]
                        exact = float(np.mean(np.all(pred == y,axis=1)))
                        for metric,value in (("raw_bce",bce),("exact",exact)):
                            error = abs(value-metrics[name][metric])
                            max_error = max(max_error,error)
                            metric_count += 1
                            if error > 1e-12:
                                errors.append({"path":str(metric_path),"metric":metric,"error":error})
    expert_count = 0
    expert_updates = 0
    matched_initial_states = 0
    matched_weighted_orders = 0
    for receipt_path in ROOT.glob("split*/expert_training/*/M_*/receipt.json"):
        receipt = read(receipt_path)
        history = read(receipt_path.parent / "history.json")
        assert len(history)==40 and receipt["updates"]==10000
        assert base.sha(receipt_path.parent/"best_BCE.pth")==receipt["checkpoint_sha256"]
        assert min(row["early_BCE"] for row in history)==receipt["best_early_BCE"]
        group = receipt_path.parent.parent.name
        split = receipt_path.parents[3].name
        if group != "ordinary_expert":
            reference = read(ROOT/split/"expert_training/ordinary_expert"/receipt_path.parent.name/"receipt.json")
            assert reference["initial_state_sha256"]==receipt["initial_state_sha256"]
            matched_initial_states += 1
            if group == "weighted_expert":
                assert reference["order_sha256"]==receipt["order_sha256"]
                matched_weighted_orders += 1
        counts_path=receipt_path.parent/"sample_counts.npz"
        if counts_path.exists():
            with np.load(counts_path) as sample:
                counts=sample["counts"]
                assert counts.shape==(40,8000) and (counts.sum(1)==8000).all()
                for row,visits in zip(history,counts):
                    assert row["unique_sampled_identities"]==np.count_nonzero(visits)
        expert_count += 1
        expert_updates += receipt["updates"]
    head_count = 0
    head_views = {"original":0,"gap10":0,"gap15":0}
    for path in ROOT.glob("split*/readouts/*/stage_*.json"):
        record = read(path)
        assert len(record["history"])==24 and record["updates"]==6000
        assert 0<=record["best_epoch"]<=24
        weak_visits = sum(row.get("weak_visits",0) for row in record["history"])
        gap15_visits = sum(row.get("gap15_visits",0) for row in record["history"])
        assert 0<=gap15_visits<=weak_visits<=192000
        head_views["original"] += 192000-weak_visits
        head_views["gap10"] += weak_visits-gap15_visits
        head_views["gap15"] += gap15_visits
        head_count += 1
    zero_residual_checks = 0
    for split in CONFIG["splits"]:
        for prefix in ("combo","ordinary","weighted"):
            family = "combined_development_results" if prefix=="combo" else f"combined_{prefix}_development_results"
            directory = ROOT/split/family
            if not (directory/"summary.json").exists():
                continue
            for seed in CONFIG["seeds"]:
                with np.load(directory/f"{prefix}_residual_zero_{seed}/predictions.npz") as zero, np.load(directory/f"{prefix}_affine_{seed}/predictions.npz") as reference:
                    np.testing.assert_array_equal(zero["identity"],reference["identity"])
                    np.testing.assert_allclose(zero["logits"],reference["logits"],rtol=0,atol=1e-12)
                    np.testing.assert_array_equal(zero["thresholds"],reference["thresholds"])
                    zero_residual_checks += 1
    result={"passed":not errors,"source_files_checked":len(read(ROOT/"source_lock.json")["files"]),
            "scalar_metrics_recomputed":metric_count,"maximum_metric_error":max_error,
            "complete_heads":head_count,"complete_experts":expert_count,"expert_updates":expert_updates,
            "head_updates":head_count*6000,"head_identity_visits":head_count*192000,
            "expert_identity_visits":expert_count*320000,"errors":errors,"time":stamp()}
    result.update(matched_expert_initial_states=matched_initial_states,
                  matched_weighted_natural_orders=matched_weighted_orders,
                  exact_zero_residual_replays=zero_residual_checks)
    assert sum(head_views.values())==result["head_identity_visits"]
    result["head_cached_feature_presentations_by_view"] = head_views
    write(ROOT/"verification.json",result)
    print(json.dumps(result),flush=True)
    assert not errors


if __name__=="__main__":
    main()
