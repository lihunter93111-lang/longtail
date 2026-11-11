"""Copy completed lightweight evidence and create the repository delivery manifest."""
import csv
import shutil
from common import *


def copy(source, destination):
    assert source.is_file(),str(source)
    destination.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(source,destination)


def main():
    assert read(ROOT/"verification.json")["passed"]
    frozen = read(ROOT/"final_freeze.json")
    assert base.sha(frozen["selection_path"])==frozen["selection_sha256"]
    for path,digest in frozen["sources"].items():
        assert base.sha(path)==digest,(path,"final scientific source hash mismatch")
    verified = []
    for key,item in frozen["models"].items():
        assert base.sha(item["path"])==item["sha256"]
        assert base.sha(item["calibration_path"])==item["calibration_sha256"]
        verified.append(key)
    output = HERE/"results"
    output.mkdir(exist_ok=True)
    for name in ("source_lock.json","baseline_registry.json","baseline_data_verification.json","smoke.json",
                 "verification.json","final_freeze.json","fusion_runtime_smoke.json","weighting_selfcheck.json","environment_receipt.json",
                 "weak_cache_verification.json","data_usage_receipt.json","final_pipeline_status.json"):
        copy(ROOT/name,output/name)
    if (ROOT/"physical_verification.json").exists():
        assert read(ROOT/"physical_verification.json")["passed"]
        copy(ROOT/"physical_verification.json",output/"physical_verification.json")
    for split in CONFIG["splits"]:
        assert read(ROOT/split/"final_confirmation_results/summary.json")["role"]=="confirmation"
        for result_root in (ROOT/split).glob("*results*"):
            if not (result_root/"summary.json").exists():
                continue
            for name in ("summary.json","metrics.csv","freeze.json","evaluation_access.json"):
                if (result_root/name).is_file():
                    copy(result_root/name,output/split/result_root.name/name)
        for role in ("development","fusion_fit","confirmation"):
            for name in ("manifest.json","preflight.json"):
                copy(ROOT/split/role/name,output/split/role/name)
        for source in (ROOT/split).glob("*.json"):
            copy(source,output/split/source.name)
        for branch in ("fusions","fusions_controls","fusions_adaptive","combined","combined_ordinary",
                       "combined_weighted","combined_ablations","combined_weighted_ablations"):
            for name in ("protocol.json","status.json"):
                source = ROOT/split/branch/name
                if source.is_file():
                    copy(source,output/split/branch/name)
        for receipt in (ROOT/split/"expert_training").glob("*/M_*/receipt.json"):
            copy(receipt,output/split/"expert_training"/receipt.parent.parent.name/receipt.parent.name/receipt.name)
        for design in (ROOT/split/"expert_training").glob("*/*.json"):
            copy(design,output/split/"expert_training"/design.parent.name/design.name)
        for item in frozen["models"].values():
            if item["split"]!=split:
                continue
            folder = Path(item["path"]).parent
            for name in ("calibration.json","receipt.json"):
                source = folder/name
                if name=="receipt.json" and item["name"]=="baseline":
                    source = BASELINES/split/"w32"/f'M_{item["seed"]}'/name
                copy(source,output/split/"frozen_models"/folder.name/name)
        if (ROOT/split/"physical_confirmation/summary.json").exists():
            for name in ("summary.json","protocol.json","metrics.csv","comparisons.csv"):
                copy(ROOT/split/"physical_confirmation"/name,output/split/"physical_confirmation"/name)
            for name in ("manifest.json","diagnostic_pairing_receipt.json","absent_pairing_receipt.json","evaluation_access.json"):
                copy(ROOT/split/"confirmation_pairs"/name,output/split/"confirmation_pairs"/name)
    for source in (ROOT/"analysis").glob("*.json"):
        copy(source,output/"analysis"/source.name)
    for name in ("sampling_exposure.csv","final_inference_cost.csv","corrected_and_new_errors.csv","expert_error_overlap.csv",
                 "future_dual_tail_quotas.csv"):
        copy(ROOT/"analysis"/name,output/"analysis"/name)
    for name in ("combination_metrics.csv","slice_metrics.csv","label_frequency_metrics.csv"):
        with (ROOT/"analysis"/name).open(encoding="utf-8-sig",newline="") as stream:
            rows = [row for row in csv.DictReader(stream) if row["family"]=="final_confirmation_results"]
        assert rows
        helper.table(output/"analysis"/f"final_{name}",rows)
    for source in (ROOT/"figures").iterdir():
        if source.is_file():
            copy(source,output/"figures"/source.name)
    for source in (ROOT/"frozen_sources").iterdir():
        if source.is_file():
            copy(source,output/"frozen_sources"/source.name)
    write(output/"delivery_verification.json",{"passed":True,"frozen_model_and_threshold_pairs":len(verified),
          "frozen_scientific_sources":len(frozen["sources"]),"models":verified,"time":stamp(),
          "large_artifact_root":str(ROOT),"remote_source_commit":CONFIG["remote_commit"]})
    repo = HERE.parents[1]
    files = []
    for path in sorted(HERE.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts or path.name in ("delivery_manifest.json","continuation_state.json"):
            continue
        if path.suffix in (".pth",".mat",".npz",".tmp",".pyc"):
            continue
        files.append({"path":path.relative_to(repo).as_posix(),"sha256":base.sha(path),"bytes":path.stat().st_size})
    write(HERE/"delivery_manifest.json",{"files":files,"large_artifacts_external":str(ROOT),"time":stamp()})
    print("DELIVERY_PACKAGED",len(files),sum(row["bytes"] for row in files),flush=True)


if __name__=="__main__":
    main()
