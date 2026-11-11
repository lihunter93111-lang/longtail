"""Freeze, generate, audit, and score fresh final cohorts with resumable stage receipts."""
import argparse
import subprocess
from common import *

MATLAB = r"D:\Matlab\bin\matlab.exe"
SOURCE_CONFIG = Path("E:/论文相关/干扰数据/复合干扰数据0716/九类/full_power_10_30dB_v1/closedset_9class_train.mat")


def matlab_string(path):
    return "'"+Path(path).as_posix().replace("'","''")+"'"


def run(stage, command, cutoff="evaluation_cutoff"):
    check_deadline(cutoff)
    write(ROOT/"final_pipeline_status.json",{"status":"running","stage":stage,"time":stamp()})
    remaining = datetime.fromisoformat(CONFIG[cutoff]).timestamp()-time.time()
    with (ROOT/f"{stage}.log").open("a",encoding="utf-8") as log:
        subprocess.run(command,cwd=HERE,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=remaining)
    print("FINAL_STAGE_COMPLETE",stage,flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection",type=Path,required=True)
    args = parser.parse_args()
    if not (ROOT/"final_freeze.json").exists():
        run("final_freeze",[sys.executable,str(HERE/"final_campaign.py"),"freeze","--selection",str(args.selection)])
    frozen = read(ROOT/"final_freeze.json")
    assert frozen["selection_sha256"]==base.sha(args.selection)
    for split,seed in zip(CONFIG["splits"],CONFIG["confirmation_seeds"]):
        if not (ROOT/split/"confirmation/evaluation_pool/generation_receipt.json").exists():
            expression = f"addpath({matlab_string(HERE)}); generate_evaluation_pool({matlab_string(SOURCE)},{matlab_string(ROOT/split/'confirmation')},{matlab_string(SOURCE_CONFIG)},{matlab_string(ROOT/'final_freeze.json')},{seed},'confirmation');"
            run(f"confirmation_generate_{split}",[MATLAB,"-batch",expression])
    if not all((ROOT/split/"confirmation/preflight.json").exists() for split in CONFIG["splits"]):
        run("confirmation_audit",[sys.executable,str(HERE/"audit_inputs.py"),"--role","confirmation"])
    for split in CONFIG["splits"]:
        if not (ROOT/split/"final_confirmation_results/summary.json").exists():
            run(f"confirmation_evaluate_{split}",[sys.executable,str(HERE/"final_campaign.py"),"evaluate","--split",split])
    if frozen["selection"].get("physical_methods"):
        for split in CONFIG["splits"]:
            if not (ROOT/split/"confirmation_pairs/absent_pairing_receipt.json").exists():
                expression = f"addpath({matlab_string(HERE)}); generate_frozen_pairs({matlab_string(SOURCE)},{matlab_string(ROOT)},'{split}');"
                run(f"physical_generate_{split}",[MATLAB,"-batch",expression])
        for split in CONFIG["splits"]:
            if not (ROOT/split/"physical_confirmation/summary.json").exists():
                run(f"physical_evaluate_{split}",[sys.executable,str(HERE/"evaluate_physical.py"),"--split",split])
        run("physical_verification",[sys.executable,str(HERE/"verify_physical.py")],cutoff="deadline")
    for script in ("verify_results.py","record_data_usage.py","analyze_slices.py","analyze_sampling.py","analyze_complementarity.py",
                   "compare_sampling_controls.py","analyze_factorial.py","benchmark_final.py","plot_final.py"):
        run(Path(script).stem,[sys.executable,str(HERE/script)],cutoff="deadline")
    write(ROOT/"final_pipeline_status.json",{"status":"scientific_evaluation_completed","time":stamp(),
          "remaining":"human-readable analysis, visual inspection, packaging and delivery verification"})
    print("FINAL_SCIENTIFIC_PIPELINE_COMPLETE",flush=True)


if __name__=="__main__":
    main()
