"""Advance the frozen screening decision through fresh-data confirmation."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def run(command, log):
    print(json.dumps({"starting":command,"log":str(log)},ensure_ascii=False),flush=True)
    with log.open("w",encoding="utf-8") as stream:
        subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,check=True,
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))


def matlab_literal(path):
    return "'"+str(path).replace("'","''")+"'"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--original",type=Path,required=True)
    parser.add_argument("--matlab-executable", default="matlab")
    parser.add_argument("--source-config", type=Path,
                        default=Path(r"E:\论文相关\干扰数据\复合干扰数据0716\九类\full_power_10_30dB_v1\closedset_9class_train.mat"))
    args = parser.parse_args()
    screen = args.root/"screen_20261106_01"
    while not (screen/"decision.json").exists():
        if (screen/"status.json").exists():
            status = json.loads((screen/"status.json").read_text(encoding="utf-8"))
            if status["status"]=="failed":
                raise RuntimeError("Screening reported a failure; inspect its receipt before recovery")
        time.sleep(30)
    decision = json.loads((screen/"decision.json").read_text(encoding="utf-8"))
    run([sys.executable,str(HERE/"summarize_bce_goal.py"),"--output",str(screen)],screen/"summary.log")
    if not decision["confirmation_required"]:
        (args.root/"goal_pipeline_complete.json").write_text(json.dumps({"status":"completed_negative_screen","screen":str(screen)}),encoding="utf-8")
        return
    data = args.root/"confirmation_data_20261106_01"
    data.mkdir(exist_ok=True)
    source_config = args.source_config
    if not all((data/role/"generation_receipt.json").exists() for role in ("fit_pool","dev_pool")):
        command = "addpath("+matlab_literal(HERE)+"); generate_confirmation_candidates("+",".join(matlab_literal(path) for path in (REPO,data,source_config,screen/"decision.json"))+");"
        run([args.matlab_executable,"-batch",command],data/"generation.log")
    run([sys.executable,str(HERE/"prepare_confirmation.py"),"--screen",str(screen),"--original",str(args.original),"--output",str(data)],data/"preparation.log")
    output = args.root/"confirm_20261106_01"
    output.mkdir(exist_ok=True)
    arms = ["A"]
    if decision["selected_training_candidate"] is not None:
        arms.append(decision["selected_training_candidate"])
    run([sys.executable,"-u",str(HERE/"run_resilient.py"),"--source",str(data),"--output",str(output),"--phase","confirm","--arms",*arms,"--seeds","2060","2061","2062","2063","2064"],output/"pipeline.log")
    run([sys.executable,str(HERE/"summarize_bce_goal.py"),"--output",str(output)],output/"summary.log")
    (args.root/"goal_pipeline_complete.json").write_text(json.dumps({"status":"completed_with_confirmation","screen":str(screen),"confirmation":str(output)},ensure_ascii=False,indent=2),encoding="utf-8")
    print("GOAL_PIPELINE_COMPLETE",flush=True)


if __name__=="__main__":
    main()
