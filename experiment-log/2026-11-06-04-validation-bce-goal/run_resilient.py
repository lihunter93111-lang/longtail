"""Retry transient Windows atomic-rename failures around the frozen trainer."""
import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("frozen_bce_trainer", HERE/"run_bce_goal.py")
trainer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(trainer)
original_write = trainer.base.write


def retry_write(path, value):
    for attempt in range(20):
        try:
            original_write(path, value)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(min(.05*(attempt+1), .5))


if __name__=="__main__":
    if "--help" in sys.argv or "-h" in sys.argv:
        trainer.main()
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output", type=Path, required=True)
    wrapper_args, _ = parser.parse_known_args()
    trainer.base.write = retry_write
    output = wrapper_args.output
    output.mkdir(parents=True,exist_ok=True)
    receipt = {
        "change": "Retry PermissionError during JSON atomic replacement; trainer and numerical protocol use their frozen source hashes",
        "wrapper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "trainer_sha256": hashlib.sha256((HERE/"run_bce_goal.py").read_bytes()).hexdigest(),
        "command": sys.argv,
        "registered_unix_time": time.time(),
    }
    retry_write(output/f"io_recovery_{time.time_ns()}.json", receipt)
    trainer.main()
