"""Restore missing logging import while retaining the frozen numeric trainer."""
import json

import training
import run_stage2

training.json = json

if __name__ == "__main__":
    run_stage2.main()
