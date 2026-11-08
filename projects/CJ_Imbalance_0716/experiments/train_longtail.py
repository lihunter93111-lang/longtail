"""Train the AMA long-tail backbone baselines with matched seed budgets."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cjlt.engine.longtail_workflow import training_main

if __name__ == "__main__":
    training_main("baseline")
