"""Evaluate portable models or export the accepted historical component graph."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cjlt.engine.longtail_workflow import evaluation_main

if __name__ == "__main__":
    evaluation_main()
