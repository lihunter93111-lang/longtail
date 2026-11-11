"""Evaluate PCA/head-width controls with their explicit architecture loader."""
import argparse
from common import *
from adaptive_readouts import load_adaptive
from evaluate_stage import evaluate_split


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--split",choices=CONFIG["splits"],required=True)
    args=parser.parse_args()
    setup()
    assert read(ROOT/args.split/"adaptive_status.json")["status"]=="completed"
    groups=["global_stats_coverage","segment_coverage","global_stats_pca256","segment_pca256",
            "global_stats_head64","segment_head64"]
    evaluate_split(args.split,"development",groups,"dimensions",load_adaptive)


if __name__=="__main__":
    main()
