"""Apply predeclared three-seed diagnostic replication rules."""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.append(str(HERE.parent / "2026-11-08-01-overnight"))
from compare_sampling import matched_compare, base


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = matched_compare(args.reference, args.candidate, args.source, [2120, 2121, 2122])
    del result["checks"]["positive_seeds_ge4"]
    result["checks"]["positive_seeds_all3"] = all(value > 0 for value in result["paired_BCE_gain"])
    result["passes"] = all(result["checks"].values())
    result["role"] = "three_seed_exploratory_independent_training_split_replication"
    result["prompt_sha256"] = base.sha(HERE / "04B_TIME_BOUNDED_REPLICATION_GOAL.txt")
    base.write(args.output, result)
    print("PARTIAL_REPLICATION", result["passes"], result["relative_BCE_reduction"], flush=True)


if __name__ == "__main__":
    main()
