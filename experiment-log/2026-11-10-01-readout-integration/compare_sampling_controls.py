"""Paired mechanism comparisons separating replacement sampling and weighting."""
from common import *
from evaluate_stage import pair_comparison


def main():
    results = []
    for split in CONFIG["splits"]:
        manifest = read(ROOT/split/"development/manifest.json")
        records = metadata(manifest["path"],manifest["indices"])
        specifications = {
            "development_results_sampling_controls":[("combo_expert","ordinary_expert"),
                ("bootstrap_expert","ordinary_expert"),("weighted_expert","ordinary_expert"),
                ("combo_expert","bootstrap_expert"),("combo_expert","weighted_expert")],
            "fusion_development_results_controls":[("combo_affine","ordinary_affine"),
                ("bootstrap_affine","ordinary_affine"),("weighted_affine","ordinary_affine"),
                ("combo_affine","bootstrap_affine"),("combo_affine","weighted_affine")]
        }
        for family,pairs in specifications.items():
            destination = ROOT/split/family
            if not (destination/"summary.json").exists():
                continue
            for candidate,reference in pairs:
                result = pair_comparison(candidate,reference,destination,records)
                results.append({"split":split,"family":family,**result})
    write(ROOT/"analysis/sampling_comparisons.json",{"results":results,"time":stamp(),
          "design":"four controls share initial states and forty-epoch budgets; weighted loss shares natural orders"})
    print("SAMPLING_COMPARISONS_COMPLETE",len(results),flush=True)


if __name__=="__main__":
    main()
