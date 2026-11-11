"""Paired-seed readout effect plots for the two development splits."""
from common import *
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import t as student_t


def main():
    methods = [("Projected global", "projected_global", "development_results"),
               ("Global statistics", "global_stats", "development_results"),
               ("Segment", "segment", "development_results"),
               ("Segment + M views", "segment_coverage", "development_results_coverage"),
               ("Scaled segment", "segment_scaled", "development_results_scaled")]
    figure, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    numbers = []
    for row, split in enumerate(CONFIG["splits"]):
        gains = {"exact": [], "bce": []}
        for label, group, result in methods:
            candidate = [read(ROOT / split / result / f"{group}_{seed}" / "metrics.json")["calibrated"] for seed in CONFIG["seeds"]]
            reference = [read(ROOT / split / result / f"baseline_{seed}" / "metrics.json")["calibrated"] for seed in CONFIG["seeds"]]
            exact = np.array([100*(a["exact"]-b["exact"]) for a,b in zip(candidate,reference)])
            bce = np.array([100*(b["raw_bce"]-a["raw_bce"])/b["raw_bce"] for a,b in zip(candidate,reference)])
            gains["exact"].append(exact)
            gains["bce"].append(bce)
            numbers.append({"split":split,"method":group,"exact_gain_pp":exact.tolist(),"relative_BCE_improvement_pct":bce.tolist()})
        for column,key in enumerate(("exact","bce")):
            ax = axes[row,column]
            means = [value.mean() for value in gains[key]]
            errors = [student_t.ppf(.975,4)*value.std(ddof=1)/np.sqrt(5) for value in gains[key]]
            positions = np.arange(len(methods))
            ax.errorbar(means,positions,xerr=errors,fmt="o",capsize=4,color="#176b87",markersize=6)
            ax.axvline(0,color="#777777",linewidth=1)
            ax.set_yticks(positions,[item[0] for item in methods])
            ax.invert_yaxis()
            ax.set_xlabel("Exact improvement (percentage points)" if key=="exact" else "Relative raw BCE improvement (%)")
            ax.set_title(f"{split} | paired model seeds")
            ax.grid(axis="x",alpha=.2)
    figure.suptitle("W32 readout development effects | five-seed mean and paired t 95% interval",fontsize=13)
    destination=ROOT/"figures"
    destination.mkdir(exist_ok=True)
    figure.savefig(destination/"readout_effects.png",dpi=180)
    plt.close(figure)
    write(destination/"readout_effects_values.json",numbers)


if __name__=="__main__":
    main()
