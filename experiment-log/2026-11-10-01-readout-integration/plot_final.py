"""Publication-style descriptive plots for the frozen confirmation comparison."""
from common import *
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    frozen = read(ROOT/"final_freeze.json")
    names = [item["name"] for item in frozen["selection"]["candidates"] if item["name"]!="baseline"]
    labels = {"reference_affine":"Reference + affine", "ordinary_affine":"Ordinary ensemble + affine",
              "combo_affine":"Combo ensemble + affine", "bootstrap_affine":"Uniform-bootstrap ensemble",
              "weighted_affine":"Weighted-loss ensemble", "coverage_affine":"Segment readout + affine",
              "segment_affine":"Natural-fit segment + affine", "combo_segment_local":"Combo + segment residual",
              "combo_global_local":"Combo + global residual", "ordinary_mean":"Ordinary mean ensemble",
              "combo_mean":"Combo mean ensemble", "combo_segment_isrj":"Combo + ISRJ segment residual",
              "combo_segment_saw_mft":"Combo + SawFM/MFT segment residual",
              "ordinary_segment_local":"Ordinary ensemble + segment residual",
              "ordinary_global_local":"Ordinary ensemble + global residual",
              "weighted_segment_local":"Weighted ensemble + segment residual",
              "weighted_global_local":"Weighted ensemble + global residual",
              "weighted_segment_isrj":"Weighted ensemble + ISRJ residual",
              "segment_pca256_affine":"Segment readout, PCA256 + affine",
              "global_pca256_affine":"Global readout, PCA256 + affine",
              "segment_head64_affine":"Segment readout, MLP64 + affine",
              "global_head64_affine":"Global readout, MLP64 + affine",
              "weighted_segment_guarded":"Weighted + guarded segment readout",
              "weighted_isrj_guarded":"Weighted + guarded ISRJ readout"}
    fig,axes = plt.subplots(2,2,figsize=(13,max(7,len(names)*.55+3)),constrained_layout=True)
    values = {}
    for column,split in enumerate(CONFIG["splits"]):
        summary = read(ROOT/split/"final_confirmation_results/summary.json")
        pairs = {p["candidate"]:p for p in summary["comparisons"] if p["reference"]=="baseline"}
        mean = np.asarray([pairs[name]["exact_gain"]*100 for name in names])
        bounds = np.asarray([pairs[name]["exact_seed_t95"] for name in names])*100
        y = np.arange(len(names))
        colors = ["#137C8B" if "segment" in name or "coverage" in name else "#4567A2" for name in names]
        axes[0,column].barh(y,mean,color=colors)
        axes[0,column].errorbar(mean,y,xerr=np.stack((mean-bounds[:,0],bounds[:,1]-mean)),fmt="none",ecolor="#222",capsize=3)
        axes[0,column].axvline(0,color="#555",linewidth=.8)
        axes[0,column].set_yticks(y,[labels.get(name,name) for name in names])
        axes[0,column].invert_yaxis()
        axes[0,column].set_xlabel("Exact-match gain over W32 (percentage points)")
        axes[0,column].set_title(f"{split} | fresh confirmation | paired seed 95% CI")
        baseline = summary["means"]["baseline"]["raw_bce"]
        reductions = [(baseline-summary["means"][name]["raw_bce"])/baseline*100 for name in names]
        axes[1,column].barh(y,reductions,color=colors)
        axes[1,column].set_yticks(y,[labels.get(name,name) for name in names])
        axes[1,column].invert_yaxis()
        axes[1,column].axvline(0,color="#555",linewidth=.8)
        axes[1,column].set_xlabel("BCE reduction over W32 (%)")
        values[split] = {"exact_gain_pp":mean.tolist(),"exact_seed_t95_pp":bounds.tolist(),"BCE_reduction_pct":reductions}
    for ax in axes.flat:
        ax.spines[["top","right"]].set_visible(False)
        ax.grid(axis="x",alpha=.18)
        ax.set_axisbelow(True)
    destination = ROOT/"figures"
    destination.mkdir(exist_ok=True)
    fig.savefig(destination/"final_confirmation.png",dpi=160)
    fig.savefig(destination/"final_confirmation.pdf")
    write(destination/"final_confirmation_values.json",{"names":names,"values":values})
    print("FINAL_PLOT_COMPLETE",flush=True)


if __name__=="__main__":
    main()
