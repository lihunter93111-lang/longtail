"""Post-freeze visualization using identical gap-15-feasible identities at every gap."""
from common import *
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    names = ["baseline","ordinary_affine","weighted_affine","weighted_segment_guarded"]
    labels = ["W32","Ordinary ensemble","Weighted ensemble","Guarded local readout"]
    colors = ["#6C757D","#4567A2","#B56530","#137C8B"]
    gaps = [0,5,10,15]
    fig,axes = plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
    tables = {}
    for column,split in enumerate(CONFIG["splits"]):
        folder = ROOT/split/"physical_confirmation"
        assert (folder/"summary.json").exists()
        with np.load(folder/"baseline_2200/gap15.npz") as data:
            positions,targets = data["positions"],data["targets"]
        curves,false_positives = [],[]
        for name in names:
            seed_curves,seed_false = [],[]
            for seed in CONFIG["seeds"]:
                directory = folder/f"{name}_{seed}"
                recalls = []
                for gap in gaps:
                    with np.load(directory/f"gap{gap}.npz") as data:
                        lookup = np.searchsorted(data["positions"],positions)
                        np.testing.assert_array_equal(data["positions"][lookup],positions)
                        np.testing.assert_array_equal(data["targets"][lookup],targets)
                        thresholds = data["thresholds"][targets]
                        scores = data["logits"][lookup][np.arange(len(targets)),targets]
                        recalls.append(float((expit(scores)>=thresholds).mean()))
                with np.load(directory/"absent.npz") as data:
                    lookup = np.searchsorted(data["positions"],positions)
                    np.testing.assert_array_equal(data["positions"][lookup],positions)
                    np.testing.assert_array_equal(data["targets"][lookup],targets)
                    scores = data["logits"][lookup][np.arange(len(targets)),targets]
                    seed_false.append(float((expit(scores)>=thresholds).mean()))
                seed_curves.append(recalls)
            curves.append(np.asarray(seed_curves))
            false_positives.append(np.asarray(seed_false))
        for name,label,color,values in zip(names,labels,colors,curves):
            mean,spread = values.mean(0)*100,values.std(0,ddof=1)*100
            axes[0,column].plot(gaps,mean,marker="o",color=color,label=label)
            axes[0,column].fill_between(gaps,np.maximum(mean-spread,0),np.minimum(mean+spread,100),color=color,alpha=.13)
        axes[0,column].set_title(f"{split} | {len(positions)} shared identities")
        axes[0,column].set_xlabel("Target gap below strongest background (dB)")
        axes[0,column].set_ylabel("Target recall (%)")
        axes[0,column].set_xticks(gaps)
        axes[0,column].set_ylim(50,101)
        means = np.asarray([values.mean()*100 for values in false_positives])
        spreads = np.asarray([values.std(ddof=1)*100 for values in false_positives])
        axes[1,column].bar(np.arange(len(names)),means,yerr=spreads,color=colors,capsize=3)
        axes[1,column].set_xticks(np.arange(len(names)),["W32","Ordinary","Weighted","Guarded\nlocal"])
        axes[1,column].set_title("Matched target-removed inputs")
        axes[1,column].set_ylabel("False-positive rate (%)")
        axes[1,column].set_ylim(0,max(means+spreads)*1.2)
        tables[split] = {"identities":len(positions),"target_counts":{LABELS[int(k)]:int((targets==k).sum()) for k in np.unique(targets)},
            "recall_per_seed":{name:values.tolist() for name,values in zip(names,curves)},
            "absent_FPR_per_seed":{name:values.tolist() for name,values in zip(names,false_positives)}}
    axes[0,0].legend(loc="lower left",fontsize=9)
    for ax in axes.flat:
        ax.spines[["top","right"]].set_visible(False)
        ax.grid(axis="y",alpha=.18)
        ax.set_axisbelow(True)
    fig.suptitle("Frozen physical interventions | five-seed mean and one SD",fontsize=13)
    destination = ROOT/"figures"
    fig.savefig(destination/"physical_common_identities.png",dpi=180)
    fig.savefig(destination/"physical_common_identities.pdf")
    write(destination/"physical_common_identities_values.json",{"role":"post-freeze descriptive visualization",
        "gaps":gaps,"subset":"the gap15-feasible identities are reused at every plotted gap within each split",
        "values":tables,"source_sha256":base.sha(Path(__file__)),"freeze_sha256":base.sha(ROOT/"final_freeze.json"),"time":stamp()})
    print("PHYSICAL_COMMON_COHORT_PLOT_COMPLETE",flush=True)


if __name__=="__main__":
    main()
