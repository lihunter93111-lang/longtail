"""Post-freeze layout correction for the unchanged confirmation statistics."""
import shutil
from common import *
import plot_final


def main():
    directory = ROOT/"figures"
    archive = directory/"draft_layout"
    archive.mkdir(exist_ok=True)
    for suffix in ("png","pdf"):
        source = directory/f"final_confirmation.{suffix}"
        destination = archive/f"final_confirmation_v1.{suffix}"
        if not destination.exists():
            shutil.copy2(source,destination)
    numeric_path = directory/"final_confirmation_values.json"
    original_hash = base.sha(numeric_path)
    plot_final.main()
    assert base.sha(numeric_path)==original_hash
    figure = plot_final.plt.gcf()
    names = read(numeric_path)["names"]
    for index,split in enumerate(CONFIG["splits"]):
        figure.axes[index].set_title(f"{split} | fresh confirmation\nPaired-seed 95% CI",fontsize=11)
        figure.axes[index].set_xlabel("Exact gain over W32 (percentage points)")
    for axis in figure.axes:
        for patch,name in zip(axis.patches,names):
            patch.set_facecolor("#137C8B" if any(word in name for word in ("segment","coverage","isrj")) else "#4567A2")
    figure.savefig(directory/"final_confirmation.png",dpi=160,bbox_inches="tight",pad_inches=.12)
    figure.savefig(directory/"final_confirmation.pdf",bbox_inches="tight",pad_inches=.12)
    write(directory/"final_layout_receipt.json",{"role":"post-freeze layout-only adjustment","numeric_payload_sha256":original_hash,
          "original_plot_source_sha256":base.sha(HERE/"plot_final.py"),"layout_source_sha256":base.sha(Path(__file__)),
          "final_freeze_sha256":base.sha(ROOT/"final_freeze.json"),"time":stamp()})
    print("FINAL_LAYOUT_VERIFIED_UNCHANGED_STATISTICS",flush=True)


if __name__=="__main__":
    main()
