"""Apply the declared minimum-budget tolerance after all nine runs finish."""
import csv
import json
from pathlib import Path

import torch
import numpy as np

import training

HERE = Path(__file__).resolve().parent
ROOT = Path("E:/LJQ_code/LJQ_code_experiment_storage/daytime_v1/20261109_01")
base = training.base


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    rows, receipts = {}, {}
    for epochs in (40, 80, 160):
        folder = ROOT / f"budget_e{epochs}"
        assert read(folder / "status.json")["status"] == "completed"
        with (folder / "evaluation_metrics.csv").open(encoding="utf-8-sig") as handle:
            rows[epochs] = [row for row in csv.DictReader(handle) if row["variant"] == "raw"]
        assert len(rows[epochs]) == 3
        receipts[epochs] = [read(folder / f"M_{seed}/receipt.json") for seed in (2100, 2101, 2102)]
    for index in range(3):
        assert len({receipts[epochs][index]["initial_state_sha256"] for epochs in receipts}) == 1
        assert receipts[40][index]["order_sha256"] == receipts[80][index]["order_sha256"][:40]
        assert receipts[40][index]["order_sha256"] == receipts[160][index]["order_sha256"][:40]
    summary = {epochs: {key: float(np.mean([float(row[key]) for row in selected]))
               for key in ("BCE", "macro_AP", "f1_exact", "tail_exact", "ECE", "confident_wrong_fraction", "confident_wrong_loss_share")}
               for epochs, selected in rows.items()}
    reference = summary[160]
    checks = {}
    for epochs, values in summary.items():
        checks[epochs] = {"BCE_within1pct": values["BCE"] <= reference["BCE"] * 1.01,
            "AP_within0_1pp": values["macro_AP"] >= reference["macro_AP"] - .001,
            "exact_within0_5pp": values["f1_exact"] >= reference["f1_exact"] - .005}
    chosen = min(epochs for epochs in checks if all(checks[epochs].values()))
    result = {"selected_epochs": chosen, "role": "adaptive_development_budget_selection",
        "rule": "shortest budget within1% BCE,0.1pp macroAP,0.5pp exact of160 epochs",
        "checks": checks, "metrics": summary, "seeds": [2100, 2101, 2102],
        "shared_initial_and_first40_orders_match": True,
        "best_BCE_epochs": {epochs: [r["best_BCE_epoch"] for r in group] for epochs, group in receipts.items()},
        "median_training_seconds": {epochs: float(np.median([r["elapsed_seconds"] for r in group])) for epochs, group in receipts.items()},
        "source_sha256": base.sha(Path(__file__)), "prompt_sha256": base.sha(HERE / "02_BUDGET_GOAL.txt")}
    base.write(ROOT / "budget_decision.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
