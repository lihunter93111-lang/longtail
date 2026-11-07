"""Resume the three verified baseline runs and diagnose epochs 10 through 160."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import time
from pathlib import Path

import torch
import numpy as np
from sklearn.metrics import average_precision_score
from torch.nn import functional as F


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "previous_diagnostic", HERE.parent / "2026-11-06-02-epoch-diagnostic/run_epoch_diagnostic.py"
)
previous = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(previous)
base = previous.base
SEEDS = (2050, 2051, 2052)
LABELS = ("FMB", "AMA", "SawFM", "SinFM", "MFT", "ISRJ")


def evaluate(model, inputs, targets):
    logits = previous.eval_logits(model, inputs)
    loss = float(F.binary_cross_entropy_with_logits(logits, targets).cpu())
    labels = targets.cpu().numpy().astype(bool)
    probabilities = torch.sigmoid(logits).cpu().numpy()
    predictions = probabilities >= 0.5
    ap = average_precision_score(labels, probabilities, average=None)
    true_positive = (predictions & labels).sum(axis=0)
    false_positive = (predictions & ~labels).sum(axis=0)
    false_negative = (~predictions & labels).sum(axis=0)
    f1 = 2 * true_positive / np.maximum(2 * true_positive + false_positive + false_negative, 1)
    return {
        "BCE": loss,
        "macro_AP": float(ap.mean()),
        "fixed05_exact": float((predictions == labels).all(axis=1).mean()),
        "fixed05_macro_F1": float(f1.mean()),
        "per_label_AP": dict(zip(LABELS, map(float, ap))),
        "per_label_FN": dict(zip(LABELS, map(int, false_negative))),
        "per_label_FP": dict(zip(LABELS, map(int, false_positive))),
    }


def analyze(output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = [json.loads((output / f"A_{seed}/history.json").read_text()) for seed in SEEDS]
    assert all([row["epoch"] for row in run] == list(range(1, 161)) for run in runs)
    base.table(output / "curves.csv", [row for run in runs for row in run])
    ap = np.array([[row["early_stop_macro_AP"] for row in run] for run in runs])
    bce = np.array([[row["early_stop_BCE"] for row in run] for run in runs])
    best = np.maximum.accumulate(ap, axis=1)
    comparisons = []
    for horizon in range(10, 161, 10):
        gap = best[:, -1] - best[:, horizon - 1]
        comparisons.append({
            "epoch_budget": horizon,
            "mean_best_AP": float(best[:, horizon - 1].mean()),
            "mean_gap_to_160": float(gap.mean()),
            "max_seed_gap": float(gap.max()),
            "passes": bool(gap.mean() <= 0.001 and gap.max() <= 0.002),
        })
    budget = next(row["epoch_budget"] for row in comparisons if row["passes"])
    mean_min_epoch = int(np.argmin(bce.mean(axis=0))) + 1
    base.table(output / "horizon_selection.csv", comparisons)
    summaries = []
    for epoch in (8, 10, 20, mean_min_epoch, 30, 50, 100, 150, 160):
        row = {"epoch": epoch}
        for key in ("train_BCE", "early_stop_BCE", "early_stop_macro_AP", "fit_eval_BCE"):
            values = [run[epoch - 1][key] for run in runs]
            if all(value is not None for value in values):
                row[key + "_mean"] = float(np.mean(values))
                row[key + "_sd"] = float(np.std(values, ddof=1))
            else:
                row[key + "_mean"] = None
                row[key + "_sd"] = None
        row["best_so_far_AP_mean"] = float(best[:, epoch - 1].mean())
        summaries.append(row)
    base.table(output / "summary_checkpoints.csv", summaries)
    selection = {
        "selected_max_epochs_AP": budget,
        "mean_validation_BCE_min_epoch": mean_min_epoch,
        "mean_validation_BCE_min": float(bce.mean(axis=0).min()),
        "criterion": "Earliest horizon10:10:160 with mean best-AP gap<=.001 and every seed gap<=.002 relative to160",
        "per_seed": [
            {
                "seed": seed,
                "min_validation_BCE_epoch": int(np.argmin(bce[index])) + 1,
                "min_validation_BCE": float(bce[index].min()),
                "best_AP_epoch_within_budget": int(np.argmax(ap[index, :budget])) + 1,
                "best_AP_epoch_within160": int(np.argmax(ap[index])) + 1,
            }
            for index, seed in enumerate(SEEDS)
        ],
        "summary_checkpoints": summaries,
        "capacity_evaluations": [json.loads((output / f"A_{seed}/capacity.json").read_text()) for seed in SEEDS],
        "roles_used": ["fit_A", "early_stop"],
        "evidence_scope": "Baseline fit and validation diagnosis; architecture sufficiency for generalization requires matched comparisons",
        "calibration_used": False,
        "development_used": False,
        "locked_test_used": False,
    }
    base.write(output / "selection.json", selection)
    plt.rcParams.update({
        "font.family": "Microsoft YaHei", "font.size": 11,
        "axes.unicode_minus": False, "axes.spines.top": False,
        "axes.spines.right": False, "savefig.dpi": 180,
    })
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.1), constrained_layout=True)
    epochs = np.arange(10, 161)
    for key, label, color in (
        ("fit_eval_BCE", "训练集 BCE（eval）", "#2674a7"),
        ("early_stop_BCE", "验证集 BCE（eval）", "#cf692d"),
    ):
        values = np.array([[row[key] for row in run if row["epoch"] % 10 == 0] for run in runs])
        mean = values.mean(axis=0)
        sd = values.std(axis=0, ddof=1)
        axes[0].plot(np.arange(10, 161, 10), mean, "o-", color=color, markersize=3, label=label)
        axes[0].fill_between(np.arange(10, 161, 10), np.maximum(0, mean-sd), mean+sd, color=color, alpha=.15)
    axes[0].set(title="相同评估模式下的训练与验证损失", ylabel="BCE", ylim=(0, None))
    axes[0].legend(fontsize=9)
    for index, seed in enumerate(SEEDS):
        axes[1].plot(epochs, bce[index, 9:], linewidth=.8, alpha=.40, label=f"seed {seed}")
    axes[1].plot(epochs, bce[:, 9:].mean(axis=0), color="#172c45", linewidth=2, label="三种子均值")
    axes[1].scatter([mean_min_epoch], [bce.mean(axis=0).min()], color="#b52f45", zorder=5)
    axes[1].set(title=f"逐轮验证损失 · 均值最低在第 {mean_min_epoch} 轮", ylabel="验证 BCE")
    axes[1].legend(fontsize=8)
    values = best[:, 9:] * 100
    axes[2].plot(epochs, ap[:, 9:].mean(axis=0)*100, color="#8799a8", linewidth=.9, alpha=.8, label="当轮 AP 均值")
    axes[2].plot(epochs, values.mean(axis=0), color="#21835c", linewidth=2, label="截至当轮最佳 AP 均值")
    axes[2].fill_between(epochs, values.mean(axis=0)-values.std(axis=0, ddof=1), values.mean(axis=0)+values.std(axis=0, ddof=1), color="#21835c", alpha=.13)
    axes[2].set(title="验证排序指标与预算选择", ylabel="Macro-AP (%)")
    axes[2].legend(fontsize=8, loc="lower right")
    for axis in axes:
        axis.axvline(budget, color="#b52f45", linestyle="--", linewidth=1)
        axis.set(xlim=(10, 160), xlabel="训练轮次")
        axis.set_xticks([10, 20, 40, 60, 80, 100, 120, 140, 160])
        axis.grid(alpha=.18)
    fig.suptitle(f"A 基线 10–160 轮实测诊断 | 3 种子 | fit 8000 / 验证 1008 | AP 容差预算 {budget} 轮", fontsize=15)
    fig.savefig(output / "loss_curves_10_160.png")
    plt.close(fig)
    # Preserve training-mode losses as a separate plot for full traceability.
    fig, axis = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    for key, label in (("train_BCE", "训练 minibatch BCE（Dropout 开启）"), ("early_stop_BCE", "验证 BCE（eval）")):
        values = np.array([[row[key] for row in run[9:]] for run in runs])
        axis.plot(epochs, values.mean(axis=0), label=label)
        axis.fill_between(epochs, values.mean(axis=0)-values.std(axis=0, ddof=1), values.mean(axis=0)+values.std(axis=0, ddof=1), alpha=.15)
    axis.set(xlim=(10, 160), xlabel="训练轮次", ylabel="BCE", title="逐轮损失 · 三种子均值与样本标准差")
    axis.axvline(budget, color="#b52f45", linestyle="--", label=f"预算 {budget} 轮")
    axis.legend(); axis.grid(alpha=.2)
    fig.savefig(output / "training_loss_10_160.png")
    plt.close(fig)
    artifacts = ["curves.csv", "summary_checkpoints.csv", "horizon_selection.csv", "selection.json", "loss_curves_10_160.png", "training_loss_10_160.png"]
    base.write(output / "artifact_sha256.json", {name: base.sha(output/name) for name in artifacts})
    base.write(output / "analysis_receipt.json", {
        "script_sha256": base.sha(Path(__file__)),
        "epochs_per_seed": [len(run) for run in runs],
        "seeds": SEEDS,
        "source": "Completed measured histories; missing fit eval values retained as null",
    })
    base.write(output / "status.json", {"status": "completed", "epochs": 160, "selected_max_epochs_AP": budget, "mean_validation_BCE_min_epoch": mean_min_epoch})
    print(json.dumps({"completed": str(output), "budget": budget, "mean_loss_min_epoch": mean_min_epoch}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    old_preflight = json.loads((args.previous / "preflight.json").read_text())
    assert base.sha(Path(base.__file__)) == old_preflight["imported_script_sha256"]
    assert base.sha(args.source / "manifest.json") == old_preflight["source_manifest_sha256"]
    manifest = json.loads((args.source / "manifest.json").read_text())
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert torch.cuda.is_available()
    protocol = {
        "source_run": str(args.source), "previous_run": str(args.previous),
        "seeds": SEEDS, "resume_epoch": 150, "target_epoch": 160,
        "batch": 32, "updates_per_epoch": 250, "learning_rate": 1e-4,
        "weight_decay": 1e-4, "clip_grad_norm": 5, "precision": "FP32; TF32 off",
        "model": "dfms_iq width16 dropout0.2; all parameters trainable",
        "selection": "Report validation BCE minimum and earliest AP-tolerance horizon (mean .001; each seed .002)",
        "script_sha256": base.sha(Path(__file__)),
        "previous_script_sha256": base.sha(Path(previous.__file__)),
        "source_manifest_sha256": base.sha(args.source / "manifest.json"),
        "torch": torch.__version__, "gpu": torch.cuda.get_device_name(),
        "calibration_used": False, "development_used": False, "locked_test_used": False,
    }
    base.write(output / "protocol.json", protocol)
    assert base.sha(Path(manifest["fit_path"])) == manifest["fit_sha256"]
    assert base.sha(Path(manifest["dev_path"])) == manifest["dev_sha256"]
    fit_x, fit_y = base.load(manifest["fit_path"], manifest["fit_A"])
    early_x, early_y = base.load(manifest["dev_path"], manifest["early_stop"])
    fit_x = fit_x.cuda(); fit_y = torch.from_numpy(fit_y).cuda()
    early_x = early_x.cuda(); early_y = torch.from_numpy(early_y).cuda()
    for seed in SEEDS:
        started = time.time()
        source_dir = args.previous / f"A_{seed}"
        seed_dir = output / f"A_{seed}"
        seed_dir.mkdir()
        recovery = torch.load(source_dir / "recovery.pth", map_location="cpu", weights_only=False)
        history = json.loads((source_dir / "history.json").read_text())
        assert recovery["epoch"] == 150 and recovery["history"] == history
        model = base.build_backbone("dfms_iq", num_classes=6, width=16, dropout=.2).cuda()
        model.load_state_dict(recovery["model_state_dict"])
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        optimizer.load_state_dict(recovery["optimizer"])
        restored = evaluate(model, early_x, early_y)
        bce_difference = abs(restored["BCE"] - history[-1]["early_stop_BCE"])
        assert bce_difference < 1e-6, (seed, bce_difference)
        generator = torch.Generator()
        generator.set_state(recovery["order_rng"])
        torch.set_rng_state(recovery["torch_rng"])
        torch.cuda.set_rng_state_all(recovery["cuda_rng"])
        orders = []
        best_ap = max(row["early_stop_macro_AP"] for row in history)
        shutil.copy2(source_dir / "best_AP.pth", seed_dir / "best_AP.pth")
        for epoch in range(151, 161):
            model.train()
            order = torch.randperm(8000, generator=generator)
            orders.append(hashlib.sha256(order.numpy().tobytes()).hexdigest())
            losses = []
            for indices in order.cuda().split(32):
                optimizer.zero_grad(set_to_none=True)
                logits = model(fit_x[indices])
                loss = F.binary_cross_entropy_with_logits(logits, fit_y[indices])
                assert torch.isfinite(loss)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                optimizer.step()
                losses.append(loss.detach())
            metrics = evaluate(model, early_x, early_y)
            fit_loss = None
            if epoch == 160:
                fit_loss = evaluate(model, fit_x, fit_y)["BCE"]
            row = {
                "seed": seed, "epoch": epoch,
                "train_BCE": float(torch.stack(losses).double().mean().cpu()),
                "early_stop_BCE": metrics["BCE"],
                "early_stop_macro_AP": metrics["macro_AP"], "fit_eval_BCE": fit_loss,
                **{f"early_AP_{label}": metrics["per_label_AP"][label] for label in LABELS},
            }
            history.append(row)
            base.write(seed_dir / "history.json", history)
            if metrics["macro_AP"] > best_ap:
                best_ap = metrics["macro_AP"]
                torch.save({"model_state_dict": model.state_dict(), "epoch": epoch, "early_stop_macro_AP": best_ap}, seed_dir / "best_AP.pth")
            print(json.dumps(row), flush=True)
            base.write(output / "status.json", {"status": "training", "seed": seed, "epoch": epoch})
        torch.save({
            "model_state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),
            "epoch": 160, "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(),
            "order_rng": generator.get_state(), "history": history,
        }, seed_dir / "recovery.pth")
        capacity = {"seed": seed, "epoch160": {"fit": evaluate(model, fit_x, fit_y), "early_stop": evaluate(model, early_x, early_y)}}
        best_state = torch.load(seed_dir / "best_AP.pth", map_location="cuda", weights_only=False)
        model.load_state_dict(best_state["model_state_dict"])
        capacity["best_AP"] = {"epoch": best_state["epoch"], "fit": evaluate(model, fit_x, fit_y), "early_stop": evaluate(model, early_x, early_y)}
        base.write(seed_dir / "capacity.json", capacity)
        base.write(seed_dir / "receipt.json", {
            "seed": seed, "resumed_from_epoch": 150, "completed_epoch": 160,
            "additional_updates": 2500, "total_updates": 40000,
            "source_recovery_sha256": base.sha(source_dir / "recovery.pth"),
            "output_recovery_sha256": base.sha(seed_dir / "recovery.pth"),
            "restored_validation_BCE_absolute_difference": bce_difference,
            "restored_rng_states": ["torch_cpu", "torch_cuda_all", "sample_order"],
            "additional_order_sha256": orders, "elapsed_seconds": time.time()-started,
            "calibration_used": False, "development_used": False, "locked_test_used": False,
        })
    analyze(output)


if __name__ == "__main__":
    main()
