"""Full-coverage combo-weighted trainer adapted from the frozen natural trainer.

Only the sample-weighted objective and dependency-root resolution differ.
"""
import hashlib
import json
import importlib.util
import math
import random
import time
from pathlib import Path
import torch
import numpy as np
from scipy.special import expit
from sklearn.metrics import average_precision_score
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
SOURCE = Path(json.loads((HERE / "protocol.json").read_text(encoding="utf-8"))["remote_source"])
spec = importlib.util.spec_from_file_location("previous_weak",SOURCE/"experiment-log/2026-11-06-05-weak-component-goal/run_weak_goal.py")
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)
base = previous.base
logits = previous.logits
atomic_torch_save = previous.atomic_torch_save
ARMS = {name:{"smoothing":0.0,"cosine":True} for name in ("B","W","M")}

def train_one(args, model, inputs, labels, early_x, early_y, seed, arm, initial_hash, fingerprint, weak_inputs, weak_positions, weak15_inputs, weak15_positions):
    folder = args.output / f"{arm}_{seed}"
    folder.mkdir(exist_ok=True)
    receipt_path = folder/"receipt.json"
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        assert receipt["protocol_sha256"] == fingerprint
        assert base.sha(folder/"best_BCE.pth") == receipt["checkpoint_sha256"]
        return receipt
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(seed)
    history = []; orders = []; best_bce = math.inf; best_ap = -math.inf
    best_epoch = 0; best_ap_epoch = 0; elapsed = 0.0
    recovery_path = folder/"recovery.pth"
    if recovery_path.exists():
        recovery = torch.load(recovery_path, map_location="cpu", weights_only=False)
        assert recovery["protocol_sha256"] == fingerprint
        assert recovery["initial_state_sha256"] == initial_hash
        model.load_state_dict(recovery["model_state_dict"])
        optimizer.load_state_dict(recovery["optimizer"])
        history = recovery["history"]; orders = recovery["orders"]
        best_bce = recovery["best_bce"]; best_ap = recovery["best_ap"]
        best_epoch = recovery["best_epoch"]; best_ap_epoch = recovery["best_ap_epoch"]
        elapsed = recovery["elapsed"]
        restored_bce = float(F.binary_cross_entropy_with_logits(logits(model, early_x), early_y).cpu())
        assert abs(restored_bce-history[-1]["early_BCE"]) < 1e-6
        generator.set_state(recovery["order_rng"])
        random.setstate(recovery["python_rng"]); np.random.set_state(recovery["numpy_rng"])
        torch.set_rng_state(recovery["torch_rng"]); torch.cuda.set_rng_state_all(recovery["cuda_rng"])
    started = time.time()
    for epoch in range(len(history)+1,args.epochs+1):
        settings = ARMS[arm]
        learning_rate = 1e-4
        if settings["cosine"]:
            learning_rate = 1e-6 + (1e-4-1e-6)*(1+math.cos(math.pi*(epoch-1)/(args.epochs-1)))/2
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        model.train()
        order = torch.randperm(8000, generator=generator)
        assert len(torch.unique(order)) == 8000
        orders.append(hashlib.sha256(order.numpy().tobytes()).hexdigest())
        if time.time() >= args.training_deadline:
            raise RuntimeError("Training deadline reached; recovery is saved at previous completed epoch")
        epoch_inputs = inputs
        augmented_count = 0
        augmented15 = torch.zeros(len(weak15_positions),dtype=torch.bool,device=weak15_positions.device)
        if arm != "B":
            selected = (weak_positions + epoch) % 2 == 0
            selected_positions = weak_positions[selected]
            epoch_inputs = inputs.clone()
            epoch_inputs[selected_positions] = weak_inputs[selected]
            augmented_count = int(selected.sum())
            if arm == "M":
                augmented15 = ((weak15_positions+epoch)%2==0) & (((epoch+weak15_positions%2)//2)%2==0)
                epoch_inputs[weak15_positions[augmented15]] = weak15_inputs[augmented15]
            else:
                augmented15 = torch.zeros(len(weak15_positions),dtype=torch.bool,device=weak15_positions.device)
        objective_losses = []; hard_losses = []
        for batch_number,batch in enumerate(order.cuda().split(32)):
            if time.time() >= args.training_deadline:
                atomic_torch_save({"model_state_dict":model.state_dict(),"optimizer":optimizer.state_dict(),
                    "epoch":epoch,"completed_batches":batch_number,"order":order,
                    "protocol_sha256":fingerprint,"partial_epoch":True},folder/"deadline_partial.pth")
                raise RuntimeError("Optimization deadline reached. Last full epoch recovery and partial snapshot saved.")
            optimizer.zero_grad(set_to_none=True)
            prediction = model(epoch_inputs[batch]); truth = labels[batch]
            target = (1-settings["smoothing"])*truth + settings["smoothing"]/2
            per_example_loss = F.binary_cross_entropy_with_logits(prediction, target, reduction="none").mean(dim=1)
            loss = (per_example_loss * args.sample_weights[batch]).mean()
            hard = F.binary_cross_entropy_with_logits(prediction.detach(), truth)
            assert torch.isfinite(loss) and torch.isfinite(hard)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5); optimizer.step()
            objective_losses.append(loss.detach()); hard_losses.append(hard)
        early_z = logits(model, early_x)
        early_bce = float(F.binary_cross_entropy_with_logits(early_z, early_y).cpu())
        early_ap = float(average_precision_score(early_y.cpu().numpy(), expit(early_z.cpu().numpy().astype(np.float64)), average="macro"))
        fit_eval = None
        if epoch % 5 == 0:
            fit_eval = float(F.binary_cross_entropy_with_logits(logits(model,inputs),labels).cpu())
        row = {"seed":seed,"arm":arm,"epoch":epoch,"augmented_presentations":augmented_count,"gap15_presentations":int(augmented15.sum()),"lr":learning_rate,"train_objective":float(torch.stack(objective_losses).double().mean().cpu()),"train_hard_BCE":float(torch.stack(hard_losses).double().mean().cpu()),"fit_eval_BCE":fit_eval,"early_BCE":early_bce,"early_macro_AP":early_ap}
        history.append(row)
        if early_bce < best_bce:
            best_bce = early_bce; best_epoch = epoch
            atomic_torch_save({"model_state_dict":model.state_dict(),"epoch":epoch,"early_BCE":early_bce},folder/"best_BCE.pth")
        if early_ap > best_ap:
            best_ap = early_ap; best_ap_epoch = epoch
            atomic_torch_save({"model_state_dict":model.state_dict(),"epoch":epoch,"early_macro_AP":early_ap},folder/"best_AP.pth")
        base.write(folder/"history.json",history)
        atomic_torch_save({"protocol_sha256":fingerprint,"initial_state_sha256":initial_hash,"model_state_dict":model.state_dict(),"optimizer":optimizer.state_dict(),"history":history,"orders":orders,"best_bce":best_bce,"best_ap":best_ap,"best_epoch":best_epoch,"best_ap_epoch":best_ap_epoch,"elapsed":elapsed+time.time()-started,"order_rng":generator.get_state(),"python_rng":random.getstate(),"numpy_rng":np.random.get_state(),"torch_rng":torch.get_rng_state(),"cuda_rng":torch.cuda.get_rng_state_all()},recovery_path)
        base.write(args.output/"status.json",{"status":"training",**row})
        if epoch % 5 == 0:
            print(json.dumps(row),flush=True)
    state = torch.load(folder/"best_BCE.pth",map_location="cuda",weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    check = float(F.binary_cross_entropy_with_logits(logits(model,early_x),early_y).cpu())
    assert abs(check-best_bce)<1e-6
    receipt = {"arm":arm,"seed":seed,"epochs":args.epochs,"updates":250*args.epochs,"unique_fit_per_epoch":8000,"initial_state_sha256":initial_hash,"order_sha256":orders,"best_BCE_epoch":best_epoch,"best_AP_epoch":best_ap_epoch,"best_early_BCE":best_bce,"best_early_AP":best_ap,"checkpoint_sha256":base.sha(folder/"best_BCE.pth"),"protocol_sha256":fingerprint,"elapsed_seconds":elapsed+time.time()-started}
    base.write(receipt_path,receipt)
    return receipt

