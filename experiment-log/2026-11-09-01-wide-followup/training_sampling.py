"""Matched resumable training adapted from the completed weak-component study."""
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
spec = importlib.util.spec_from_file_location("previous_weak",HERE.parent/"2026-11-06-05-weak-component-goal/run_weak_goal.py")
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
    history = []; orders = []; sampling_counts = []; best_bce = math.inf; best_ap = -math.inf
    best_epoch = 0; best_ap_epoch = 0; elapsed = 0.0
    recovery_path = folder/"recovery.pth"
    if recovery_path.exists():
        recovery = torch.load(recovery_path, map_location="cpu", weights_only=False)
        assert recovery["protocol_sha256"] == fingerprint
        assert recovery["initial_state_sha256"] == initial_hash
        model.load_state_dict(recovery["model_state_dict"])
        optimizer.load_state_dict(recovery["optimizer"])
        history = recovery["history"]; orders = recovery["orders"]
        sampling_counts = recovery["sampling_counts"]
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
        order = torch.multinomial(args.sample_probabilities, 8000, replacement=True, generator=generator)
        counts = torch.bincount(order, minlength=8000).numpy().astype(np.int32)
        sampling_counts.append(counts)
        unique_identities = int(np.count_nonzero(counts))
        assert counts.sum() == 8000
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
        weak_mask = torch.zeros(8000,dtype=torch.bool)
        weak_mask[selected_positions.cpu()] = True
        gap15_mask = torch.zeros(8000,dtype=torch.bool)
        gap15_mask[weak15_positions[augmented15].cpu()] = True
        actual_weak = int(weak_mask[order].sum())
        actual_gap15 = int(gap15_mask[order].sum())
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
            loss = F.binary_cross_entropy_with_logits(prediction, target)
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
        row = {"seed":seed,"arm":arm,"epoch":epoch,"augmented_presentations":actual_weak,"gap15_presentations":actual_gap15,"eligible_augmented_identities":augmented_count,"unique_sampled_identities":unique_identities,"repetition_fraction":1-unique_identities/8000,"lr":learning_rate,"train_objective":float(torch.stack(objective_losses).double().mean().cpu()),"train_hard_BCE":float(torch.stack(hard_losses).double().mean().cpu()),"fit_eval_BCE":fit_eval,"early_BCE":early_bce,"early_macro_AP":early_ap}
        history.append(row)
        if early_bce < best_bce:
            best_bce = early_bce; best_epoch = epoch
            atomic_torch_save({"model_state_dict":model.state_dict(),"epoch":epoch,"early_BCE":early_bce},folder/"best_BCE.pth")
        if early_ap > best_ap:
            best_ap = early_ap; best_ap_epoch = epoch
            atomic_torch_save({"model_state_dict":model.state_dict(),"epoch":epoch,"early_macro_AP":early_ap},folder/"best_AP.pth")
        base.write(folder/"history.json",history)
        atomic_torch_save({"protocol_sha256":fingerprint,"initial_state_sha256":initial_hash,"model_state_dict":model.state_dict(),"optimizer":optimizer.state_dict(),"history":history,"orders":orders,"sampling_counts":sampling_counts,"best_bce":best_bce,"best_ap":best_ap,"best_epoch":best_epoch,"best_ap_epoch":best_ap_epoch,"elapsed":elapsed+time.time()-started,"order_rng":generator.get_state(),"python_rng":random.getstate(),"numpy_rng":np.random.get_state(),"torch_rng":torch.get_rng_state(),"cuda_rng":torch.cuda.get_rng_state_all()},recovery_path)
        base.write(args.output/"status.json",{"status":"training",**row})
        if epoch % 5 == 0:
            print(json.dumps(row),flush=True)
    state = torch.load(folder/"best_BCE.pth",map_location="cuda",weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    check = float(F.binary_cross_entropy_with_logits(logits(model,early_x),early_y).cpu())
    assert abs(check-best_bce)<1e-6
    counts_array = np.stack(sampling_counts)
    np.savez_compressed(folder/"sample_counts.npz",counts=counts_array,probabilities=args.sample_probabilities.numpy())
    total_counts = counts_array.sum(axis=0).astype(np.float64)
    receipt = {"sampling":"cardinality_stratified_combo_mix_0.5","fit_identity_count":8000,
        "sample_counts_sha256":base.sha(folder/"sample_counts.npz"),
        "total_unique_identities":int(np.count_nonzero(total_counts)),
        "realized_effective_sample_size":float(total_counts.sum()**2/np.square(total_counts).sum()),
        "expected_weight_effective_sample_size":float(1/args.sample_probabilities.square().sum()),
        "arm":arm,"seed":seed,"epochs":args.epochs,"updates":250*args.epochs,"unique_fit_per_epoch":[row["unique_sampled_identities"] for row in history],"initial_state_sha256":initial_hash,"order_sha256":orders,"best_BCE_epoch":best_epoch,"best_AP_epoch":best_ap_epoch,"best_early_BCE":best_bce,"best_early_AP":best_ap,"checkpoint_sha256":base.sha(folder/"best_BCE.pth"),"protocol_sha256":fingerprint,"elapsed_seconds":elapsed+time.time()-started}
    base.write(receipt_path,receipt)
    return receipt

