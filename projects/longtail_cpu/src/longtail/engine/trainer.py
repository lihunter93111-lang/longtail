"""Training math independent of file formats and model architecture."""
from contextlib import nullcontext
import torch
import torch.distributed as dist
from torch import nn


class WeakPositiveBCE(nn.Module):
    """Mild fit-truth supervision, with unit expected weight mass per label."""
    def __init__(self, weak_counts, sample_count, device):
        super().__init__()
        if weak_counts is None:
            raise ValueError('Weak-positive BCE requires verified fit-only component truth')
        counts = torch.as_tensor(weak_counts, dtype=torch.float32, device=device)
        if counts.shape != (6,) or not torch.isfinite(counts).all() or ((counts < 0) | (counts > sample_count)).any():
            raise ValueError('Invalid fit weak-positive counts')
        selected = torch.tensor([False, False, True, False, True, True], device=device)
        self.register_buffer('selected', selected)
        self.register_buffer('normalizer', 1 + .5 * selected * counts / sample_count)

    def forward(self, logits, targets, weak_mask):
        if weak_mask.shape != targets.shape:
            raise ValueError('Weak mask and labels must have identical shapes')
        weights = 1 + .5 * (weak_mask.bool() & targets.bool() & self.selected)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        return (loss * weights / self.normalizer).mean()


def build_loss(config, counts, sample_count, device, weak_counts=None):
    if config['loss'] == 'bce':
        return nn.BCEWithLogitsLoss()
    if config['loss'] == 'weak_positive_bce':
        return WeakPositiveBCE(weak_counts, sample_count, device)
    prior = torch.tensor(counts, dtype=torch.float32, device=device) / sample_count
    if ((prior <= 0) | (prior >= 1)).any():
        raise ValueError('Logit adjustment requires positive and negative examples per label')
    if config['loss'] == 'weighted_bce':
        return nn.BCEWithLogitsLoss(pos_weight=(1-prior)/prior)
    if config['loss'] != 'logit_adjustment':
        raise ValueError('Unsupported loss')
    bias = torch.logit(prior)
    return lambda logits, targets: nn.functional.binary_cross_entropy_with_logits(logits + bias, targets)


def build_optimizer(model, config):
    optimizer = {'adamw': torch.optim.AdamW, 'adam': torch.optim.Adam}[config['optimizer']]
    return optimizer(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])


def train_epoch(model, loader, optimizer, loss_fn, device, config, scaler):
    model.train()
    total = torch.zeros(2, dtype=torch.float64, device=device)
    dtype = {'float16': torch.float16, 'bfloat16': torch.bfloat16}.get(config['amp'])
    for batch in loader:
        signal = batch['signal'].to(device, non_blocking=True)
        target = batch['target'].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        context = torch.autocast(device.type, dtype=dtype) if dtype is not None else nullcontext()
        with context:
            logits = model(signal)
            if logits.shape != target.shape:
                raise ValueError('Model must output [batch, number_of_labels] logits')
            if isinstance(loss_fn, WeakPositiveBCE):
                if 'weak_mask' not in batch:
                    raise ValueError('Weak-positive BCE needs per-component fit truth')
                loss = loss_fn(logits, target, batch['weak_mask'].to(device, non_blocking=True))
            else:
                loss = loss_fn(logits, target)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite training loss')
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if config['gradient_clip']:
            nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'], error_if_nonfinite=True)
        scaler.step(optimizer)
        scaler.update()
        total[0] += loss.detach().double() * len(target)
        total[1] += len(target)
    if dist.is_initialized():
        dist.all_reduce(total)
    return float((total[0] / total[1]).item()), int(total[1].item())
