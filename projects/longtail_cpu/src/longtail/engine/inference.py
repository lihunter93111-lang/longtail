"""Inference shared by validation, calibration and standalone evaluation."""
import numpy as np
import torch


def predict(model, loader, device):
    model.eval()
    ys, zs, indices, identities = [], [], [], []
    truth = {}
    with torch.inference_mode():
        for batch in loader:
            z = model(batch['signal'].to(device, non_blocking=True))
            if z.shape != batch['target'].shape or not torch.isfinite(z).all():
                raise ValueError('Invalid model output')
            ys.append(batch['target'].numpy())
            zs.append(z.float().cpu().numpy())
            indices.append(batch['index'].numpy())
            identities.append(batch['identity'].numpy())
            for key in ['weak_mask', 'relative_power_db', 'component_jnr_db', 'component_power_db']:
                if key in batch:
                    truth.setdefault(key, []).append(batch[key].numpy())
    return {'targets': np.concatenate(ys).astype(np.uint8), 'logits': np.concatenate(zs),
            'indices': np.concatenate(indices), 'identities': np.concatenate(identities), **{key: np.concatenate(values) for key, values in truth.items()}}
