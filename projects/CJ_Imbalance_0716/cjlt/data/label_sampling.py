"""Deterministic identity budgets and label/combo exposure sampling."""
import hashlib
import numpy as np


def epoch_seed(seed, epoch, stream=0):
    value = "longtail-sampler-v2:{}:{}:{}".format(int(seed), int(epoch), int(stream)).encode()
    return int.from_bytes(hashlib.blake2b(value, digest_size=8).digest(), "little") & ((1 << 63) - 1)


def sampling_probabilities(labels, mode):
    labels = np.asarray(labels, dtype=np.float64)
    if labels.ndim != 2 or not len(labels) or not np.isin(labels, [0, 1]).all() or not (labels.sum(1) > 0).all():
        raise ValueError("Expected nonempty multilabel rows")
    if mode == "uniform":
        weights = np.ones(len(labels))
    elif mode == "label_aware":
        counts = labels.sum(0)
        if (counts == 0).any():
            raise ValueError("Label-aware sampling needs each label in fit")
        weights = (labels / counts).sum(1)
    elif mode == "combo_aware":
        codes = labels.astype(np.int64) @ (1 << np.arange(labels.shape[1]))
        weights = 1.0 / np.bincount(codes)[codes]
    else:
        raise ValueError("Unknown sampling mode: {}".format(mode))
    return weights / weights.sum()


def draw_indices(labels, count, seed, epoch, mode="uniform", stream=0):
    # Keep the original Torch RNG draw streams for the existing natural/combo protocols.
    import torch
    if count < 1:
        raise ValueError("Draw count must be positive")
    generator = torch.Generator().manual_seed(epoch_seed(seed, epoch, stream))
    if mode == "uniform":
        if count > len(labels):
            raise ValueError("Natural per-epoch budget exceeds independent fit identities")
        return torch.randperm(len(labels), generator=generator)[:count].numpy()
    if mode == "mixed_combo":
        combo_count = count // 2
        natural = torch.randperm(len(labels), generator=generator)[:count - combo_count]
        if len(natural) != count - combo_count:
            raise ValueError("Mixed natural budget exceeds fit pool")
        probabilities = torch.from_numpy(sampling_probabilities(labels, "combo_aware"))
        combo_generator = torch.Generator().manual_seed(epoch_seed(seed, epoch, stream + 1))
        combined = torch.cat((natural, torch.multinomial(probabilities, combo_count, True, generator=combo_generator)))
        shuffle = torch.Generator().manual_seed(epoch_seed(seed, epoch, stream + 2))
        return combined[torch.randperm(count, generator=shuffle)].numpy()
    probabilities = torch.from_numpy(sampling_probabilities(labels, mode))
    return torch.multinomial(probabilities, count, True, generator=generator).numpy()
