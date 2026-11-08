"""Train-only label/combo resampling with explicit exposure audits."""
import hashlib
import numpy as np
import torch
from torch.utils.data import Sampler


def sampler_epoch_seed(seed, epoch, stream=0):
    """Derive independent reproducible streams without seed+epoch collisions."""
    value = f'longtail-sampler-v2:{int(seed)}:{int(epoch)}:{int(stream)}'.encode()
    return int.from_bytes(hashlib.blake2b(value, digest_size=8).digest(), 'little') & ((1 << 63) - 1)


def sampling_probabilities(labels, mode):
    y=np.asarray(labels,dtype=np.float64)
    if y.ndim!=2 or not len(y) or not np.isin(y,[0,1]).all() or not (y.sum(1)>0).all():
        raise ValueError('Sampling requires nonempty binary multi-label rows')
    if mode=='uniform':weights=np.ones(len(y))
    elif mode=='label_aware':
        counts=y.sum(0)
        if (counts==0).any():raise ValueError('Every sampled label needs positive examples')
        # Equivalent to choosing a label uniformly, then a positive identity
        # uniformly for that label. Co-occurring labels remain correlated.
        weights=(y/counts).sum(1)
    elif mode=='combo_aware':
        code=y.astype(np.int64)@(1<<np.arange(y.shape[1]))
        counts=np.bincount(code);weights=1/counts[code]
    else:raise ValueError('Unsupported sampling mode')
    return weights/weights.sum()


class EpochWeightedSampler(Sampler):
    def __init__(self, labels, mode, seed):
        self.labels=np.asarray(labels,dtype=np.uint8);self.mode=mode;self.seed=int(seed);self.epoch=0
        self.probabilities=sampling_probabilities(self.labels,mode)
        self.weights=torch.from_numpy(self.probabilities);self.last_indices=None

    def set_epoch(self,epoch):self.epoch=int(epoch)

    def __len__(self):return len(self.labels)

    def __iter__(self):
        generator=torch.Generator().manual_seed(sampler_epoch_seed(self.seed,self.epoch))
        self.last_indices=torch.multinomial(self.weights,len(self),replacement=True,generator=generator).numpy()
        return iter(self.last_indices.tolist())

    def audit(self):
        if self.last_indices is None:raise RuntimeError('Sampler has not drawn this epoch')
        observed=self.labels[self.last_indices];code=observed.astype(np.int64)@(1<<np.arange(observed.shape[1]))
        return {'mode':self.mode,'replacement':True,'draws':len(observed),
                'rng_scheme':'blake2b-seed-epoch-stream-v2',
                'rng_seed':sampler_epoch_seed(self.seed,self.epoch),
                'draw_order_sha256':hashlib.sha256(self.last_indices.astype('<i8').tobytes()).hexdigest(),
                'unique_fit_identities':len(np.unique(self.last_indices)),
                'label_exposures':observed.sum(0).astype(int).tolist(),
                'combination_exposures':np.bincount(code,minlength=1<<observed.shape[1])[1:].tolist(),
                'mean_cardinality':float(observed.sum(1).mean())}


class EpochMixedComboSampler(EpochWeightedSampler):
    """Half natural identities without replacement, half combo-balanced draws."""
    def __init__(self, labels, seed):
        super().__init__(labels, 'combo_aware', seed)
        self.mode = 'mixed_combo'
        self.last_natural_indices = self.last_combo_indices = None

    def __iter__(self):
        natural_generator = torch.Generator().manual_seed(sampler_epoch_seed(self.seed, self.epoch, 0))
        combo_generator = torch.Generator().manual_seed(sampler_epoch_seed(self.seed, self.epoch, 1))
        shuffle_generator = torch.Generator().manual_seed(sampler_epoch_seed(self.seed, self.epoch, 2))
        combo_count = len(self) // 2
        natural = torch.randperm(len(self), generator=natural_generator)[:len(self) - combo_count]
        combo = torch.multinomial(self.weights, combo_count, replacement=True, generator=combo_generator)
        joined = torch.cat([natural, combo])
        self.last_natural_indices, self.last_combo_indices = natural.numpy(), combo.numpy()
        self.last_indices = joined[torch.randperm(len(self), generator=shuffle_generator)].numpy()
        return iter(self.last_indices.tolist())

    def audit(self):
        record = super().audit()
        record.update(natural_draws_without_replacement=len(self.last_natural_indices),
                      combo_draws_with_replacement=len(self.last_combo_indices),
                      rng_seeds=[sampler_epoch_seed(self.seed, self.epoch, s) for s in range(3)])
        return record


class EpochBudgetSampler(EpochWeightedSampler):
    """A fixed natural draw budget independent of the available identity pool."""
    def __init__(self, labels, seed, draws):
        super().__init__(labels, 'uniform', seed)
        self.draws = int(draws)
        if not 0 < self.draws <= len(self.labels):
            raise ValueError('Natural draw budget must be within the identity pool')

    def __len__(self):
        return self.draws

    def __iter__(self):
        generator = torch.Generator().manual_seed(sampler_epoch_seed(self.seed, self.epoch))
        self.last_indices = torch.randperm(len(self.labels), generator=generator)[:self.draws].numpy()
        return iter(self.last_indices.tolist())

    def audit(self):
        record = super().audit()
        record.update(replacement=False, fit_pool_size=len(self.labels))
        return record
