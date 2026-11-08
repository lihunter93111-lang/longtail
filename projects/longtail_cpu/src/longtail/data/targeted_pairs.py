"""Rejection-sampled one-bit pairs within original independent10--30dB physics."""
import numpy as np
from . import one_bit
from .v3_fresh import compose_v3

GAP_BINS = [(6., 10.), (10., 15.), (15., 20.)]
MAX_ATTEMPTS = 16384


def initialize(physics):
    one_bit.initialize(physics)


def conditional_worker(task):
    background, target, gap_bin, first_seed, count, edge, classes = task
    lo, hi = GAP_BINS[gap_bin]
    active = np.flatnonzero(((background | (1 << target)) >> np.arange(6)) & 1)
    position = int(np.flatnonzero(active == target)[0])
    selected, attempts = [], []
    for offset in range(MAX_ATTEMPTS):
        seed = first_seed + offset
        sample = compose_v3(one_bit._SIM, [classes[c] for c in active], seed)
        gap = -float(sample['relative'][position])
        if lo <= gap < hi:
            selected.append(seed)
            attempts.append(offset+1)
        if len(selected) == count:
            break
    if len(selected) != count:
        raise RuntimeError(f'Conditional cell exhausted: {task[:6]} accepted={len(selected)}')
    block = one_bit.chunk_worker(([background]*count, [target]*count, selected, [edge]*count, classes))
    actual_gap = -block['relative_power_dB'][target, 1::2]
    assert np.all((actual_gap >= lo) & (actual_gap < hi))
    block['target_gap_bin'] = np.full((1, 2*count), gap_bin, np.int8)
    block['rejection_attempts'] = np.repeat(np.asarray(attempts, np.int32), 2)[None, :]
    return block
