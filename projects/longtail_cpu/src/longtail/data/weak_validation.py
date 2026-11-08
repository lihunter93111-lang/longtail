"""Paired natural/hard power views with identical waveform latents and noise."""
import hashlib
import numpy as np
from .signals import Simulator
from .v3_fresh import compose_v3

_SIM = None
TARGET_LABELS = (2, 4, 5)


def initialize(physics):
    global _SIM
    _SIM = Simulator(physics)


def conditional_jnr(seed, active_count, target_position):
    if active_count < 2 or not 0 <= target_position < active_count:
        raise ValueError('A targeted weak example needs two active components')
    raw = hashlib.blake2b(f'weak-power-v1:{int(seed)}'.encode(), digest_size=8).digest()
    stream = np.random.default_rng(int.from_bytes(raw, 'little'))
    for _ in range(10000):
        values = stream.uniform(10, 30, active_count)
        if values[target_position] <= values.max() - 10:
            return values
    raise RuntimeError('Conditional JNR sampler failed')


def paired_sample(simulator, active_labels, classes, seed, target=-1):
    natural = compose_v3(simulator, [classes[c] for c in active_labels], seed, return_components=True)
    if target < 0:
        return natural, natural
    position = np.flatnonzero(np.asarray(active_labels) == target)
    if len(position) != 1 or target not in TARGET_LABELS:
        raise ValueError('Target must be an active SawFM, MFT or ISRJ label')
    new_jnr = conditional_jnr(seed, len(active_labels), int(position[0]))
    components = natural['clean_components'] * 10 ** ((new_jnr - natural['jnr'])[:, None] / 20)
    power = 10 * np.log10(np.mean(np.abs(components) ** 2, axis=1))
    hard = {**natural, 'iq': components.sum(0) + natural['receiver_noise'], 'jnr': new_jnr,
            'power': power, 'relative': power - power.max(), 'clean_components': components}
    return natural, hard


def _block(length, seeds):
    n = len(seeds)
    result = {'iq_complex': np.empty((length, n), np.complex128),
              'label_selected': np.zeros((6, n), np.uint8),
              'sample_seed': np.asarray(seeds, np.uint32)[None, :],
              'receiver_noise_seed': np.empty((1, n), np.uint32),
              'hard_target_label': np.full((1, n), -1, np.int8),
              'weak_component_mask': np.zeros((6, n), np.uint8)}
    for k in ['component_effective_JNR_dB', 'component_measured_power_dB', 'relative_power_dB', 'component_seeds']:
        result[k] = np.full((6, n), np.nan)
    return result


def chunk_worker(task):
    codes, seeds, targets, classes = task
    outputs = [_block(_SIM.length, seeds), _block(_SIM.length, seeds)]
    for i, (code, seed, target) in enumerate(zip(codes, seeds, targets)):
        active = np.flatnonzero((int(code) >> np.arange(6)) & 1)
        views = paired_sample(_SIM, active, classes, int(seed), int(target))
        for v, (block, sample) in enumerate(zip(outputs, views)):
            block['iq_complex'][:, i] = sample['iq']
            block['label_selected'][active, i] = 1
            block['receiver_noise_seed'][0, i] = sample['noise_seed']
            block['hard_target_label'][0, i] = target if v else -1
            for name, field in [('component_effective_JNR_dB', 'jnr'), ('component_measured_power_dB', 'power'),
                                ('relative_power_dB', 'relative'), ('component_seeds', 'component_seeds')]:
                block[name][active, i] = sample[field]
            block['weak_component_mask'][active, i] = (len(active) > 1) & (sample['relative'] <= -10 + 1e-6)
    return tuple(outputs)
