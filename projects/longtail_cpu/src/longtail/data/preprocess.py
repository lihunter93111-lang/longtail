"""All signal preprocessing is controlled by dataloader.preprocess."""
import numpy as np


def transform(iq, config):
    values = np.array(iq, dtype=np.float32, copy=True)
    if values.ndim != 2 or values.shape[0] != 2:
        raise ValueError('I/Q must have shape [2, length]')
    start = config['crop_start']
    length = config['crop_length']
    stop = values.shape[-1] if length is None else start + length
    if not 0 <= start < stop <= values.shape[-1]:
        raise ValueError('Crop exceeds signal length')
    values = values[:, start:stop:config['stride']]
    if config['remove_dc']:
        values = values - values.mean(axis=-1, keepdims=True)
    if config['normalization'] == 'complex_rms':
        power = np.mean(values[0] ** 2 + values[1] ** 2)
        values = values / np.sqrt(power + config['epsilon'])
    elif config['normalization'] != 'none':
        raise ValueError('Unknown normalization')
    values = values * config['scale']
    if not np.isfinite(values).all():
        raise ValueError('Signal contains nonfinite values')
    return np.ascontiguousarray(values, dtype=np.float32)
