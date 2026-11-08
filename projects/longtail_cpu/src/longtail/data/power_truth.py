"""Read explicit component truth; never infer it from a mixed I/Q waveform."""
import h5py
import numpy as np


def read_power_truth(path, labels):
    with h5py.File(path, 'r') as f:
        schema = f.attrs.get('power_truth_schema')
        if isinstance(schema, bytes):
            schema = schema.decode('utf-8')
        if schema != 'per-label-v1':
            return None
        keys = ['component_effective_JNR_dB', 'component_measured_power_dB', 'relative_power_dB', 'weak_component_mask']
        if any(key not in f for key in keys):
            raise ValueError('Incomplete per-label power truth')
        jnr, power, relative, weak = [np.asarray(f[key]).T for key in keys]
        threshold = float(f.attrs['weak_gap_db'])
        noise_power_db = f.attrs.get('noise_power_db')
    active = labels.astype(bool)
    if any(value.shape != labels.shape for value in [jnr, power, relative, weak]) or not np.isfinite(threshold) or threshold <= 0:
        raise ValueError('Invalid per-label power truth shape or threshold')
    if not np.isin(weak, [0, 1]).all():
        raise ValueError('Weak mask must be binary')
    if any(not np.isfinite(value[active]).all() or not np.isnan(value[~active]).all() for value in [jnr, power, relative]):
        raise ValueError('Active power must be finite; absent labels must use NaN')
    expected_relative = power - np.nanmax(power, axis=1, keepdims=True)
    if not np.allclose(relative[active], expected_relative[active], atol=1e-5):
        raise ValueError('Relative power differs from measured component powers')
    if noise_power_db is not None and not np.allclose(jnr[active] + float(noise_power_db), power[active], atol=1e-5):
        raise ValueError('Component JNR differs from measured power and noise reference')
    expected_weak = active & (active.sum(1, keepdims=True) > 1) & (relative <= -threshold + 1e-6)
    if not np.array_equal(weak.astype(bool), expected_weak):
        raise ValueError('Weak mask differs from measured power and threshold')
    return {'component_jnr_db': jnr, 'component_power_db': power,
            'relative_power_db': relative, 'weak_mask': weak.astype(bool)}
