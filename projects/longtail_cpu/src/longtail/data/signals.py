"""Six physical signal models ported from the legacy equations to NumPy.

PCG64 streams define a new reproducible dataset, not MATLAB RNG parity.
Each function returns a clean, unscaled component and its physical parameters.
"""
from copy import deepcopy
import numpy as np


DEFAULT_PHYSICS = {
    'radar': {'fs_hz': 10e6, 'pri_s': 300e-6, 'pulse_width_s': 100e-6,
              'bandwidth_hz': 5e6, 'carrier_hz': 10e9, 'noise_power': 1.0},
    'arrival_delay_s': [10e-6, 50e-6],
    'FMB': {'bandwidth_factor': [1.5, 3.0], 'center_frequency_hz': [0.0, 0.0]},
    'AMA': {'bandwidth_factor': [1.0, 2.0], 'center_frequency_hz': [0.0, 0.0],
            'modulation_index': [1.0, 3.0]},
    'SawFM': {'sweep_bandwidth_hz': [2e6, 5e6], 'sweep_period_s': [20e-6, 80e-6]},
    'SinFM': {'center_frequency_hz': [0.5e6, 1.5e6], 'frequency_deviation_hz': [1e6, 3e6],
              'modulation_frequency_hz': [10e3, 20e3]},
    'MFT': {'num_targets': [3, 8], 'target_delay_s': [30e-6, 50e-6],
            'target_relative_amplitude': [0.5, 1.5]},
    'ISRJ': {'sample_width_s': [5e-6, 20e-6], 'num_replicas': [1, 5]},
}
REGISTRY_IDS = {'FMB': 1, 'AMA': 2, 'SawFM': 3, 'SinFM': 4, 'MFT': 6, 'ISRJ': 8}
PARAMETERS = {
    'FMB': ['bandwidth_factor', 'interference_bandwidth_hz', 'center_frequency_hz', 'arrival_delay_s'],
    'AMA': ['bandwidth_factor', 'interference_bandwidth_hz', 'center_frequency_hz', 'modulation_index', 'arrival_delay_s'],
    'SawFM': ['sweep_bandwidth_hz', 'sweep_slope_hz_per_s', 'sweep_period_s', 'arrival_delay_s'],
    'SinFM': ['center_frequency_hz', 'frequency_deviation_hz', 'modulation_frequency_hz', 'modulation_period_s', 'arrival_delay_s'],
    'MFT': ['num_targets', 'mean_delay_s', 'mean_spacing_s'] +
           [f'delay_{i:02d}_s' for i in range(1, 9)] + [f'amplitude_{i:02d}' for i in range(1, 9)],
    'ISRJ': ['sample_width_s', 'num_replicas', 'sample_period_s', 'arrival_delay_s', 'mode_code'],
}


def rng(seed):
    return np.random.Generator(np.random.PCG64(int(seed)))


def substream_seed(sample_seed, stream_id, type_id=0):
    return (int(sample_seed) + stream_id * 2246822519 + type_id * 2654435761) % 2**32


def _sample(stream, bounds):
    return bounds[0] + (bounds[1] - bounds[0]) * stream.random()


def _round_samples(seconds, fs):
    # MATLAB rounds positive half samples away from zero.
    return int(np.floor(seconds * fs + 0.5))


def _colored_noise(stream, length, bandwidth, fs, normalization):
    frequency = np.fft.fftfreq(length, 1 / fs)
    support = np.abs(frequency) <= bandwidth / 2
    realized = np.ptp(frequency[support & (frequency != 0)])
    values = np.fft.ifft(np.fft.fft(stream.standard_normal(length)) * support).real
    values -= values.mean()
    scale = np.max(np.abs(values)) if normalization == 'peak' else np.sqrt(np.mean(values**2))
    return values / scale, float(realized)


class Simulator:
    def __init__(self, physics=None):
        self.physics = deepcopy(DEFAULT_PHYSICS if physics is None else physics)
        self.radar = self.physics['radar']
        self.fs = self.radar['fs_hz']
        self.length = _round_samples(self.radar['pri_s'], self.fs)
        pulse_length = _round_samples(self.radar['pulse_width_s'], self.fs)
        if self.fs <= 0 or self.length < 1 or not 1 <= pulse_length <= self.length or self.radar['noise_power'] <= 0:
            raise ValueError('Invalid sampling, pulse duration or noise power')
        self.time = np.arange(self.length) / self.fs
        self.reference = np.zeros(self.length, dtype=np.complex128)
        t = np.arange(pulse_length) / self.fs
        self.reference[:pulse_length] = np.exp(1j * np.pi * self.radar['bandwidth_hz'] /
                                                self.radar['pulse_width_s'] * t**2)

    def component(self, code, seed):
        stream = rng(seed)
        signal, parameters = getattr(self, '_' + code)(stream, self.physics[code])
        signal = np.asarray(signal, dtype=np.complex128)
        power = np.mean(np.abs(signal)**2)
        if not np.isfinite(signal).all() or not np.isfinite(power) or power <= 0:
            raise ValueError(f'Invalid clean component: {code}')
        return signal / np.sqrt(power), parameters

    def _arrival(self, stream):
        return _round_samples(_sample(stream, self.physics['arrival_delay_s']), self.fs)

    def _FMB(self, stream, p):
        requested = _sample(stream, p['bandwidth_factor']) * self.radar['bandwidth_hz']
        center = _sample(stream, p['center_frequency_hz'])
        delay = self._arrival(stream)
        guard = self.fs / 2 - self.fs * 1e-12
        noise, bandwidth = _colored_noise(stream, self.length, min(requested, 2*(guard-abs(center))), self.fs, 'peak')
        frequency = center + bandwidth / 2 * noise
        signal = np.exp(2j*np.pi*np.cumsum(frequency)/self.fs)
        return np.roll(signal, delay), {'bandwidth_factor': bandwidth/self.radar['bandwidth_hz'],
            'interference_bandwidth_hz': bandwidth, 'center_frequency_hz': center, 'arrival_delay_s': delay/self.fs}

    def _AMA(self, stream, p):
        requested = _sample(stream, p['bandwidth_factor']) * self.radar['bandwidth_hz']
        center = _sample(stream, p['center_frequency_hz'])
        index = _sample(stream, p['modulation_index'])
        delay = self._arrival(stream)
        guard = self.fs / 2 - self.fs * 1e-12
        noise, bandwidth = _colored_noise(stream, self.length, min(requested, 2*(guard-abs(center))), self.fs, 'rms')
        signal = (1 + index * noise) * np.exp(2j*np.pi*center*self.time)
        return np.roll(signal, delay), {'bandwidth_factor': bandwidth/self.radar['bandwidth_hz'],
            'interference_bandwidth_hz': bandwidth, 'center_frequency_hz': center,
            'modulation_index': index, 'arrival_delay_s': delay/self.fs}

    def _SawFM(self, stream, p):
        bandwidth = min(_sample(stream, p['sweep_bandwidth_hz']), self.fs*(1-2e-12))
        period = _sample(stream, p['sweep_period_s'])
        delay = self._arrival(stream)
        frequency = -bandwidth/2 + bandwidth*np.mod(self.time, period)/period
        signal = np.exp(2j*np.pi*np.cumsum(frequency)/self.fs)
        return np.roll(signal, delay), {'sweep_bandwidth_hz': bandwidth, 'sweep_slope_hz_per_s': bandwidth/period,
            'sweep_period_s': period, 'arrival_delay_s': delay/self.fs}

    def _SinFM(self, stream, p):
        center = _sample(stream, p['center_frequency_hz'])
        upper = min(p['frequency_deviation_hz'][1], self.fs/2-self.fs*1e-12-abs(center))
        deviation = _sample(stream, [p['frequency_deviation_hz'][0], upper])
        frequency = _sample(stream, p['modulation_frequency_hz'])
        delay = self._arrival(stream)
        instantaneous = center + deviation*np.cos(2*np.pi*frequency*self.time)
        signal = np.exp(2j*np.pi*np.cumsum(instantaneous)/self.fs)
        return np.roll(signal, delay), {'center_frequency_hz': center, 'frequency_deviation_hz': deviation,
            'modulation_frequency_hz': frequency, 'modulation_period_s': 1/frequency, 'arrival_delay_s': delay/self.fs}

    def _MFT(self, stream, p):
        count = int(stream.integers(p['num_targets'][0], p['num_targets'][1]+1))
        arrival = self._arrival(stream)
        minimum, maximum = [_round_samples(v, self.fs) for v in p['target_delay_s']]
        relative = minimum + stream.choice(maximum-minimum+1, size=count, replace=False)
        amplitudes = stream.uniform(*p['target_relative_amplitude'], size=count)
        order = np.argsort(relative)
        delays, amplitudes = relative[order]+arrival, amplitudes[order]
        signal = sum(a*np.roll(self.reference, int(d)) for a, d in zip(amplitudes, delays))
        parameters = {'num_targets': count, 'mean_delay_s': np.mean(delays)/self.fs,
                      'mean_spacing_s': np.mean(np.diff(delays))/self.fs if count > 1 else np.nan}
        parameters.update({f'delay_{i+1:02d}_s': delays[i]/self.fs for i in range(count)})
        parameters.update({f'amplitude_{i+1:02d}': amplitudes[i] for i in range(count)})
        return signal, parameters

    def _ISRJ(self, stream, p):
        width = max(1, _round_samples(_sample(stream, p['sample_width_s']), self.fs))
        replicas = int(stream.integers(p['num_replicas'][0], p['num_replicas'][1]+1))
        arrival = self._arrival(stream)
        period = (1+replicas)*width
        sampled = self.reference * (np.arange(self.length) % period < width)
        signal = sampled.copy()
        for replica in range(1, replicas+1):
            shift = replica*width
            signal[shift:] += sampled[:-shift]
        return np.roll(signal, arrival), {'sample_width_s': width/self.fs, 'num_replicas': replicas,
            'sample_period_s': period/self.fs, 'arrival_delay_s': arrival/self.fs, 'mode_code': 0}

    def compose(self, active_codes, sample_seed, repetition, gaps, nominal_jnr_db=30., weak_gap_db=10.):
        """Compose all paired power views from the very same clean components."""
        seeds = [substream_seed(sample_seed, 1, REGISTRY_IDS[c]) for c in active_codes]
        generated = [self.component(c, seed) for c, seed in zip(active_codes, seeds)]
        clean = np.stack([item[0] for item in generated])
        noise_seed = substream_seed(sample_seed, 2)
        stream = rng(noise_seed)
        noise = np.sqrt(self.radar['noise_power']/2) * (stream.standard_normal(self.length) +
                                                       1j*stream.standard_normal(self.length))
        selected = (repetition-1) % len(active_codes)
        views = []
        for gap in gaps:
            jnr = np.full(len(active_codes), float(nominal_jnr_db))
            attenuation = np.zeros(len(active_codes), dtype=bool)
            if len(active_codes) > 1 and gap > 0:
                jnr[selected] -= gap
                attenuation[selected] = True
            components = clean * np.sqrt(self.radar['noise_power']) * 10**(jnr[:, None]/20)
            power = 10*np.log10(np.mean(np.abs(components)**2, axis=1))
            relative = power - power.max()
            weak = (len(active_codes) > 1) & (relative <= -weak_gap_db+1e-6)
            views.append({'iq_complex': components.sum(0)+noise, 'component_effective_JNR_dB': jnr,
                          'component_measured_power_dB': power, 'relative_power_dB': relative,
                          'weak_component_mask': weak, 'selected_attenuation_mask': attenuation})
        return views, seeds, noise_seed, [item[1] for item in generated]
