from copy import deepcopy
import numpy as np
import pytest
import h5py
from longtail.data.signals import Simulator, DEFAULT_PHYSICS, REGISTRY_IDS, rng, substream_seed
from longtail.data.generate_files import generate
from longtail.data.generation import plan
from longtail.data.joint import prepare_joint
from longtail.data.dataset import IQDataset, read_metadata
from longtail.config import load_config
from longtail.io import write_json, read_json, sha256


@pytest.mark.parametrize('code', list(REGISTRY_IDS))
def test_six_components_are_finite_unit_power_and_reproducible(code):
    simulator = Simulator()
    signal, parameters = simulator.component(code, 105)
    repeated, repeat_parameters = simulator.component(code, 105)
    assert signal.shape == (3000,) and np.iscomplexobj(signal)
    assert np.isfinite(signal).all()
    assert np.mean(np.abs(signal)**2) == pytest.approx(1, abs=1e-12)
    np.testing.assert_array_equal(signal, repeated)
    assert parameters == repeat_parameters
    different, _ = simulator.component(code, 106)
    assert not np.array_equal(signal, different)


def test_fm_phase_laws_and_single_target_reference():
    settings = deepcopy(DEFAULT_PHYSICS)
    settings['arrival_delay_s'] = [0, 0]
    settings['SinFM'].update(center_frequency_hz=[1e6, 1e6], frequency_deviation_hz=[0, 0])
    settings['SawFM'].update(sweep_bandwidth_hz=[2e6, 2e6], sweep_period_s=[50e-6, 50e-6])
    settings['MFT'].update(num_targets=[1, 1], target_delay_s=[30e-6, 30e-6],
                           target_relative_amplitude=[1, 1])
    simulator = Simulator(settings)
    signal, _ = simulator.component('SinFM', 8)
    expected = np.exp(2j*np.pi*1e6*np.arange(1, 3001)/1e7)
    np.testing.assert_allclose(signal, expected, atol=3e-12)
    saw, _ = simulator.component('SawFM', 8)
    frequency = -1e6 + 2e6*(simulator.time % 50e-6)/50e-6
    expected = np.exp(2j*np.pi*np.cumsum(frequency)/1e7)
    np.testing.assert_allclose(saw, expected, atol=1e-12)
    mft, parameters = simulator.component('MFT', 8)
    expected = np.roll(simulator.reference, 300)*np.sqrt(3)
    np.testing.assert_allclose(mft, expected, atol=1e-12)
    assert parameters['delay_01_s'] == pytest.approx(30e-6)
    assert np.isnan(parameters['mean_spacing_s'])


def test_isrj_replica_uses_zero_delay_then_common_arrival():
    settings = deepcopy(DEFAULT_PHYSICS)
    settings['arrival_delay_s'] = [10e-6, 10e-6]
    settings['ISRJ'] = {'sample_width_s': [5e-6, 5e-6], 'num_replicas': [1, 1]}
    simulator = Simulator(settings)
    signal, parameters = simulator.component('ISRJ', 1)
    before_arrival = np.roll(signal, -100)
    # First two forwarding segments carry the exact same phase history.
    np.testing.assert_allclose(before_arrival[:50], before_arrival[50:100], atol=1e-12)
    assert not np.any(before_arrival[1000:])
    assert parameters['sample_period_s'] == pytest.approx(10e-6)


def test_paired_views_have_exact_component_power_noise_and_rotating_weak_role():
    simulator = Simulator()
    codes = list(REGISTRY_IDS)
    for repetition in range(1, 7):
        views, seeds, noise_seed, _ = simulator.compose(codes, 3800010000+repetition, repetition, [0, 6, 12, 24])
        clean = np.stack([simulator.component(c, s)[0] for c, s in zip(codes, seeds)])
        noise_rng = rng(noise_seed)
        noise = (noise_rng.standard_normal(3000)+1j*noise_rng.standard_normal(3000))/np.sqrt(2)
        assert np.mean(np.abs(noise)**2) == pytest.approx(1, abs=0.08)
        for gap, view in zip([0, 6, 12, 24], views):
            selected = repetition-1
            expected_power = np.full(6, 30.)
            expected_power[selected] -= gap
            np.testing.assert_allclose(view['component_measured_power_dB'], expected_power, atol=1e-12)
            gains = 10**(expected_power/20)
            np.testing.assert_allclose(view['iq_complex'], gains@clean+noise, atol=1e-12)
            assert view['weak_component_mask'].sum() == (gap >= 10)
            if gap >= 10:
                assert view['weak_component_mask'][selected]
    singles, _, _, _ = simulator.compose(['AMA'], 100, 1, [0, 24])
    np.testing.assert_array_equal(singles[0]['iq_complex'], singles[1]['iq_complex'])
    assert not singles[1]['weak_component_mask'].any()


def test_python_generation_to_existing_loader_and_manifest(tmp_path, monkeypatch):
    from longtail.data import generation
    from longtail.data.manage import CLASSES, verify
    monkeypatch.setattr(generation, 'REPO_ROOT', tmp_path)
    write_json(tmp_path / 'counts.json', {'classes': CLASSES, 'local_root': 'old', 'files': {
        'train': {'combination_counts': [2]*63}, 'development': {'combination_counts': [3]*63}}})
    write_json(tmp_path / 'generator.json', {'base_seed': 3800000000, 'label_count_source': 'counts.json',
        'gaps_db': [0, 12], 'nominal_jnr_db': 30, 'weak_gap_db': 10, 'output_dir': 'raw'})
    settings = plan(tmp_path / 'generator.json')
    receipt = generate(settings)
    assert receipt['status'] == 'completed' and len(receipt['files']) == 4
    assert read_json(tmp_path / 'raw/generation_complete.json') == receipt
    zero = tmp_path / 'raw/label_if3/gap_0/development.h5'
    weak = tmp_path / 'raw/label_if3/gap_12/development.h5'
    y, identity, length = read_metadata(zero)
    np.testing.assert_array_equal(identity, read_metadata(weak)[1])
    assert length == 3000 and y.shape == (189, 6)
    with h5py.File(zero) as a, h5py.File(weak) as b:
        np.testing.assert_array_equal(a['component_seeds'], b['component_seeds'])
        np.testing.assert_array_equal(a['parameter_values'], b['parameter_values'])
        assert a.attrs['generator'] == 'numpy-pcg64-v1'
    manifest = prepare_joint(tmp_path / 'raw', tmp_path / 'joint', train_gaps=(0, 12))
    assert verify(tmp_path / 'joint')['verified']
    cfg = load_config('configs/baseline.json')
    dataset = IQDataset(tmp_path / 'joint' / manifest['files']['train']['path'], [0, 15], cfg['dataloader']['preprocess'])
    assert dataset[0]['signal'].shape == (2, 3000)
    assert dataset[1]['weak_mask'].shape == (6,)
    dataset.close()
    for relative, record in receipt['files'].items():
        assert record['sha256'] == sha256(tmp_path / 'raw' / relative)
    with pytest.raises(FileExistsError):
        generate(settings)
