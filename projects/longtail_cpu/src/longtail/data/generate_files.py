"""Stream NumPy-generated waveforms and per-label truth into HDF5 files."""
from contextlib import ExitStack
import json
from pathlib import Path
import h5py
import numpy as np
from ..io import write_json, sha256
from .signals import Simulator, PARAMETERS
from .dataset import read_metadata
from .power_truth import read_power_truth


def _create(path, n, simulator, plan):
    path.parent.mkdir(parents=True, exist_ok=True)
    f = h5py.File(path, 'x')
    columns = min(64, n)
    f.create_dataset('iq_complex', (simulator.length, n), dtype=np.complex64,
                     chunks=(simulator.length, columns), compression='lzf')
    for name, dtype, fill in [('label_selected', np.uint8, 0), ('component_seeds', np.float64, np.nan),
                             ('component_effective_JNR_dB', np.float64, np.nan),
                             ('component_measured_power_dB', np.float64, np.nan),
                             ('relative_power_dB', np.float64, np.nan),
                             ('weak_component_mask', np.uint8, 0), ('selected_attenuation_mask', np.uint8, 0)]:
        f.create_dataset(name, (6, n), dtype=dtype, fillvalue=fill)
    for key in ('sample_seed', 'receiver_noise_seed'):
        f.create_dataset(key, (1, n), dtype=np.uint32)
    names = [f'{code}.{name}' for code, fields in PARAMETERS.items() for name in fields]
    f.create_dataset('parameter_names', data=np.asarray(names, dtype=h5py.string_dtype('utf-8')))
    f.create_dataset('parameter_values', (len(names), n), dtype=np.float64, fillvalue=np.nan)
    f.attrs.update(power_truth_schema='per-label-v1', weak_gap_db=plan['weak_gap_db'],
                   noise_power_db=10*np.log10(simulator.radar['noise_power']),
                   classes_json=json.dumps(plan['classes']), generator='numpy-pcg64-v1',
                   physics_json=json.dumps(simulator.physics), generation_plan_sha256=plan['sha256'])
    return f, names


def generate(plan):
    root = Path(plan['output_dir'])
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f'Use an empty generation directory: {root}')
    simulator = Simulator(plan['physics'])
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / 'generation_plan.json', plan)
    records = {}
    for profile in plan['profiles']:
        for role, counts, offset in [('train', profile['counts_train'], 0),
                                     ('development', plan['counts_development'], 10000000)]:
            codes = np.repeat(np.arange(1, 64), counts)
            n = len(codes)
            repetition = np.concatenate([np.arange(1, count+1) for count in counts])
            files = []
            with ExitStack() as stack:
                for gap in plan['gaps_db']:
                    path = root / profile['name'] / f'gap_{gap:g}' / f'{role}.h5'
                    f, names = _create(path.with_suffix('.partial.h5'), n, simulator, plan)
                    stack.enter_context(f)
                    files.append((gap, f, path))
                name_indices = {name: i for i, name in enumerate(names)}
                for start in range(0, n, 64):
                    stop = min(start+64, n)
                    # Buffer one chunk per view; components are generated only once per identity.
                    buffers = [{key: np.asarray(f[key][:, start:stop]) for key in
                                ['label_selected', 'component_seeds', 'sample_seed', 'receiver_noise_seed',
                                 'component_effective_JNR_dB', 'component_measured_power_dB', 'relative_power_dB',
                                 'weak_component_mask', 'selected_attenuation_mask', 'parameter_values']} for _, f, _ in files]
                    waves = [np.empty((simulator.length, stop-start), dtype=np.complex64) for _ in files]
                    for index in range(start, stop):
                        active = np.flatnonzero((int(codes[index]) >> np.arange(6)) & 1)
                        active_codes = [plan['classes'][i] for i in active]
                        sample_seed = plan['base_seed'] + offset + int(codes[index])*10000 + int(repetition[index])
                        views, seeds, noise_seed, parameters = simulator.compose(active_codes, sample_seed,
                            int(repetition[index]), plan['gaps_db'], plan['nominal_jnr_db'], plan['weak_gap_db'])
                        for view_index, (view, buffer) in enumerate(zip(views, buffers)):
                            column = index-start
                            waves[view_index][:, column] = view['iq_complex']
                            buffer['label_selected'][active, column] = 1
                            buffer['component_seeds'][active, column] = seeds
                            buffer['sample_seed'][0, column] = sample_seed
                            buffer['receiver_noise_seed'][0, column] = noise_seed
                            for key in ['component_effective_JNR_dB', 'component_measured_power_dB',
                                        'relative_power_dB', 'weak_component_mask', 'selected_attenuation_mask']:
                                buffer[key][active, column] = view[key]
                            for code, values in zip(active_codes, parameters):
                                for name, value in values.items():
                                    buffer['parameter_values'][name_indices[f'{code}.{name}'], column] = value
                    for (_, f, _), wave, buffer in zip(files, waves, buffers):
                        f['iq_complex'][:, start:stop] = wave
                        for key, values in buffer.items():
                            f[key][:, start:stop] = values
                    if start % 1024 == 0 or stop == n:
                        print(json.dumps({'profile': profile['name'], 'role': role, 'generated_identities': stop,
                                          'total': n, 'power_views': len(files)}), flush=True)
            for gap, _, path in files:
                path.with_suffix('.partial.h5').replace(path)
                y, _, _ = read_metadata(path)
                truth = read_power_truth(path, y)
                records[str(path.relative_to(root))] = {'samples': len(y), 'sha256': sha256(path),
                    'bytes': path.stat().st_size, 'weak_positive_count': int(truth['weak_mask'].sum())}
    receipt = {'status': 'completed', 'generator': 'numpy-pcg64-v1', 'plan_sha256': plan['sha256'],
               'power_truth_schema': 'per-label-v1', 'files': records}
    write_json(root / 'generation_complete.json', receipt)
    return receipt
