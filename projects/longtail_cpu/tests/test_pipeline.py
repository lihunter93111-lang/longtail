import copy
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
import torch
from longtail.config import resolve_config, load_config
from longtail.engine.checkpoint import load_checkpoint
from longtail.io import read_json, write_json
from longtail.train import run
from longtail.evaluate import run as evaluate
from longtail.models.factory import build_model
from longtail.data.loaders import make_dataset, make_loader
from longtail.engine.inference import predict


def test_train_checkpoint_evaluate_resume_and_config_guard(experiment):
    root, config, _ = experiment
    run(config)
    checkpoint = root / 'run/best.pt'
    payload, saved = load_checkpoint(checkpoint)
    assert saved == config and payload['decision']['fit_role'] == 'calibration'
    assert payload['history'][0]['selection_split'] == 'check'
    assert read_json(root / 'run/status.json')['status'] == 'completed'
    metrics = evaluate(checkpoint, output=root / 'evaluation', device_name='cpu')
    records = np.load(root / 'evaluation/predictions.npz')
    model = build_model(config['model']);model.load_state_dict(payload['model']);model.eval()
    dataset = make_dataset(read_json(root / 'dataset.json'), root, payload['split'], 'development_test', config['dataloader'])
    direct = predict(model, make_loader(dataset, config['dataloader'], batch_size=64), torch.device('cpu'))
    np.testing.assert_array_equal(records['indices'], direct['indices'])
    np.testing.assert_allclose(records['logits'], direct['logits'])
    assert metrics['calibrated']['samples'] == 63
    before = (root / 'run/status.json').read_bytes()
    with pytest.raises(FileExistsError):
        run(config)
    assert (root / 'run/status.json').read_bytes() == before
    run(config, resume=root / 'run/last.pt')
    new_payload, _ = load_checkpoint(checkpoint)
    assert new_payload['decision'] == payload['decision']
    changed = copy.deepcopy(config);changed['dataloader']['preprocess']['scale'] = 2
    write_json(root / 'run/config.json', changed)
    with pytest.raises(ValueError, match='config.json'):
        load_checkpoint(checkpoint)


def test_json_inheritance_and_unknown_parameters_are_checked(tmp_path):
    config = load_config('configs/baseline.json')
    config['train']['typo_lr'] = 1
    write_json(tmp_path / 'bad.json', config)
    with pytest.raises(ValueError, match='train'):
        resolve_config(tmp_path / 'bad.json')
    config = load_config('configs/baseline.json')
    config['model']['unexpected'] = 1
    write_json(tmp_path / 'bad-model.json', config)
    with pytest.raises(ValueError, match='model'):
        resolve_config(tmp_path / 'bad-model.json')
    write_json(tmp_path / 'cycle.json', {'extends': 'cycle.json'})
    with pytest.raises(ValueError, match='Cyclic'):
        load_config(tmp_path / 'cycle.json')


@pytest.mark.parametrize('architecture', ['dfms', 'transformer'])
def test_two_process_ddp_and_reload(experiment, architecture):
    root, config, _ = experiment
    config['dataloader']['batch_size'] = 16
    config['train']['epochs'] = 2
    if architecture == 'transformer':
        config['model'] = {'entrypoint': 'longtail.models.transformer:build_model',
                           'kwargs': {'dimension': 16, 'heads': 2, 'layers': 1, 'patch_size': 8}}
    write_json(root / 'config.json', config)
    result = subprocess.run([sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=2',
                             '-m', 'longtail.train', '--config', str(root / 'config.json')],
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    status = read_json(root / 'run/status.json')
    assert status['world_size'] == 2 and status['status'] == 'completed'
    payload, _ = load_checkpoint(root / 'run/last.pt')
    assert len(payload['random_states']) == 2
    assert all(row['samples_seen'] == 252 for row in payload['history'])
    result = evaluate(root / 'run/best.pt', output=root / 'ddp_evaluation', device_name='cpu')
    assert result['calibrated']['samples'] == 63
