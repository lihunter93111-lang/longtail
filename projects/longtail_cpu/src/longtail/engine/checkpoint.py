"""Checkpoints are bound to their resolved JSON configuration and split."""
import os
from pathlib import Path
import torch
from ..io import fingerprint, read_json, sha256


def save_checkpoint(path, payload):
    path = Path(path)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(path):
    path = Path(path).resolve()
    payload = torch.load(path, map_location='cpu', weights_only=True)
    config = read_json(path.parent / 'config.json')
    if payload['config_sha256'] != fingerprint(config) or payload['config'] != config:
        raise ValueError('Checkpoint and config.json do not match')
    if payload['split_sha256'] != fingerprint({k: v for k, v in payload['split'].items() if k != 'sha256'}):
        raise ValueError('Checkpoint split fingerprint mismatch')
    return payload, config


def initialize(model, config):
    if config is None:
        return {'mode': 'random'}
    path = Path(config['checkpoint'])
    actual = sha256(path)
    if config['sha256'] and actual != config['sha256']:
        raise ValueError('Initialization checkpoint SHA-256 mismatch')
    payload = torch.load(path, map_location='cpu', weights_only=True)
    source = payload.get('model_state_dict', payload.get('model', payload.get('state_dict', payload)))
    prefix = config['prefix']
    state = {k[len(prefix):]: v for k, v in source.items() if k.startswith(prefix)}
    state = {k: v for k, v in state.items() if not any(k.startswith(p) for p in config['exclude'])}
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [k for k in missing if not any(k.startswith(p) for p in config['exclude'])]
    if missing or unexpected:
        raise ValueError(f'Initialization state mismatch: missing={missing}, unexpected={unexpected}')
    return {'mode': 'checkpoint', 'path': str(path), 'sha256': actual}
