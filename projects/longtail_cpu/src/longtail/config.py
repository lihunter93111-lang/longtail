"""Resolve JSON inheritance once; a run saves the fully resolved configuration."""
from copy import deepcopy
from pathlib import Path
from .io import read_json, fingerprint

REPO_ROOT = Path(__file__).resolve().parents[2]


DEFAULTS = {'data': {'split_seed': 20260905,
          'per_combination': {'calibration': 16, 'check': 16, 'development_test': 64},
          'verify_hashes': True},
 'dataloader': {'batch_size': 32,
                'workers': 4,
                'pin_memory': True,
                'persistent_workers': False,
                'cache': 'memory',
                'preprocess': {'normalization': 'complex_rms',
                               'epsilon': 1e-12,
                               'remove_dc': False,
                               'crop_start': 0,
                               'crop_length': None,
                               'stride': 1,
                               'scale': 1.0}},
 'model': {'kwargs': {}},
 'train': {'device': 'auto',
           'sampler': 'uniform',
           'draws_per_epoch': None,
           'optimizer': 'adamw',
           'gradient_clip': 5.0,
           'amp': 'none',
           'selection_split': 'check',
           'selection_metric': 'macro_aupr',
           'loss': 'bce',
           'scheduler': 'none',
           'initialization': None,
           'sync_batchnorm': False,
           'cpu_threads': 2},
 'evaluation': {'batch_size': 64,
                'calibration': 'f1',
                'threshold_grid': [0.05, 0.95, 181],
                'ece_bins': 15},
 'run': {'output_dir': 'runs/baseline_seed{seed}_{signature}'}}

def merge(base, update):
    result = deepcopy(base)
    for key, value in update.items():
        if key == 'model' and isinstance(value, dict) and value.get('entrypoint', result.get(key, {}).get('entrypoint')) != result.get(key, {}).get('entrypoint'):
            result[key] = deepcopy(value)
            result[key].setdefault('kwargs', {})
        elif isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _load_raw(path, seen=()):
    path = Path(path).resolve()
    if path in seen:
        raise ValueError('Cyclic config inheritance')
    config = read_json(path)
    parent = config.pop('extends', None)
    if parent:
        config = merge(_load_raw(path.parent / parent, (*seen, path)), config)
    return config


def load_config(path):
    config = merge(DEFAULTS, _load_raw(path))
    if 'entrypoint' in config['model']:
        from .models.factory import resolve_model_config
        config['model'] = resolve_model_config(config['model'])
    return config


def validate(config):
    allowed = {
        'data': {'manifest', 'split_seed', 'per_combination', 'verify_hashes'},
        'dataloader': {'batch_size', 'workers', 'pin_memory', 'persistent_workers', 'cache', 'preprocess'},
        'model': {'entrypoint', 'kwargs'},
        'train': {'seed', 'epochs', 'device', 'optimizer', 'lr', 'weight_decay', 'gradient_clip', 'sampler', 'draws_per_epoch',
                  'amp', 'selection_split', 'selection_metric', 'loss', 'scheduler',
                  'initialization', 'sync_batchnorm', 'cpu_threads'},
        'evaluation': {'batch_size', 'calibration', 'threshold_grid', 'ece_bins'},
        'run': {'output_dir'},
    }
    if set(config) != set(allowed):
        raise ValueError(f'Configuration must have exactly these sections: {list(allowed)}')
    for section, keys in allowed.items():
        if set(config[section]) != keys:
            raise ValueError(f'Invalid {section} keys: expected {sorted(keys)}')
    if config['dataloader']['cache'] not in ('memory', 'none'):
        raise ValueError('Invalid dataloader cache')
    pre = config['dataloader']['preprocess']
    if set(pre) != {'normalization', 'epsilon', 'remove_dc', 'crop_start', 'crop_length', 'stride', 'scale'}:
        raise ValueError('Invalid preprocessing keys')
    if pre['normalization'] not in ('complex_rms', 'none') or pre['epsilon'] <= 0:
        raise ValueError('Invalid normalization')
    if pre['crop_start'] < 0 or pre['stride'] < 1 or (pre['crop_length'] is not None and pre['crop_length'] < 1):
        raise ValueError('Invalid crop or stride')
    if config['dataloader']['batch_size'] < 1 or config['dataloader']['workers'] < 0:
        raise ValueError('Invalid loader batch size or workers')
    train = config['train']
    if train['epochs'] < 1 or train['lr'] <= 0 or train['weight_decay'] < 0 or train['gradient_clip'] < 0:
        raise ValueError('Invalid training hyperparameters')
    if train['optimizer'] not in ('adamw', 'adam') or train['loss'] not in ('bce', 'weighted_bce', 'logit_adjustment', 'weak_positive_bce'):
        raise ValueError('Unsupported optimizer or loss')
    budget = train['draws_per_epoch']
    if budget is not None and (not isinstance(budget, int) or isinstance(budget, bool) or budget < 1 or train['sampler'] != 'uniform'):
        raise ValueError('A fixed draw budget requires positive integer draws and natural sampling')
    if train['sampler'] not in ('uniform','label_aware','combo_aware','mixed_combo'):
        raise ValueError('Unsupported training sampler')
    if train['device'] not in ('auto', 'cuda', 'cpu') or train['cpu_threads'] < 1:
        raise ValueError('Training device must be auto/cuda/cpu, with positive CPU threads')
    if train['amp'] not in ('none', 'float16', 'bfloat16'):
        raise ValueError('Invalid AMP mode')
    if train['selection_split'] not in ('check', 'calibration') or train['selection_metric'] not in ('macro_aupr', 'macro_f1', 'subset_accuracy', 'weak_macro_aupr'):
        raise ValueError('Invalid checkpoint selection protocol')
    if train['scheduler'] not in ('none', 'cosine'):
        raise ValueError('Unsupported scheduler')
    if config['evaluation']['calibration'] not in ('f1', 'f1_exact', 'none'):
        raise ValueError('Unsupported threshold calibration')
    lo, hi, steps = config['evaluation']['threshold_grid']
    if not 0 < lo < hi < 1 or steps < 2 or config['evaluation']['batch_size'] < 1 or config['evaluation']['ece_bins'] < 1:
        raise ValueError('Invalid evaluation settings')
    counts = config['data']['per_combination']
    if set(counts) != {'calibration', 'check', 'development_test'} or min(counts.values()) < 1:
        raise ValueError('Invalid development role counts')


def resolve_config(path):
    config = load_config(path)
    validate(config)
    config['run']['output_dir'] = config['run']['output_dir'].format(
        seed=config['train']['seed'], signature=fingerprint(config)[:10])
    for section, key in [('data', 'manifest'), ('run', 'output_dir')]:
        value = Path(config[section][key])
        config[section][key] = str((REPO_ROOT / value).resolve() if not value.is_absolute() else value)
    init = config['train']['initialization']
    if init is not None:
        if set(init) != {'checkpoint', 'sha256', 'prefix', 'exclude'}:
            raise ValueError('Invalid initialization keys')
        init['checkpoint'] = str((REPO_ROOT / init['checkpoint']).resolve())
    return config
