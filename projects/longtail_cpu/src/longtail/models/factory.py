"""A new model needs only an importable file and a JSON entrypoint."""
from importlib import import_module
from inspect import signature, Parameter
import json
from torch import nn


def resolve_model_config(config):
    """Freeze constructor defaults into run JSON without duplicating user config."""
    module, name = config['entrypoint'].split(':')
    constructor = getattr(import_module(module), name)
    defaults = {name: p.default for name, p in signature(constructor).parameters.items()
                if name != 'num_classes' and p.default is not Parameter.empty}
    kwargs = {**defaults, **config.get('kwargs', {})}
    # Convert tuples to JSON arrays so checkpoint/config comparisons are stable.
    return {**config, 'kwargs': json.loads(json.dumps(kwargs, allow_nan=False))}


def build_model(config, num_classes=None):
    module, name = config['entrypoint'].split(':')
    kwargs = dict(config.get('kwargs', {}))
    if num_classes is not None:
        if 'num_classes' in kwargs and kwargs['num_classes'] != num_classes:
            raise ValueError('Model class count does not match dataset')
        kwargs['num_classes'] = num_classes
    model = getattr(import_module(module), name)(**kwargs)
    if not isinstance(model, nn.Module):
        raise TypeError('The model factory must return torch.nn.Module')
    return model
