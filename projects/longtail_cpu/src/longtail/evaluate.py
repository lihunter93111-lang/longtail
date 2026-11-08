"""Evaluate a checkpoint using only its saved configuration and frozen decision."""
import argparse
from copy import deepcopy
from pathlib import Path
import numpy as np
import torch
from .io import read_json, write_json, sha256
from .models.factory import build_model
from .data.splits import validate_split
from .data.loaders import make_dataset, make_loader
from .data.dataset import read_metadata
from .metrics import evaluate_metrics
from .weak_metrics import weak_metrics
from .engine.checkpoint import load_checkpoint
from .engine.inference import predict


def run(checkpoint, split_name='development_test', data_root=None, output=None, device_name='auto', view=None):
    checkpoint = Path(checkpoint)
    payload, config = load_checkpoint(checkpoint)
    decision = payload['decision']
    if decision is None:
        raise ValueError('Checkpoint is not calibrated; use the finalized best.pt, not last.pt')
    if decision['split_sha256'] != payload['split_sha256']:
        raise ValueError('Decision belongs to a different data split')
    torch.set_num_threads(config['train']['cpu_threads'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device_name == 'auto' else torch.device(device_name)
    manifest = read_json(checkpoint.parent / 'dataset.json')
    root = Path(data_root) if data_root is not None else Path(config['data']['manifest']).parent
    validate_split(payload['split'], manifest)
    source = root / manifest['files']['development']['path']
    if sha256(source) != manifest['files']['development']['sha256']:
        raise ValueError('Evaluation source hash mismatch')
    evaluated_source = manifest['files']['development']
    if view is not None:
        if view not in manifest.get('evaluation_views', {}):
            raise ValueError(f'Unknown power view: {view}')
        evaluated_source = manifest['evaluation_views'][view]
        relative = Path(evaluated_source['path'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Unsafe evaluation view path')
        alternate = root / relative
        if sha256(alternate) != evaluated_source['sha256']:
            raise ValueError('Evaluation view hash mismatch')
        original_y, original_ids, length = read_metadata(source)
        alternate_y, alternate_ids, alternate_length = read_metadata(alternate)
        if length != alternate_length or not np.array_equal(original_y, alternate_y) or not np.array_equal(original_ids, alternate_ids):
            raise ValueError('Evaluation views must have identical row labels and identities')
        manifest = deepcopy(manifest)
        manifest['files']['development'] = evaluated_source
    model = build_model(config['model'], num_classes=len(manifest['classes'])).to(device)
    model.load_state_dict(payload['model'], strict=True)
    dataset = make_dataset(manifest, root, payload['split'], split_name, config['dataloader'])
    try:
        records = predict(model, make_loader(dataset, config['dataloader'], batch_size=config['evaluation']['batch_size']), device)
    finally:
        dataset.close()
    common = {'classes': manifest['classes'], 'combination_counts': payload['split']['fit_combination_counts'],
              'ece_bins': config['evaluation']['ece_bins']}
    metrics = {'fixed_0.5': evaluate_metrics(records['targets'], records['logits'], **common),
               'calibrated': evaluate_metrics(records['targets'], records['logits'], thresholds=decision['thresholds'], **common)}
    for name, metric in metrics.items():
        metric['weak_component'] = weak_metrics(records['targets'], records['logits'], manifest['classes'],
            metric['thresholds'], records.get('weak_mask'), records.get('relative_power_db'), payload['split']['fit_label_counts'])
    output = Path(output) if output is not None else checkpoint.parent / 'evaluation' / split_name
    if view is not None and output == checkpoint.parent / 'evaluation' / split_name:
        output = output / view
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Refusing to overwrite evaluation: {output}')
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / 'predictions.npz', **records)
    write_json(output / 'metrics.json', {'checkpoint': str(checkpoint.resolve()), 'checkpoint_sha256': sha256(checkpoint),
               'config_sha256': payload['config_sha256'], 'split_sha256': payload['split_sha256'],
               'split_role': split_name, 'power_view': view, 'evaluated_source': evaluated_source,
               'decision': decision, 'metrics': metrics, 'locked_test_used': False})
    print({'split': split_name, 'samples': len(records['targets']),
           'macro_f1': metrics['calibrated']['macro_f1'], 'subset_accuracy': metrics['calibrated']['subset_accuracy']})
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--split', choices=['check', 'development_test', 'calibration'], default='development_test')
    parser.add_argument('--data-root', type=Path, help='Relocate data without changing model or preprocessing')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--view', help='Paired power view from the saved dataset manifest, e.g. gap_24')
    args = parser.parse_args()
    run(args.checkpoint, args.split, args.data_root, args.output, args.device, args.view)


if __name__ == '__main__':
    main()
