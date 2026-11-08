"""JSON-driven single-device / torchrun DDP training entrypoint."""
import argparse
import json
from pathlib import Path
import time
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from .config import resolve_config
from .io import read_json, write_json, fingerprint
from .models.factory import build_model
from .data.splits import build_split
from .data.loaders import make_dataset, make_loader
from .data.sampling import EpochWeightedSampler, EpochMixedComboSampler, EpochBudgetSampler
from .metrics import evaluate_metrics, fit_thresholds
from .weak_metrics import weak_metrics
from .engine import distributed as distributed
from .engine.checkpoint import initialize, load_checkpoint, save_checkpoint
from .engine.inference import predict
from .engine.trainer import build_optimizer, build_loss, train_epoch


def run(config, resume=None):
    rank, world, device = distributed.setup(config['train']['device'], config['train']['cpu_threads'])
    output = Path(config['run']['output_dir'])
    owns_output = False
    try:
        distributed.seed_all(config['train']['seed'])
        manifest_path = Path(config['data']['manifest'])
        manifest = read_json(manifest_path)
        if rank == 0:
            if resume is None and output.exists() and any(output.iterdir()):
                raise FileExistsError(f'Refusing to overwrite a run: {output}')
            output.mkdir(parents=True, exist_ok=True)
            owns_output = True
            split = build_split(manifest, manifest_path.parent, config['data'])
            if resume:
                saved, previous = load_checkpoint(resume)
                if previous != config or saved['world_size'] != world or saved['split'] != split:
                    raise ValueError('Resume requires identical config, world size and data split')
            print(json.dumps({'run_dir': str(output), 'world_size': world}), flush=True)
            write_json(output / 'config.json', config)
            write_json(output / 'split.json', split)
            write_json(output / 'dataset.json', manifest)
            write_json(output / 'status.json', {'status': 'running', 'world_size': world, 'device': str(device),
                                               'torch': str(torch.__version__), 'cuda': torch.version.cuda})
        distributed.barrier()
        split = read_json(output / 'split.json')
        model = build_model(config['model'], num_classes=len(manifest['classes'])).to(device)
        initialization = initialize(model, config['train']['initialization']) if resume is None else None
        if config['train']['sync_batchnorm']:
            if world == 1 or device.type != 'cuda':
                raise ValueError('SyncBatchNorm requires multi-GPU DDP')
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        optimizer = build_optimizer(model, config['train'])
        scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['train']['epochs'])
                     if config['train']['scheduler'] == 'cosine' else None)
        if device.type != 'cuda' and config['train']['amp'] == 'float16':
            raise ValueError('float16 AMP requires CUDA; CPU supports bfloat16 or none')
        scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda' and config['train']['amp'] == 'float16')
        start_epoch, best, history = 1, -float('inf'), []
        if resume:
            saved, _ = load_checkpoint(resume)
            model.load_state_dict(saved['model'], strict=True)
            optimizer.load_state_dict(saved['optimizer'])
            scaler.load_state_dict(saved['scaler'])
            if scheduler:
                scheduler.load_state_dict(saved['scheduler'])
            start_epoch = saved['epoch'] + 1
            best, history = saved['best_score'], saved['history']
            initialization = saved['initialization']
        raw_model = model
        if world > 1:
            model = DistributedDataParallel(model, device_ids=[device.index] if device.type == 'cuda' else None)
        if resume:
            distributed.restore_random_state(saved['random_states'][rank], device)
        else:
            distributed.seed_all(config['train']['seed'] + rank)
        loader_config = config['dataloader']
        global_batch = loader_config['batch_size']
        if global_batch % world:
            raise ValueError('Global batch_size must be divisible by WORLD_SIZE')
        fit = make_dataset(manifest, manifest_path.parent, split, 'fit', loader_config)
        sampler = DistributedSampler(fit, num_replicas=world, rank=rank, shuffle=True,
                                     seed=config['train']['seed'], drop_last=False) if world > 1 else None
        sampling_mode=config['train'].get('sampler','uniform')
        if sampling_mode!='uniform':
            if world!=1:
                raise ValueError('Label/combo sampling currently requires independent world_size=1 jobs')
            sampler=(EpochMixedComboSampler(fit.labels[fit.indices],config['train']['seed'])
                     if sampling_mode=='mixed_combo' else
                     EpochWeightedSampler(fit.labels[fit.indices],sampling_mode,config['train']['seed']))
        draw_budget = config['train'].get('draws_per_epoch')
        if draw_budget is not None:
            if world != 1 or sampling_mode != 'uniform':
                raise ValueError('Fixed draw budgets require single-device natural sampling')
            sampler = EpochBudgetSampler(fit.labels[fit.indices], config['train']['seed'], draw_budget)
        weak_counts = None
        if config['train']['loss'] == 'weak_positive_bce':
            if fit.power_truth is None:
                raise ValueError('Weak-positive BCE requires fit component truth')
            weak_counts = (fit.power_truth['weak_mask'][fit.indices] & fit.labels[fit.indices].astype(bool)).sum(0)
        criterion = build_loss(config['train'], split['fit_label_counts'], len(fit), device, weak_counts=weak_counts)
        if rank == 0 and weak_counts is not None:
            write_json(output / 'loss_audit.json', {'fit_weak_counts': weak_counts.astype(int).tolist(),
                'selected_labels': ['SawFM', 'MFT', 'ISRJ'], 'weak_positive_weight': 1.5,
                'fit_mean_weight_normalizers': criterion.normalizer.cpu().tolist(), 'truth_role': 'fit'})
        selection = None
        if rank == 0:
            selection = make_dataset(manifest, manifest_path.parent, split, config['train']['selection_split'], loader_config)
        started = time.time()
        loader = make_loader(fit, loader_config, training=True, sampler=sampler,
                             batch_size=global_batch // world, seed=config['train']['seed'])
        for epoch in range(start_epoch, config['train']['epochs'] + 1):
            epoch_started = time.perf_counter()
            if sampler is not None:
                sampler.set_epoch(epoch)
            loader.generator.manual_seed(config['train']['seed'] + epoch)
            loss, sampled = train_epoch(model, loader, optimizer, criterion, device, config['train'], scaler)
            train_seconds = time.perf_counter() - epoch_started
            if scheduler:
                scheduler.step()
            distributed.sync_buffers(raw_model)
            random_states = distributed.gather_random_states(device)
            if rank == 0:
                selection_started = time.perf_counter()
                val_loader = make_loader(selection, loader_config, batch_size=config['evaluation']['batch_size'])
                records = predict(raw_model, val_loader, device)
                metrics = evaluate_metrics(records['targets'], records['logits'], split['classes'])
                metrics['weak_component'] = weak_metrics(records['targets'], records['logits'], split['classes'],
                    metrics['thresholds'], records.get('weak_mask'), records.get('relative_power_db'), split['fit_label_counts'])
                name = config['train']['selection_metric']
                score = metrics['weak_component'].get(name) if name == 'weak_macro_aupr' else metrics[name]
                if score is None:
                    raise ValueError('Checkpoint selection metric is undefined; weak selection requires verified weak positives in check')
                improved = score > best
                best = max(best, score)
                row = {'epoch': epoch, 'loss': loss, 'samples_seen': sampled,
                       'sampler_padding': sampled - (draw_budget if draw_budget is not None else len(fit)),
                       'fit_pool_size': len(fit), 'expected_draws_per_epoch': draw_budget if draw_budget is not None else len(fit),
                       'lr': optimizer.param_groups[0]['lr'],
                       'train_seconds': train_seconds,
                       'selection_seconds': time.perf_counter() - selection_started,
                       'epoch_seconds': time.perf_counter() - epoch_started,
                       'selection_split': config['train']['selection_split'], 'validation': metrics}
                if isinstance(sampler,EpochWeightedSampler):
                    row['sampling']=sampler.audit()
                history.append(row)
                payload = {'schema_version': 1, 'model': raw_model.state_dict(), 'optimizer': optimizer.state_dict(),
                           'scaler': scaler.state_dict(), 'scheduler': scheduler.state_dict() if scheduler else None,
                           'epoch': epoch, 'best_score': best, 'history': history, 'random_states': random_states,
                           'world_size': world, 'config': config, 'config_sha256': fingerprint(config),
                           'split': split, 'split_sha256': split['sha256'], 'initialization': initialization,
                           'decision': None}
                save_checkpoint(output / 'last.pt', payload)
                if improved:
                    save_checkpoint(output / 'best.pt', payload)
                write_json(output / 'history.json', history)
                print(json.dumps({'epoch': epoch, 'loss': loss, 'selection_score': score, 'samples': sampled,
                                  'epoch_seconds': row['epoch_seconds']}), flush=True)
            distributed.barrier()
        if rank == 0:
            best_payload, _ = load_checkpoint(output / 'best.pt')
            raw_model.load_state_dict(best_payload['model'], strict=True)
            calibration = make_dataset(manifest, manifest_path.parent, split, 'calibration', loader_config)
            records = predict(raw_model, make_loader(calibration, loader_config,
                              batch_size=config['evaluation']['batch_size']), device)
            calibration_method=config['evaluation']['calibration']
            thresholds = (fit_thresholds(records['targets'], records['logits'],
                              None if calibration_method=='f1_exact' else config['evaluation']['threshold_grid'])
                          if calibration_method in ('f1','f1_exact') else [0.5] * len(split['classes']))
            best_payload['decision'] = {'method': config['evaluation']['calibration'], 'thresholds': thresholds,
                                        'fit_role': 'calibration', 'split_sha256': split['sha256']}
            save_checkpoint(output / 'best.pt', best_payload)
            write_json(output / 'decision.json', best_payload['decision'])
            write_json(output / 'status.json', {'status': 'completed', 'world_size': world, 'device': str(device),
                       'parameters': sum(p.numel() for p in raw_model.parameters()), 'best_epoch': best_payload['epoch'],
                       'best_selection_score': best, 'elapsed_seconds': time.time() - started,
                       'initialization': initialization, 'torch': str(torch.__version__), 'cuda': torch.version.cuda,
                       'checkpoint_selection_role': config['train']['selection_split'],
                       'threshold_fit_role': 'calibration', 'locked_test_used': False})
            calibration.close()
        distributed.barrier()
        fit.close()
        if selection is not None:
            selection.close()
    except Exception as error:
        if rank == 0 and owns_output:
            write_json(output / 'failure.json', {'type': type(error).__name__, 'error': str(error)})
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args()
    run(resolve_config(args.config), args.resume)


if __name__ == '__main__':
    main()
