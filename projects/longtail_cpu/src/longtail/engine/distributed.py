"""torchrun DDP setup, rank coordination and random states."""
import os
import random
from datetime import timedelta
import numpy as np
import torch
import torch.distributed as dist


def setup(device_name, cpu_threads):
    torch.set_num_threads(cpu_threads)
    world = int(os.environ.get('WORLD_SIZE', '1'))
    rank = int(os.environ.get('RANK', '0'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    if device_name == 'auto':
        device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(f'cuda:{local_rank}' if device_name == 'cuda' else device_name)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group('nccl' if device.type == 'cuda' else 'gloo', timeout=timedelta(minutes=10))
    return rank, world, device


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def barrier():
    if dist.is_initialized():
        dist.barrier(device_ids=[torch.cuda.current_device()] if dist.get_backend() == 'nccl' else None)


def sync_buffers(model):
    if dist.is_initialized():
        for buffer in model.buffers():
            dist.broadcast(buffer, src=0)


def gather_random_states(device):
    state = {'cpu': torch.get_rng_state(), 'python': random.getstate()}
    if device.type == 'cuda':
        state['cuda'] = torch.cuda.get_rng_state(device)
    if not dist.is_initialized():
        return [state]
    result = [None] * dist.get_world_size() if dist.get_rank() == 0 else None
    dist.gather_object(state, result, dst=0)
    return result


def restore_random_state(state, device):
    torch.set_rng_state(state['cpu'].cpu())
    random.setstate(state['python'])
    if device.type == 'cuda':
        torch.cuda.set_rng_state(state['cuda'].cpu(), device)
