"""Version-bound interfaces for the November 12 integration experiment."""
import hashlib,importlib.util,json,sys,time
from pathlib import Path
from datetime import datetime,timezone
import torch
import numpy as np
import h5py

HERE=Path(__file__).resolve().parent
CONFIG=json.loads((HERE/'protocol.json').read_text(encoding='utf-8'))
REPO=Path(CONFIG['repository']);ROOT=Path(CONFIG['storage'])
SOURCE=Path(CONFIG['historical_source']);PREVIOUS=Path(CONFIG['historical_readouts'])
BASELINES=Path(CONFIG['baseline_root']);MARGINAL=Path(CONFIG['marginal_root'])

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value

prior=module('nov11_interfaces',REPO/'experiment-log/2026-11-11-01-label-marginal-controls/common.py')
legacy=module('nov10_readout_interfaces',REPO/'experiment-log/2026-11-10-01-readout-integration/common.py')
base=prior.base;helper=prior.helper;LABELS=prior.LABELS
read=prior.read;write=prior.write;sha=prior.sha;metadata=prior.metadata;evaluate=prior.evaluate;objective_weights=prior.objective_weights
stamp=prior.stamp

def setup():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assert torch.cuda.is_available()

def checkpoint(split,kind,seed):
    expert_seed=seed+CONFIG['expert_seed_offset']
    if kind=='reference':return BASELINES/split/'w32'/f'M_{seed}/best_BCE.pth'
    if kind=='marginal':return MARGINAL/split/'models'/f'marginal_{expert_seed}/model.pth'
    if kind in ['natural','combo']:
        group='ordinary_expert' if kind=='natural' else 'weighted_expert'
        return PREVIOUS/split/'expert_training'/group/f'M_{expert_seed}/best_BCE.pth'
    assert kind in ['global','segment']
    return PREVIOUS/split/'readouts'/f"{CONFIG['readout_sources'][kind]}_{seed}/model.pth"

def load_model(split,kind,seed,device='cuda'):
    path=checkpoint(split,kind,seed)
    if kind in ['global','segment']:
        state=torch.load(path,map_location='cpu',weights_only=False)
        model=legacy.ReadoutBundle(state['variant'],state['stages'],state['width'])
        model.load_state_dict(state['model_state_dict'])
    else:
        model=base.build_backbone('dfms_iq',num_classes=6,width=32,dropout=.2)
        model.load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['model_state_dict'])
    return model.to(device).eval().requires_grad_(False)

def role_manifest(split,role):
    if role=='fusion_fit':return read(PREVIOUS/split/'fusion_fit/manifest.json')
    if role=='calibration':
        m=read(BASELINES/split/'data/manifest.json')
        return {'path':m['dev_path'],'sha256':m['dev_sha256'],'indices':m['calibration']}
    return read(ROOT/split/role/'manifest.json')

def code_hashes():
    paths=list(HERE.glob('*.py'))+list(HERE.glob('*.m'))+[HERE/'protocol.json',HERE/'GOAL_PROMPT.txt',Path(prior.__file__),Path(legacy.__file__),Path(base.__file__),Path(helper.__file__)]
    paths+=list((SOURCE/'projects/CJ_Imbalance_0716/cjlt').rglob('*.py'))
    return {str(p):sha(p) for p in sorted(set(paths))}
