"""Inference from this repository plus hash-matched external weights and decision JSON.

No campaign config or E-drive source checkout is imported. Path mapping is an
explicit JSON dictionary from original absolute paths to relocated local paths.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import torch
import h5py
import numpy as np

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/'projects/CJ_Imbalance_0716'))
from cjlt.models.backbone_adapter import build_backbone
LABELS=['FMB','AMA','SawFM','SinFM','MFT','ISRJ']

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read_path_map(path):
    path=Path(path).resolve()
    return {k:str(Path(v) if Path(v).is_absolute() else path.parent/v) for k,v in read(path).items()}

def load_iq(path,indices):
    indices=np.asarray(indices,dtype=int);order=np.argsort(indices);inverse=np.argsort(order)
    with h5py.File(path,'r') as f:
        raw=np.asarray(f['iq_complex'][:,indices[order]])
        labels=np.asarray(f['known_label'][:,indices[order]],dtype=np.float32).T[inverse]
    x=np.stack([raw['real'].T,raw['imag'].T],1).astype(np.float32)[inverse]
    x/=np.sqrt(np.mean(x[:,0]**2+x[:,1]**2,axis=1)+1e-12)[:,None,None]
    assert x.shape==(len(indices),2,3000) and np.isfinite(x).all()
    return torch.from_numpy(np.ascontiguousarray(x)),labels

class PortableDecision(torch.nn.Module):
    def __init__(self,record,mapping=None):
        super().__init__();mapping=mapping or {}
        def backbone(key,digest_key):
            path=mapping.get(record[key],record[key]);assert sha(path)==record[digest_key],path
            model=build_backbone('dfms_iq',num_classes=6,width=32,dropout=.2)
            model.load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['model_state_dict'])
            return model.eval().requires_grad_(False)
        self.expert=backbone('model_path','model_sha256');self.reference=None
        self.register_buffer('thresholds',torch.tensor(record['thresholds'],dtype=torch.float64))
        if record['kind']=='fusion':
            self.reference=backbone('reference_path','reference_sha256')
            self.register_buffer('weights',torch.tensor(record['state']['weights'],dtype=torch.float64).reshape(6,2))
            self.register_buffer('biases',torch.tensor(record['state']['biases'],dtype=torch.float64))
    def forward(self,x):
        z=self.expert(x).double()
        if self.reference is not None:z=self.reference(x).double()*self.weights[:,0]+z*self.weights[:,1]+self.biases
        return z

def predict(decision,manifest,mapping=None,device='cpu',limit=None):
    mapping=mapping or {};record=read(decision);data=read(manifest)
    path=mapping.get(data['path'],data['path']);assert sha(path)==data['sha256']
    indices=data['indices'] if limit is None else data['indices'][:limit]
    x,y=load_iq(path,indices);model=PortableDecision(record,mapping).to(device).eval()
    out=[]
    with torch.inference_mode():
        for batch in x.split(64):out.append(model(batch.to(device)).cpu().numpy())
    logits=np.concatenate(out)
    return {'logits':logits,'thresholds':np.asarray(record['thresholds']),'labels':y}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--decision',required=True);p.add_argument('--manifest',required=True)
    p.add_argument('--path-map');p.add_argument('--device',default='cpu');p.add_argument('--output',required=True)
    a=p.parse_args();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    result=predict(a.decision,a.manifest,read_path_map(a.path_map) if a.path_map else {},a.device)
    target=Path(a.output);assert not target.exists(),'Choose a new output; original predictions are immutable'
    target.parent.mkdir(parents=True,exist_ok=True);np.savez(target,**result)
    print(json.dumps({'output':str(target),'sha256':sha(target),'labels':LABELS,'samples':len(result['labels'])}))
