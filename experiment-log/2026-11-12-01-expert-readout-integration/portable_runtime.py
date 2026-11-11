"""Inference-only loader; no experiment storage paths or training helpers."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'source'))
from cjlt.models.backbone_adapter import build_backbone
from cjlt.models.segment_readout import ResidualReadout,segment_statistics
from cjlt.engine.feature_transform import FeatureTransform

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def normalize_iq(values):
    x=np.asarray(values,dtype=np.float32).copy()
    if x.ndim!=3 or x.shape[1:]!=(2,3000) or not np.isfinite(x).all():raise ValueError('Expected finite [N,2,3000] I/Q')
    x/=np.sqrt(np.mean(x[:,0]**2+x[:,1]**2,axis=1)+1e-12)[:,None,None]
    return torch.from_numpy(np.ascontiguousarray(x))

class Readout(torch.nn.Module):
    def __init__(self,state):
        super().__init__();self.variant=state['variant'];self.stages=state['stages']
        self.backbone=build_backbone('dfms_iq',6,width=state['width'],dropout=.2)
        self.heads=torch.nn.ModuleList();self.transforms=torch.nn.ModuleList()
        for i,stage in enumerate(self.stages):
            w=state['model_state_dict'];components=w[f'transforms.{i}.components'];hidden=w[f'heads.{i}.hidden.weight']
            self.transforms.append(FeatureTransform(components.shape[1],components.shape[0]))
            self.heads.append(ResidualReadout(components.shape[0],hidden.shape[0],len(stage['targets'])))
        self.load_state_dict(state['model_state_dict'])
    def branches(self,x):
        temporal=self.backbone.forward_temporal_features(x)
        projected=self.backbone.feature_projection(self.backbone.global_pool(temporal))
        z=self.backbone.forward_classifier(projected)
        if self.variant=='projected_global':features=projected
        else:features=segment_statistics(temporal,(1,) if self.variant=='global_stats' else (1,2,4))
        local=z.clone()
        for stage,head,transform in zip(self.stages,self.heads,self.transforms):local[:,stage['targets']]+=head(transform(features))
        return z,local.double()-z.double()
    def forward(self,x):
        z,delta=self.branches(x);return z.double()+delta

class PortableRuntime(torch.nn.Module):
    def __init__(self,bundle,record_path,verify=True):
        super().__init__();bundle=Path(bundle);self.record=json.loads((bundle/record_path).read_text(encoding='utf-8'));r=self.record
        def load(key):
            ref=r['models'][key];path=bundle/ref['path']
            if verify and digest(path)!=ref['sha256']:raise ValueError('Checkpoint hash mismatch')
            state=torch.load(path,map_location='cpu',weights_only=False)
            if 'stages' in state:return Readout(state)
            model=build_backbone('dfms_iq',6,width=32,dropout=.2);model.load_state_dict(state['model_state_dict']);return model
        self.affine=r['kind']=='single_affine';self.single=r['kind'] in ['single','single_affine']
        if self.single:self.main=load(r['group']);self.expert=None
        else:
            self.main=load('reference' if r['readout']=='none' else r['readout']);self.expert=load(r['expert'])
            self.register_buffer('weights',torch.tensor(r['weights'],dtype=torch.float64));self.register_buffer('biases',torch.tensor(r['biases'],dtype=torch.float64))
        self.register_buffer('thresholds',torch.tensor(r['thresholds'],dtype=torch.float64))
        if self.affine:
            self.register_buffer('affine_scales',torch.tensor(r['scales'],dtype=torch.float64))
            self.register_buffer('affine_biases',torch.tensor(r['biases'],dtype=torch.float64))
        self.eval().requires_grad_(False)
    def forward(self,x):
        if self.single:
            z=self.main(x).double();return z*self.affine_scales+self.affine_biases if self.affine else z
        if self.record['readout']=='none':z=self.main(x);delta=torch.zeros_like(z).double()
        else:z,delta=self.main.branches(x)
        return (torch.stack([z.double(),self.expert(x).double(),delta],-1)*self.weights).sum(-1)+self.biases

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,default=HERE);p.add_argument('--record',required=True)
    p.add_argument('--iq-npy',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cpu');a=p.parse_args()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    x=normalize_iq(np.load(a.iq_npy));model=PortableRuntime(a.bundle,a.record).to(a.device);outputs=[]
    with torch.inference_mode():
        for batch in x.split(64):outputs.append(model(batch.to(a.device)).cpu())
    logits=torch.cat(outputs);np.savez(a.output,logits=logits.numpy(),probabilities=logits.sigmoid().numpy(),labels=(logits.sigmoid()>=model.thresholds.cpu()).numpy())
