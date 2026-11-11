"""Two-backbone normal forward, sharing the main backbone with its local head."""
import argparse
from common import *

class FactorialRuntime(torch.nn.Module):
    def __init__(self,split,record):
        super().__init__();self.record=record;seed=record['seed'];self.affine=record['kind']=='single_affine';self.single=record['kind'] in ['single','single_affine']
        for m in record['models'].values():assert sha(m['path'])==m['sha256']
        if self.single:self.main=load_model(split,record['group'],seed,'cpu');self.expert=None
        else:
            self.main=load_model(split,'reference' if record['readout']=='none' else record['readout'],seed,'cpu')
            self.expert=load_model(split,record['expert'],seed,'cpu')
            self.register_buffer('weights',torch.tensor(record['weights'],dtype=torch.float64))
            self.register_buffer('biases',torch.tensor(record['biases'],dtype=torch.float64))
        self.register_buffer('thresholds',torch.tensor(record['thresholds'],dtype=torch.float64))
        if self.affine:
            self.register_buffer('affine_scales',torch.tensor(record['scales'],dtype=torch.float64))
            self.register_buffer('affine_biases',torch.tensor(record['biases'],dtype=torch.float64))
    def forward(self,x):
        if self.single:
            z=self.main(x).double();return z*self.affine_scales+self.affine_biases if self.affine else z
        if self.record['readout']=='none':reference=self.main(x);delta=torch.zeros_like(reference).double()
        else:
            reference,features=self.main.feature_branches(x);local=reference.clone()
            for stage,head,transform in zip(self.main.stages,self.main.heads,self.main.transforms):
                local[:,stage['targets']]+=head(transform(features[self.main.variant]))
            delta=local.double()-reference.double()
        expert=self.expert(x).double()
        return (torch.stack([reference.double(),expert,delta],dim=-1)*self.weights).sum(-1)+self.biases

def verify(split,role):
    m=role_manifest(split,role);positions=np.linspace(0,len(m['indices'])-1,64,dtype=int)
    x,y=base.load(m['path'],np.asarray(m['indices'])[positions]);records=[];codes=y.astype(int)@(1<<np.arange(6));assert len(np.unique(codes))==63
    for path in sorted((ROOT/split/'decision_models').glob('*_22*.json')):
        rec=read(path);model=FactorialRuntime(split,rec).cuda().eval()
        with torch.inference_mode():z=model(x.cuda()).cpu().numpy()
        expected=np.load(ROOT/split/(role+'_results')/(path.stem+'.npz'))['logits'][positions]
        np.testing.assert_allclose(z,expected,rtol=1e-6,atol=1e-6)
        records.append({'name':path.stem,'maximum_error':float(abs(z-expected).max()),'backbones':1 if rec['kind']=='single' else 2})
        del model
    assert len(records)==95
    write(ROOT/split/(role+'_runtime_verification.json'),{'passed':True,'records':records,'replay_positions':positions.tolist(),'distinct_combinations':len(np.unique(codes)),
        'source_sha256':sha(Path(__file__)),'time':stamp()})
    print('RUNTIME_VERIFIED',split,role,len(records),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--split',required=True);p.add_argument('--role',default='development');a=p.parse_args();setup();verify(a.split,a.role)
