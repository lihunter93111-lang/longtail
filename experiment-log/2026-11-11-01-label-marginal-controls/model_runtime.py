"""Portable normal-forward loader for single and two-backbone decision records."""
from common import *

class DecisionModel(torch.nn.Module):
    def __init__(self,record,path_mapping=None):
        super().__init__()
        mapping=path_mapping or {}
        self.kind=record['kind']
        self.expert=self._backbone(record['model_path'],record['model_sha256'],mapping)
        self.reference=None
        if self.kind=='fusion':
            self.reference=self._backbone(record['reference_path'],record['reference_sha256'],mapping)
            self.register_buffer('weights',torch.tensor(record['state']['weights'],dtype=torch.float64).reshape(6,2))
            self.register_buffer('biases',torch.tensor(record['state']['biases'],dtype=torch.float64))
        self.register_buffer('thresholds',torch.tensor(record['thresholds'],dtype=torch.float64))

    @staticmethod
    def _backbone(original,digest,mapping):
        path=Path(mapping.get(original,original))
        assert sha(path)==digest,path
        model=base.build_backbone('dfms_iq',num_classes=6,width=32,dropout=.2)
        state=torch.load(path,map_location='cpu',weights_only=False)
        model.load_state_dict(state['model_state_dict'])
        return model.eval().requires_grad_(False)

    def forward(self,x):
        right=self.expert(x).double()
        if self.reference is None:return right
        left=self.reference(x).double()
        return left*self.weights[:,0]+right*self.weights[:,1]+self.biases

def load_decision(path,device='cpu',path_mapping=None):
    model=DecisionModel(read(path),path_mapping).to(device).eval().requires_grad_(False)
    return model

def verify_split(split,role='development'):
    m=read(ROOT/split/role/'manifest.json')
    x,_=base.load(m['path'],m['indices'][:64])
    records=[]
    for p in sorted((ROOT/split/'decision_models').glob('*.json')):
        if p.name=='protocol.json':continue
        predictions=ROOT/split/(role+'_results')/(p.stem+'.npz')
        if not predictions.is_file():continue
        model=load_decision(p,'cuda')
        with torch.inference_mode(): actual=model(x.cuda()).cpu().numpy()
        expected=np.load(predictions)['logits'][:64]
        np.testing.assert_allclose(actual,expected,atol=1e-6,rtol=1e-6)
        records.append({'method':p.stem,'maximum_error':float(np.max(abs(actual-expected))),'decision_sha256':sha(p)})
        del model
    write(ROOT/split/(role+'_runtime_replay.json'),{'passed':True,'records':records,'source_sha256':sha(Path(__file__))})

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--split',required=True);p.add_argument('--role',default='development')
    a=p.parse_args();setup();verify_split(a.split,a.role)
