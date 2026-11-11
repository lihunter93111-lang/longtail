"""Cross-check the repository-only loader against all frozen confirmation decisions."""
from common import *
from portable_inference import PortableDecision,load_iq

def main():
    setup();records=[]
    for split in CONFIG['splits']:
        manifest=read(ROOT/split/'confirmation/manifest.json')
        x,y=load_iq(manifest['path'],manifest['indices'][:64]);x=x.cuda()
        for path in sorted((ROOT/split/'decision_models').glob('*_23*.json')):
            model=PortableDecision(read(path)).cuda().eval()
            with torch.inference_mode():actual=model(x).cpu().numpy()
            expected=np.load(ROOT/split/'confirmation_results'/(path.stem+'.npz'))['logits'][:64]
            np.testing.assert_allclose(actual,expected,atol=1e-6,rtol=1e-6)
            records.append({'split':split,'decision':path.stem,'maximum_error':float(np.max(abs(actual-expected)))})
            del model
    assert len(records)==100
    write(ROOT/'portable_inference_verification.json',{'passed':True,'records':records,'source_sha256':sha(HERE/'portable_inference.py'),'time':stamp()})
    print('PORTABLE_INFERENCE_VERIFIED',len(records),flush=True)

if __name__=='__main__':main()
