"""Score only frozen, fully matched matrix triads on new confirmation."""
import argparse
from common import *

def score(split):
    freeze=read(ROOT/'final_freeze.json')
    for path,expected in freeze['files'].items():assert sha(path)==expected,path
    m=role_manifest(split,'confirmation');assert m['freeze_sha256']==sha(ROOT/'final_freeze.json') and sha(m['path'])==m['sha256']
    x,y=base.load(m['path'],m['indices']);meta=metadata(m['path'],m['indices']);np.testing.assert_array_equal(y,meta['labels']);records=[]
    for triad in freeze['matrix_completed_triads']:
        if triad['split']!=split:continue
        cell=triad['cell'];seed=triad['seed'];folder=ROOT/'matrix'/split/cell;out=folder/'confirmation_results';out.mkdir(exist_ok=True)
        np.savez(out/'metadata.npz',**meta)
        for expert in ['natural','marginal','combo']:
            model_path=folder/'models'/f'{expert}_{seed}/model.pth';state=torch.load(model_path,map_location='cpu',weights_only=False)
            model=base.build_backbone('dfms_iq',6,width=32,dropout=.2).cuda();model.load_state_dict(state['model_state_dict']);model.eval()
            thresholds=read(model_path.parent/'calibration.json')['thresholds'];np.testing.assert_array_equal(thresholds,state['thresholds'])
            z=base.predict(model,x,'cuda').astype(float);replay=base.predict(model,x[:64],'cuda').astype(float);np.testing.assert_allclose(z[:64],replay,atol=1e-6,rtol=1e-6)
            np.savez(out/f'{expert}_{seed}.npz',logits=z,thresholds=thresholds)
            row={'method':expert,'seed':seed,**evaluate(z,meta,thresholds),'prediction_sha256':sha(out/f'{expert}_{seed}.npz'),'model_sha256':sha(model_path),
                 'freeze_sha256':sha(ROOT/'final_freeze.json'),'replay_max_error':float(abs(z[:64]-replay).max())}
            write(out/f'{expert}_{seed}.json',row);records.append({'cell':cell,**row});del model
            print('MATRIX_CONFIRMED',split,cell,expert,seed,row['metrics']['f1_exact'],flush=True)
    write(ROOT/'matrix'/split/'confirmation_summary.json',{'results':records,'time':stamp()})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--split',required=True);a=p.parse_args();setup();score(a.split)
