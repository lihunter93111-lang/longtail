"""Verify paired references and train only the new marginal-matched control."""
import argparse
import random
import time
from types import SimpleNamespace
from common import *
import training_marginal as trainer

def audit(split):
    data=BASELINES/split/'data'
    m=read(data/'manifest.json')
    checks=[]
    for role in ['fit','dev']:
        path=Path(m[role+'_path'])
        assert sha(path)==m[role+'_sha256']
        checks.append({'path':str(path),'sha256':m[role+'_sha256']})
    for group in ['ordinary_expert','weighted_expert']:
        folder=PREVIOUS/split/'expert_training'/group
        p=read(folder/'protocol.json')
        assert sha(data/'manifest.json')==p['manifest_sha256']
        for name,digest in p['views'].items():
            assert sha(data/name)==digest
        for file,digest in p['source_sha256'].items():
            assert sha(file)==digest,(file,'source changed')
        for seed in CONFIG['seeds']:
            rec=read(folder/f'M_{seed}'/'receipt.json')
            assert rec['epochs']==40 and rec['updates']==10000
            assert sha(folder/f'M_{seed}'/'best_BCE.pth')==rec['checkpoint_sha256']
    result={'status':'passed','time':stamp(),'manifest_sha256':sha(data/'manifest.json'),'sources':checks,'models_verified':10,'views_verified':2}
    write(ROOT/split/'reference_audit.json',result)
    return m

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--split',required=True,choices=CONFIG['splits'])
    parser.add_argument('--seeds',nargs='+',type=int)
    args=parser.parse_args()
    setup()
    m=audit(args.split)
    x,y=base.load(m['fit_path'],m['fit_A'])
    early_x,early_y=base.load(m['dev_path'],m['early_stop'])
    cal_x,cal_y=base.load(m['dev_path'],m['calibration'])
    weak10,positions10,_=helper.load_views(BASELINES/args.split/'data/fit_gap10.mat')
    weak15,positions15,_=helper.load_views(BASELINES/args.split/'data/fit_gap15.mat')
    weights,combo,design=objective_weights(y)
    write(ROOT/args.split/'objective_design.json',design)
    tensors=[x,torch.from_numpy(y),early_x,torch.from_numpy(early_y),weak10,torch.from_numpy(positions10),weak15,torch.from_numpy(positions15)]
    x,yt,early_x,early_yt,weak10,positions10,weak15,positions15=[v.cuda() for v in tensors]
    output=ROOT/args.split/'training/marginal'
    output.mkdir(parents=True,exist_ok=True)
    protocol={'split':args.split,'arm':'label_positive_negative_marginal_matching','epochs':40,'seeds':CONFIG['seeds'],'manifest_sha256':sha(BASELINES/args.split/'data/manifest.json'),'weight_sha256':hashlib.sha256(weights.tobytes()).hexdigest(),'source_sha256':{str(p):sha(p) for p in [HERE/'run_marginal.py',HERE/'common.py',HERE/'training_marginal.py',HERE/'protocol.json',HERE/'GOAL_PROMPT.txt',HERE/'00_E_DRIVE_AMENDMENT_GOAL.txt',Path(helper.__file__),Path(base.__file__)]},'goal':'C combo versus B label marginals matched'}
    pp=output/'protocol.json'
    if pp.exists(): assert read(pp)==protocol
    else: write(pp,protocol)
    for seed in args.seeds or CONFIG['seeds']:
        if time.time()>=datetime.fromisoformat(CONFIG['training_cutoff']).timestamp(): break
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        model=base.build_backbone('dfms_iq',num_classes=6,width=32,dropout=.2).cuda()
        initial=helper.model_hash(model)
        ordinary=read(PREVIOUS/args.split/'expert_training/ordinary_expert'/f'M_{seed}/receipt.json')
        weighted=read(PREVIOUS/args.split/'expert_training/weighted_expert'/f'M_{seed}/receipt.json')
        assert initial==ordinary['initial_state_sha256']==weighted['initial_state_sha256']
        options=SimpleNamespace(output=output,epochs=40,training_deadline=datetime.fromisoformat(CONFIG['training_cutoff']).timestamp(),loss_weights=torch.from_numpy(weights).cuda())
        rec=trainer.train_one(options,model,x,yt,early_x,early_yt,seed,'M',initial,sha(pp),weak10,positions10,weak15,positions15)
        assert rec['order_sha256']==ordinary['order_sha256']==weighted['order_sha256']
        z=base.predict(model,cal_x,'cuda').astype(float)
        thresholds=helper.fit_thresholds(z,cal_y)
        destination=ROOT/args.split/'models'/f'marginal_{seed}'
        destination.mkdir(parents=True,exist_ok=True)
        helper.atomic_torch_save({'model_state_dict':model.state_dict(),'width':32,'labels':LABELS,'thresholds':thresholds,'seed':seed,'receipt':rec},destination/'model.pth')
        write(destination/'calibration.json',{'thresholds':thresholds,'source_sha256':m['dev_sha256'],'indices':m['calibration']})
        np.savez(destination/'calibration_logits.npz',logits=z,labels=cal_y)
        write(destination/'receipt.json',{'model_sha256':sha(destination/'model.pth'),'time':stamp(),'training':rec,'matched_initialization':True,'matched_orders':True})
        print('MARGINAL_COMPLETE',args.split,seed,flush=True)
        del model
    write(ROOT/args.split/'training_status.json',{'time':stamp(),'completed':[p.name for p in (ROOT/args.split/'models').glob('marginal_*')]})

if __name__=='__main__': main()
