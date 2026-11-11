"""Matched complete-coverage A/B/C training across quota cells, with recovery."""
import argparse,random
from types import SimpleNamespace
from common import *

TRAINING_SOURCE=REPO/'experiment-log/2026-11-11-01-label-marginal-controls/training_marginal.py'
trainer=module('matrix_training_backend',TRAINING_SOURCE)
MATRIX=read(HERE/'matrix_protocol.json')

def train_cell(split,cell,seeds):
    assert read(ROOT/'matrix/data_acceptance.json')['passed']
    folder=ROOT/'matrix'/split/cell;data_folder=folder/'data';m=read(data_folder/'manifest.json')
    assert sha(m['fit_path'])==m['fit_sha256'] and sha(m['dev_path'])==m['dev_sha256']
    receipt=read(data_folder/'fit_pairing_receipt.json')
    assert receipt['identity_count']==8000 and receipt['single_target_delta_verified']
    x,y=base.load(m['fit_path'],m['fit_A']);early_x,early_y=base.load(m['dev_path'],m['early_stop']);cal_x,cal_y=base.load(m['dev_path'],m['calibration'])
    dm=role_manifest(split,'development');dev_x,dev_y=base.load(dm['path'],dm['indices']);dev_meta=metadata(dm['path'],dm['indices'])
    np.testing.assert_array_equal(dev_y,dev_meta['labels'])
    weak10,positions10,_=helper.load_views(data_folder/'fit_gap10.mat');weak15,positions15,_=helper.load_views(data_folder/'fit_gap15.mat')
    marginal,combo,design=objective_weights(y,MATRIX['mixture']);weights={'natural':np.ones_like(y),'marginal':marginal,'combo':combo}
    write(folder/'objective_design.json',design)
    tensors=[x,torch.from_numpy(y),early_x,torch.from_numpy(early_y),weak10,torch.from_numpy(positions10),weak15,torch.from_numpy(positions15)]
    x,yt,early_x,early_yt,weak10,positions10,weak15,positions15=[v.cuda() for v in tensors]
    deadline=datetime.fromisoformat(MATRIX['training_cutoff']).timestamp()
    files=[HERE/'train_matrix.py',HERE/'common.py',HERE/'matrix_protocol.json',HERE/'04A_MATRIX_EXECUTION_GOAL.txt',TRAINING_SOURCE,Path(helper.__file__),Path(base.__file__)]
    for seed in seeds:
        # Reserve at least a conservative full matched triad; do not start an arbitrary lone arm near cutoff.
        missing=[e for e in MATRIX['experts'] if not (folder/'models'/f'{e}_{seed}/receipt.json').exists()]
        if missing and time.time()+450*len(missing)>deadline:
            write(folder/f'not_started_{seed}.json',{'seed':seed,'missing':missing,'reason':'insufficient time for conservative matched triad','time':stamp()});break
        initial_hashes=[];order_hashes=[];augmentation=[]
        for expert in MATRIX['experts']:
            output=folder/'training'/expert;output.mkdir(parents=True,exist_ok=True)
            protocol={'split':split,'cell':cell,'expert':expert,'epochs':40,'batch_size':32,'width':32,'seeds':MATRIX['seeds'],
                'manifest_sha256':sha(data_folder/'manifest.json'),'views':{name:sha(data_folder/name) for name in ['fit_gap10.mat','fit_gap15.mat']},
                'weight_sha256':hashlib.sha256(weights[expert].tobytes()).hexdigest(),'source_sha256':{str(p):sha(p) for p in files}}
            pp=output/'protocol.json'
            if pp.exists():assert read(pp)==protocol
            else:write(pp,protocol)
            random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
            model=base.build_backbone('dfms_iq',num_classes=6,width=32,dropout=.2).cuda();initial=helper.model_hash(model)
            options=SimpleNamespace(output=output,epochs=40,training_deadline=deadline,loss_weights=torch.from_numpy(weights[expert]).cuda())
            rec=trainer.train_one(options,model,x,yt,early_x,early_yt,seed,'M',initial,sha(pp),weak10,positions10,weak15,positions15)
            assert rec['epochs']==40 and rec['updates']==10000
            selected=output/f'M_{seed}/best_BCE.pth';assert sha(selected)==rec['checkpoint_sha256']
            model.load_state_dict(torch.load(selected,map_location='cuda',weights_only=False)['model_state_dict'])
            thresholds=helper.fit_thresholds(base.predict(model,cal_x,'cuda').astype(float),cal_y)
            target=folder/'models'/f'{expert}_{seed}';target.mkdir(parents=True,exist_ok=True)
            helper.atomic_torch_save({'model_state_dict':model.state_dict(),'width':32,'labels':LABELS,'thresholds':thresholds,'seed':seed,'receipt':rec},target/'model.pth')
            write(target/'receipt.json',{'model_sha256':sha(target/'model.pth'),'training':rec,'time':stamp()})
            write(target/'calibration.json',{'thresholds':thresholds,'path':m['dev_path'],'source_sha256':m['dev_sha256'],'indices':m['calibration']})
            z=base.predict(model,dev_x,'cuda').astype(float);result=evaluate(z,dev_meta,thresholds)
            out=folder/'development_results';out.mkdir(exist_ok=True)
            np.savez(out/f'{expert}_{seed}.npz',logits=z,thresholds=thresholds);np.savez(out/'metadata.npz',**dev_meta)
            write(out/f'{expert}_{seed}.json',{'method':expert,'seed':seed,**result,'prediction_sha256':sha(out/f'{expert}_{seed}.npz'),'model_sha256':sha(target/'model.pth')})
            initial_hashes.append(rec['initial_state_sha256']);order_hashes.append(rec['order_sha256'])
            history=read(output/f'M_{seed}/history.json');augmentation.append([(r['augmented_presentations'],r['gap15_presentations']) for r in history])
            print('MATRIX_MODEL_COMPLETE',split,cell,expert,seed,'Exact',result['metrics']['f1_exact'],'seconds',rec['elapsed_seconds'],flush=True)
            del model
        assert len(set(initial_hashes))==1 and all(v==order_hashes[0] for v in order_hashes)
        assert all(v==augmentation[0] for v in augmentation)
        write(folder/f'matched_triad_{seed}.json',{'passed':True,'seed':seed,'experts':MATRIX['experts'],'initial_sha256':initial_hashes[0],
            'orders_sha256':hashlib.sha256(json.dumps(order_hashes[0]).encode()).hexdigest(),'matched_view_presentations':True,'time':stamp()})
    print('MATRIX_CELL_BATCH_FINISHED',split,cell,seeds,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--split',required=True,choices=CONFIG['splits']);p.add_argument('--cells',nargs='+',choices=list(MATRIX['cells']),default=list(MATRIX['cells']));p.add_argument('--seeds',nargs='+',type=int,default=MATRIX['seeds']);a=p.parse_args();setup()
    for cell in a.cells:train_cell(a.split,cell,a.seeds)
