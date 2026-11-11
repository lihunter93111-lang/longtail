"""Independently reconstruct data, order, augmentation and checkpoint contracts."""
from common import *

def verify():
    torch.set_num_threads(4);cfg=read(HERE/'matrix_protocol.json');rows=[];cells=[];largest=0.
    for split in CONFIG['splits']:
        for cell in cfg['cells']:
            folder=ROOT/'matrix'/split/cell;m=read(folder/'data/manifest.json');indices=np.asarray(m['fit_A'])
            assert len(indices)==len(np.unique(indices))==8000 and sha(m['fit_path'])==m['fit_sha256']
            assert sha(m['dev_path'])==m['dev_sha256']
            with h5py.File(m['fit_path'],'r') as f:y=np.asarray(f['known_label'],dtype=np.float32).T[indices]
            codes=y.astype(int)@(1<<np.arange(6));counts=np.bincount(codes,minlength=64)[1:]
            np.testing.assert_array_equal(counts,cfg['cells'][cell]['counts']);np.testing.assert_array_equal(y.sum(0),cfg['cells'][cell]['label_margins'])
            weights_b,weights_c,_=objective_weights(y,cfg['mixture'])
            errors=[np.max(abs((weights_b.astype(float)*mask).mean(0)-(weights_c.astype(float)*mask).mean(0))) for mask in [y,1-y]]
            assert max(errors)<1e-7;largest=max(largest,max(errors))
            positions={}
            for gap in [10,15]:
                with h5py.File(folder/'data'/f'fit_gap{gap}.mat','r') as f:positions[gap]=np.asarray(f['positions']).reshape(-1).astype(int)
            assert np.isin(positions[15],positions[10]).all()
            expected_views=[]
            for epoch in range(1,41):
                selected10=(positions[10]+epoch)%2==0
                selected15=((positions[15]+epoch)%2==0)&(((epoch+positions[15]%2)//2)%2==0)
                assert np.isin(positions[15][selected15],positions[10][selected10]).all()
                expected_views.append((int(selected10.sum()),int(selected15.sum())))
            assert sum(v[0] for v in expected_views)==20*len(positions[10])
            assert sum(v[1] for v in expected_views)==10*len(positions[15])
            cells.append({'split':split,'cell':cell,'positive_negative_float32_weight_margin_max_error':max(errors),
                'views10_identities':len(positions[10]),'views15_identities':len(positions[15]),'full_epoch_identities':8000,
                'augmented_identity_presentations':sum(v[0] for v in expected_views),'gap15_identity_presentations':sum(v[1] for v in expected_views)})
            for seed in cfg['seeds']:
                if not (folder/f'matched_triad_{seed}.json').exists():continue
                generator=torch.Generator().manual_seed(seed);orders=[]
                for _ in range(40):orders.append(hashlib.sha256(torch.randperm(8000,generator=generator).numpy().tobytes()).hexdigest())
                torch.manual_seed(seed);initial=base.build_backbone('dfms_iq',6,width=32,dropout=.2);initial_hash=helper.model_hash(initial);del initial
                for expert in cfg['experts']:
                    train=folder/'training'/expert/f'M_{seed}';model_folder=folder/'models'/f'{expert}_{seed}'
                    protocol=read(train.parent/'protocol.json');receipt=read(train/'receipt.json');export=read(model_folder/'receipt.json');history=read(train/'history.json')
                    assert protocol['manifest_sha256']==sha(folder/'data/manifest.json')
                    for path,digest in protocol['source_sha256'].items():assert sha(path)==digest,path
                    for name,digest in protocol['views'].items():assert sha(folder/'data'/name)==digest
                    weights={'natural':np.ones_like(y),'marginal':weights_b,'combo':weights_c}[expert]
                    assert hashlib.sha256(weights.tobytes()).hexdigest()==protocol['weight_sha256']
                    assert receipt['epochs']==40 and receipt['updates']==10000 and receipt['unique_fit_per_epoch']==8000
                    assert receipt['initial_state_sha256']==initial_hash and receipt['order_sha256']==orders
                    assert [(r['augmented_presentations'],r['gap15_presentations']) for r in history]==expected_views
                    assert sha(train/'best_BCE.pth')==receipt['checkpoint_sha256'] and sha(model_folder/'model.pth')==export['model_sha256']
                    assert export['training']==receipt
                    selected=torch.load(train/'best_BCE.pth',map_location='cpu',weights_only=False)
                    deployed=torch.load(model_folder/'model.pth',map_location='cpu',weights_only=False)
                    a=selected['model_state_dict'];b=deployed['model_state_dict'];assert a.keys()==b.keys() and all(torch.equal(a[k],b[k]) for k in a)
                    assert deployed['width']==32 and deployed['labels']==LABELS
                    np.testing.assert_array_equal(deployed['thresholds'],read(model_folder/'calibration.json')['thresholds'])
                    rows.append({'split':split,'cell':cell,'expert':expert,'seed':seed,'initial_exact':True,'orders_reconstructed':True,
                        'augmentation_counts_reconstructed':True,'export_equals_selected_checkpoint':True,'selected_epoch':selected['epoch']})
                    del selected,deployed,a,b
    result={'passed':True,'models':len(rows),'records':rows,'cells':cells,'maximum_float32_objective_margin_error':largest,'time':stamp()}
    write(ROOT/'matrix_contract_verification.json',result);print('MATRIX_CONTRACT_VERIFIED',len(rows),'margin_max_error',largest,flush=True)

if __name__=='__main__':verify()
