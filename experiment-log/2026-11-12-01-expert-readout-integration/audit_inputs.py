"""Audit raw weights, exported heads, data hashes and matched backbone state."""
from common import *

def main():
    setup();records=[];files={}
    previous=read(MARGINAL/'final_freeze.json')
    for path,digest in previous['files'].items():assert sha(path)==digest,path
    for split in CONFIG['splits']:
        manifest=read(BASELINES/split/'data/manifest.json')
        for role in ['fit','dev']:
            path=Path(manifest[role+'_path']);assert sha(path)==manifest[role+'_sha256'];files[str(path)]=manifest[role+'_sha256']
        for name in ['fit_gap10.mat','fit_gap15.mat']:
            path=BASELINES/split/'data'/name;files[str(path)]=sha(path)
        ff=role_manifest(split,'fusion_fit');assert sha(ff['path'])==ff['sha256'];files[ff['path']]=ff['sha256']
        x,_=base.load(manifest['dev_path'],manifest['calibration'][:32])
        for seed in CONFIG['seeds']:
            reference=load_model(split,'reference',seed,'cpu')
            for kind in ['reference','natural','marginal','combo','global','segment']:
                path=checkpoint(split,kind,seed);digest=sha(path);files[str(path)]=digest
                extra={}
                if kind in ['global','segment']:
                    receipt=read(path.parent/'receipt.json');assert receipt['checkpoint_sha256']==digest
                    model=load_model(split,kind,seed,'cpu')
                    left=reference.state_dict();right=model.backbone.state_dict()
                    assert left.keys()==right.keys() and all(torch.equal(left[k],right[k]) for k in left)
                    assert all(head.hidden.out_features==32 and head.hidden.in_features==128 for head in model.heads)
                    with torch.inference_mode():
                        natural=reference(x);pre,features=model.feature_branches(x)
                        torch.testing.assert_close(natural,pre,rtol=1e-6,atol=1e-6)
                    extra={'head_selected_epochs':[s['best_epoch'] for s in receipt['stages']],
                        'targets':[s['targets'] for s in receipt['stages']],'main_backbone_exact_match':True,
                        'input_dimension':int(features[model.variant].shape[1]),'readout_hidden':32,'pca_dimension':128}
                    del model
                records.append({'split':split,'seed':seed,'kind':kind,'path':str(path),'sha256':digest,**extra})
            del reference
    write(ROOT/'input_audit.json',{'passed':True,'models':records,'data_and_model_files':files,'prior_frozen_files_verified':len(previous['files']),
        'source_sha256':code_hashes(),'time':stamp()})
    print('INPUT_AUDIT_PASSED',len(records),'models',flush=True)

if __name__=='__main__':main()
