"""Common calibration FPR target; separate from primary F1 decisions."""
import argparse
from common import *
from scipy.special import expit
from verify_core import average_precision
from analyze_matrix import interval

TARGET=.05

def negative_thresholds(z,y):
    result=[];counts=[]
    for j in range(6):
        negative=np.sort(z[y[:,j]==0,j]);allowed=int(np.floor(TARGET*len(negative)))
        cutoff=float(np.nextafter(negative[-allowed-1],np.inf));actual=int(np.sum(negative>=cutoff))
        assert actual<=allowed
        result.append(cutoff);counts.append({'negatives':len(negative),'allowed_fp':allowed,'actual_fp':actual,'actual_fpr':actual/len(negative)})
    return result,counts

def run(role):
    if role=='confirmation':
        for path,expected in read(ROOT/'final_freeze.json')['files'].items():assert sha(path)==expected,path
    matrix=read(HERE/'matrix_protocol.json');rows=[]
    for split in CONFIG['splits']:
        for cell in matrix['cells']:
            folder=ROOT/'matrix'/split/cell;m=read(folder/'data/manifest.json');cal_x=cal_y=None
            for seed in matrix['seeds']:
                if not (folder/f'matched_triad_{seed}.json').exists():continue
                for expert in matrix['experts']:
                    model_path=folder/'models'/f'{expert}_{seed}/model.pth';state_path=model_path.parent/'fixed_fpr_operating_point.json'
                    binding={'model_sha256':sha(model_path),'calibration_source_sha256':m['dev_sha256'],
                        'calibration_indices_sha256':hashlib.sha256(np.asarray(m['calibration'],dtype='<i8').tobytes()).hexdigest(),'target_fpr':TARGET}
                    if state_path.exists():state=read(state_path);assert state['binding']==binding
                    else:
                        assert role=='development' and not (ROOT/'final_freeze.json').exists()
                        if cal_x is None:cal_x,cal_y=base.load(m['dev_path'],m['calibration'])
                        model=base.build_backbone('dfms_iq',6,width=32,dropout=.2).cuda();model.load_state_dict(torch.load(model_path,map_location='cpu',weights_only=False)['model_state_dict'])
                        z=base.predict(model,cal_x,'cuda').astype(float);thresholds,counts=negative_thresholds(z,cal_y)
                        state={'binding':binding,'logit_thresholds':thresholds,'calibration':counts,'source_sha256':sha(Path(__file__)),'time':stamp()}
                        write(state_path,state);del model
                    source=folder/(role+'_results');data=np.load(source/f'{expert}_{seed}.npz');meta=np.load(source/'metadata.npz');z=data['logits'];y=meta['labels'].astype(bool)
                    pred=z>=np.asarray(state['logit_thresholds']);p=expit(z);labels=[]
                    for j,label in enumerate(LABELS):
                        positive=y[:,j];negative=~positive;weak=meta['weak'][:,j];weak_ap_mask=negative|weak
                        labels.append({'label':label,'FPR':float(pred[negative,j].mean()),'recall':float(pred[positive,j].mean()),
                            'weak_recall':float(pred[weak,j].mean()),'weak_positive_n':int(weak.sum()),'negative_n':int(negative.sum()),
                            'weak_vs_negative_AP':average_precision(y[weak_ap_mask,j],p[weak_ap_mask,j])})
                    row={'split':split,'cell':cell,'seed':seed,'expert':expert,'exact':float(np.all(pred==y,axis=1).mean()),
                         'BCE':float(np.mean(np.logaddexp(0,z)-y*z)),'labels':labels,'operating_point_sha256':sha(state_path),
                         'prediction_source_sha256':sha(source/f'{expert}_{seed}.npz')}
                    write(folder/('fixed_fpr_'+role+'_results')/f'{expert}_{seed}.json',row);rows.append(row)
    comparisons=[]
    for split in CONFIG['splits']:
        for cell in matrix['cells']:
            by={(r['expert'],r['seed']):r for r in rows if r['split']==split and r['cell']==cell};seeds=sorted({s for e,s in by})
            if not seeds:continue
            for a,b in [('combo','marginal'),('marginal','natural'),('combo','natural')]:
                comparison={'split':split,'cell':cell,'candidate':a,'reference':b,'exact_difference':interval([by[a,s]['exact']-by[b,s]['exact'] for s in seeds]),'labels':[]}
                for j,label in enumerate(LABELS):
                    comparison['labels'].append({'label':label,**{key:interval([by[a,s]['labels'][j][key]-by[b,s]['labels'][j][key] for s in seeds]) for key in ['FPR','recall','weak_recall','weak_vs_negative_AP']}})
                comparisons.append(comparison)
                print('OPERATING_POINT',role,split,cell,a,b,'Exactpp',100*comparison['exact_difference']['mean'],'n',len(seeds),flush=True)
    # Discrete ties and the zero-FP boundary must both respect the target.
    synthetic=np.zeros((100,6));labels=np.zeros((100,6));thresholds,counts=negative_thresholds(synthetic,labels)
    assert all(c['actual_fp']==0 for c in counts) and all(t>0 for t in thresholds)
    result={'role':role,'target_calibration_fpr':TARGET,'records':rows,'comparisons':comparisons,'ties_verified':True,
        'boundary':'Secondary diagnostic; calibrated FPR target need not hold on evaluation. Weak AP uses weak positives plus all negatives only. Primary F1 decisions unchanged.',
        'time':stamp()}
    write(ROOT/'analysis'/f'matrix_operating_points_{role}.json',result)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--role',default='development');a=p.parse_args();setup();run(a.role)
