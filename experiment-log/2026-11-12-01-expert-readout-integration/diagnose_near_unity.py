"""Read-only trajectory audit of completed near-unity A/B objective controls."""
from common import *


def diagnose():
    rows=[]
    matrix=read(HERE/'matrix_protocol.json')
    for split in CONFIG['splits']:
        for cell in ['CF7p85_IF1','CF16_near_IF1']:
            folder=ROOT/'matrix'/split/cell
            design=read(folder/'objective_design.json')
            weight_delta=float(np.max(np.abs(np.asarray(design['positive_weight']+design['negative_weight'])-1)))
            margin_delta=float(np.max(np.abs(np.asarray(design['target_p'])-design['p'])))
            for seed in matrix['seeds']:
                marker=folder/f'matched_triad_{seed}.json'
                if not marker.exists():continue
                assert read(marker)['passed']
                receipts={};histories={};scores={};inputs={str(marker):sha(marker)}
                for expert in ['natural','marginal']:
                    rp=folder/'models'/f'{expert}_{seed}/receipt.json'
                    hp=folder/'training'/expert/f'M_{seed}/history.json'
                    sp=folder/'development_results'/f'{expert}_{seed}.json'
                    receipts[expert]=read(rp)['training'];histories[expert]=read(hp);scores[expert]=read(sp)['metrics']
                    inputs.update({str(p):sha(p) for p in [rp,hp,sp]})
                    history=histories[expert];receipt=receipts[expert]
                    assert len(history)==receipt['epochs']==40 and receipt['updates']==10000
                    assert min(history,key=lambda row:row['early_BCE'])['epoch']==receipt['best_BCE_epoch']
                    assert abs(min(row['early_BCE'] for row in history)-receipt['best_early_BCE'])<1e-12
                a,b=receipts['natural'],receipts['marginal']
                assert a['initial_state_sha256']==b['initial_state_sha256'] and a['order_sha256']==b['order_sha256']
                ha,hb=histories['natural'],histories['marginal']
                for ra,rb in zip(ha,hb):
                    assert all(ra[k]==rb[k] for k in ['epoch','lr','augmented_presentations','gap15_presentations'])
                early_difference=[rb['early_BCE']-ra['early_BCE'] for ra,rb in zip(ha,hb)]
                rows.append({'split':split,'cell':cell,'seed':seed,'max_absolute_weight_minus_one':weight_delta,
                    'max_absolute_label_probability_change':margin_delta,'matched_initial_order_views_and_budget':True,
                    'best_BCE_epochs':{e:receipts[e]['best_BCE_epoch'] for e in receipts},
                    'same_selected_epoch':a['best_BCE_epoch']==b['best_BCE_epoch'],
                    'first_epoch_early_BCE_B_minus_A':early_difference[0],
                    'maximum_absolute_same_epoch_early_BCE_difference':float(np.max(np.abs(early_difference))),
                    'development_B_minus_A':{k:scores['marginal'][k]-scores['natural'][k] for k in ['f1_exact','fixed05_exact','macro_AP','BCE']},
                    'input_sha256':inputs})
    result={'passed':True,'records':rows,'scope':'Observed trajectory audit; not a deterministic repeatability or causal attribution experiment.',
        'future_control':'Exact-unity A/A repeats, near-unity B, common final epoch and predeclared best-epoch selection on fresh development.',
        'source_sha256':sha(Path(__file__)),'goal_sha256':sha(HERE/'04G_NEAR_UNITY_DIAGNOSTIC_GOAL.txt'),'time':stamp()}
    write(ROOT/'analysis/near_unity_development.json',result)
    print('NEAR_UNITY_AUDITED',len(rows),flush=True)
    for row in rows:print(row['split'],row['cell'],row['seed'],'max_weight_delta',row['max_absolute_weight_minus_one'],
        'epochs',row['best_BCE_epochs'],'B-A_Exactpp',100*row['development_B_minus_A']['f1_exact'],flush=True)


if __name__=='__main__':diagnose()
