"""Actual identities, repeated visits, inherited versus newly trained work."""
from common import *

def count_rows(path):
    with h5py.File(path,'r') as f:return int(f['known_label'].shape[1])

def audit():
    core=[];matrix=[];cfg=read(HERE/'matrix_protocol.json');durations=[]
    for split in CONFIG['splits']:
        m=read(BASELINES/split/'data/manifest.json');fusion=role_manifest(split,'fusion_fit')
        core.append({'split':split,'fit_candidate_rows':count_rows(m['fit_path']),'actual_fit_identities':len(m['fit_A']),
            'early_identities':len(m['early_stop']),'calibration_identities':len(m['calibration']),
            'fusion_fit_identities':len(fusion['indices']),'parameter_fit_identity_union':len(m['fit_A'])+len(fusion['indices']),
            'selection_plus_parameter_fit_identity_union':len(m['fit_A'])+len(fusion['indices'])+len(m['early_stop'])+len(m['calibration']),
            'development_identities':4032,'confirmation_identities':4032,
            'historical_backbones_reused':20,'historical_readout_models_reused':10,
            'new_affine_models':45,'affine_optimizer_steps_per_model':1800,'affine_epochs_per_label':300,
            'affine_parameters_per_model':24,'guarded_derived_decisions':30,'single_model_decisions':20})
        union=set();cells=[]
        for cell in cfg['cells']:
            folder=ROOT/'matrix'/split/cell;manifest=read(folder/'data/manifest.json');union.update(manifest['fit_A']);models=[]
            for seed in cfg['seeds']:
                if not (folder/f'matched_triad_{seed}.json').exists():continue
                for e in cfg['experts']:
                    receipt=read(folder/'models'/f'{e}_{seed}/receipt.json')['training'];history=read(folder/'training'/e/f'M_{seed}/history.json')
                    models.append({'expert':e,'seed':seed,'epochs':receipt['epochs'],'updates':receipt['updates'],
                        'identity_presentations':receipt['unique_fit_per_epoch']*receipt['epochs'],
                        'weak_view_presentations':sum(r['augmented_presentations'] for r in history),
                        'gap15_view_presentations':sum(r['gap15_presentations'] for r in history),
                        'process_elapsed_seconds':receipt['elapsed_seconds'],'best_BCE_epoch':receipt['best_BCE_epoch'],
                        'best_AP_epoch':receipt['best_AP_epoch'],'last_fit_BCE':history[-1]['fit_eval_BCE'],
                        'last_early_BCE':history[-1]['early_BCE'],'last_early_macro_AP':history[-1]['early_macro_AP'],
                        'early_BCE_change_epoch35_to40':history[-1]['early_BCE']-history[34]['early_BCE']})
                    durations.append(receipt['elapsed_seconds'])
            cells.append({'cell':cell,'fit_n':len(manifest['fit_A']),'candidate_pool_rows':count_rows(manifest['fit_path']),
                'early_n':len(manifest['early_stop']),'calibration_n':len(manifest['calibration']),'models':models})
        matrix.append({'split':split,'fit_identity_union_across_all_four_designed_cells':len(union),'cells':cells})
    controls=[read(p) for p in (ROOT/'supervision_control').glob('*/models/reference_affine_22*.json')];assert len(controls)==10
    finished=[];unfinished=[]
    for path in sorted((ROOT/'matrix').glob('*/*/training/*/M_*/receipt.json')):
        receipt=read(path);matched=(path.parents[3]/f"matched_triad_{receipt['seed']}.json").exists()
        finished.append({'path':str(path),'epochs':receipt['epochs'],'updates':receipt['updates'],
                         'identity_presentations':receipt['epochs']*receipt['unique_fit_per_epoch'],
                         'elapsed_seconds':receipt['elapsed_seconds'],'in_complete_matched_triad':matched})
    for path in sorted((ROOT/'matrix').glob('*/*/training/*/M_*/history.json')):
        if (path.parent/'receipt.json').exists():continue
        try:epochs=len(read(path))
        except json.JSONDecodeError:epochs=None
        unfinished.append({'path':str(path.parent),'completed_epochs_observed':epochs,
                           'updates_lower_bound':None if epochs is None else 250*epochs})
    observed_updates=sum(row['updates'] for row in finished)+sum(row['updates_lower_bound'] or 0 for row in unfinished)
    supervision={'decisions':len(controls),'new_backbones':0,'optimizer_steps':sum(r['training']['optimizer_steps'] for r in controls),
                 'effective_parameters_per_decision':12,'allocated_parameters_per_decision':24,
                 'fusion_fit_identities_per_cohort':4032,'new_identity_union_beyond_core':0,
                 'sum_fit_phase_elapsed_seconds':sum(r['training']['elapsed_seconds'] for r in controls),
                 'timing_excludes':'threshold fitting and evaluation; includes cached-logit reads and affine optimization',
                 'scope':'single frozen backbone with extra supervised affine calibration, not equal total compute control'}
    result={'core':core,'matrix':matrix,'supervision_control':supervision,'new_complete_backbones':len(durations),'new_backbone_updates':10000*len(durations),
        'new_backbone_identity_presentations':320000*len(durations),'sum_training_process_elapsed_seconds':float(sum(durations)),
        'all_finished_training_runs':finished,'unfinished_training_directories':unfinished,
        'all_finished_backbones':len(finished),'observed_training_updates_lower_bound':observed_updates,
        'observed_training_identity_presentations_lower_bound':32*observed_updates,
        'count_scope':'new_complete_backbones and its update/visit totals cover complete matched triads only; all_finished_training_runs also includes any unmatched finished arm; unfinished histories provide a lower bound and do not establish live process status',
        'warning':'Sum of per-process elapsed time includes GPU contention and is not exclusive GPU compute time. Same identities recur across seeds and quota cells. Candidate pool rows are not all used for training.',
        'time':stamp()}
    write(ROOT/'budget_audit.json',result);print('BUDGET_AUDITED',len(durations),flush=True)

if __name__=='__main__':audit()
