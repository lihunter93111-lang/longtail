"""Frozen candidates -> large independent calibration -> low-dose targeted pairs."""
import argparse
from datetime import datetime
from html import escape
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
SEEDS = list(range(2050,2055))
SLOTS = [list(range(8+4*i,12+4*i)) for i in range(4)]+[list(range(40,44))]
STAGES = ['calibration','targeted']
STYLE = '<style>body{font:16px/1.8 system-ui;max-width:1280px;margin:24px auto;padding:24px;color:#234}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:7px;border:1px solid #ccd}th{background:#e9f3f4}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style>'


def now():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')


def table(headers,rows):
    return '<table><tr>'+''.join('<th>'+escape(str(v))+'</th>' for v in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+escape(str(v))+'</td>' for v in row)+'</tr>' for row in rows)+'</table>'


def prepare(suite,data,report):
    import numpy as np
    import torch
    from longtail.io import read_json,write_json,sha256
    from longtail.data.splits import build_split
    from longtail.data.dataset import IQDataset
    from longtail.engine.checkpoint import load_checkpoint
    from longtail.replication_temporal import load_backbone
    from longtail.models.temporal_heads import GlobalResidual
    from longtail.targeted_validation import SawMFTHead,compose_targets,candidate_logits
    if suite.exists():
        raise FileExistsError('Use a new suite')
    ready = read_json(data/'generation_complete.json')
    assert ready['status'] == 'completed'
    assert set(sum(SLOTS,[])) <= os.sched_getaffinity(0)
    previous = ROOT/'runs/replication_temporal_cpu_20261104'
    selected = read_json(previous/'selected_models.json')
    former = read_json(previous/'plan.json')
    references = {}
    for seed in SEEDS:
        new = next(v for v in selected['models'] if v['seed'] == seed)
        assert new['local_architecture'] == 'global32'
        refs = {'new':{},'old':{}}
        for kind,key in [('B','B'),('linear','linear_head'),('local','local_head')]:
            refs['new'][kind] = new['components'][key]
        for kind,key in [('B','B'),('linear','linear'),('local','local')]:
            source = former['references'][str(seed)][key]
            refs['old'][kind] = {'path':source['checkpoint'],'sha256':source['sha256']}
        decisions = read_json(previous/'retrain'/f'seed{seed}/decisions_retrain.json')['thresholds']
        refs['new']['thresholds'] = decisions['global32']
        refs['old']['thresholds'] = decisions['old_candidate']
        source = former['references'][str(seed)]['FMB']
        refs['FMB'] = {'path':source['checkpoint'],'sha256':source['sha256']}
        for version in ['new','old']:
            for kind in ['B','linear','local']:
                assert sha256(refs[version][kind]['path']) == refs[version][kind]['sha256']
        assert sha256(refs['FMB']['path']) == refs['FMB']['sha256']
        references[str(seed)] = refs
    _,cfg = load_checkpoint(references['2050']['new']['B']['path'])
    config = json.loads(json.dumps(cfg['data']))
    config['per_combination'] = {'calibration':128,'check':32,'development_test':128}
    # Real I/Q checks before source and protocol freeze.
    torch.set_num_threads(4)
    manifest = Path(ready['manifests']['calibration'])
    dataset = IQDataset(manifest.parent/'train.h5',np.arange(8),cfg['dataloader']['preprocess'])
    x = torch.stack([dataset[i]['signal'] for i in range(8)])
    y = torch.stack([dataset[i]['target'] for i in range(8)])
    dataset.close()
    backbone,_ = load_backbone(Path(references['2050']['new']['B']['path']))
    with torch.no_grad():
        feature = backbone.forward_features(x).detach()
    payload = torch.load(references['2050']['new']['local']['path'],map_location='cpu',weights_only=True)
    original = GlobalResidual()
    original.load_state_dict(payload['state_dict'])
    original.eval().requires_grad_(False)
    copy = SawMFTHead(payload['state_dict'])
    torch.testing.assert_close(copy(feature),original(feature)[:,[0,1]],rtol=1e-5,atol=2e-6)
    optimizer = torch.optim.AdamW(copy.parameters(),lr=.0003)
    for _ in range(2):
        optimizer.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(copy(feature),y[:,[2,4]])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(copy.parameters(),5,error_if_nonfinite=True)
        optimizer.step()
    assert all(torch.equal(v,payload['state_dict'][k]) for k,v in original.state_dict().items())
    suite.mkdir(parents=True)
    report.mkdir(parents=True,exist_ok=True)
    hashes = {}
    for tag,path in ready['manifests'].items():
        m = read_json(path)
        split = build_split(m,Path(path).parent,config)
        write_json(suite/'splits'/(tag+'.json'),split)
        hashes[path] = sha256(path)
        for info in m['files'].values():
            hashes[str(Path(path).parent/info['path'])] = info['sha256']
    pairs = read_json(ready['pair_manifest'])
    hashes[ready['pair_manifest']] = sha256(ready['pair_manifest'])
    groups = {role:{v['edge_id'] for v in pairs['edges'] if v['role'] == role} for role in ['fit','check','development_test']}
    assert all(not groups[a]&groups[b] for a,b in [('fit','check'),('fit','development_test'),('check','development_test')])
    for info in pairs['files'].values():
        hashes[info['path']] = info['sha256']
    for info in ready['shifts'].values():
        hashes[info['path']] = info['sha256']
    source = []
    files = sorted((ROOT/'src').rglob('*.py'))+sorted((ROOT/'experiments').glob('*.py'))+sorted((ROOT/'tests').glob('*.py'))+sorted((ROOT/'configs').glob('*.json'))+[ROOT/'pyproject.toml',ROOT/'uv.lock']
    for file in files:
        dest = suite/'source_snapshot'/file.relative_to(ROOT)
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(file,dest)
        source.append({'path':str(file.relative_to(ROOT)),'sha256':sha256(dest)})
    protocol = {
        'stage_order':STAGES,'seeds':SEEDS,'CPU_only':True,'parallel_jobs':5,'physical_cores_per_job':4,
        'immutable_references':'new natural global32 candidate and previous paired candidate; all backbone/linear/local/FMB source files SHA-bound',
        'calibration':{'natural_pool':{'calibration':8064,'check':2016,'report':8064},
            'changed_thresholds':['SawFM','MFT'],'other_four_thresholds':'locked at original candidate decision values',
            'rules':['locked','f1_1008','f1_8064','fpr03','fpr05','fpr07','weak_guard'],
            'small_pool_control':'block0 of8 disjoint1008 blocks, each16 identities per63 combinations; no best-block selection',
            'FPR_rules':'same empirical3/5/7% calibration target FP budgets for all models; report FPR is measured, not guaranteed equal',
            'weak_guard':'maximize calibration Exact over two target thresholds, shared max FPR=max(new/old large-F1)+0.5pp; weak recall>=max(new/old large-F1)-1pp, own overall recall>=large-F1-1pp; explicit infeasibility',
            'selection':'one shared policy maximizing mean check Exact over new/old candidates, with each candidate Exact>=locked-0.2pp, Saw/MFT weak recall>=locked-1pp, per-class FPR<=locked+1pp, tailF1>=locked-0.3pp; locked always eligible'},
        'targeted':{'new_models':15,'methods':{'natural':0.,'pair1':.0625,'pair2':.125},
            'initialization':'copy current new-candidate local hidden layer and SawFM/MFT output rows; reference candidate files never changed',
            'parameters':4194,'changed_logit_columns':['SawFM','MFT'],
            'other_four_logits':'exactly copied from original frozen new candidate, including ISRJ',
            'optimizer':'AdamW','lr':.0003,'weight_decay':.0001,'scheduler':'24epoch cosine','epochs':24,'steps_per_epoch':250,
            'batch_size':32,'steps_per_model':6000,'views_per_model':192000,'total_optimizer_steps':90000,
            'batches':{'natural':'32 natural','pair1':'30 natural+1 pair','pair2':'28 natural+2 pairs'},
            'gap_bins_db':[[6,10],[10,15],[15,20]],'targets':['SawFM','MFT'],
            'sampling':'balanced six target-gap cells across24epochs, natural identities without replacement within epoch',
            'loss':'ordinary BCE on two target logits; no focal, margin or extra positive weights',
            'checkpoint':'check mean weak AUPR of SawFM/MFT; all arms use same criterion and budget',
            'data':'fixed existing natural32k fit; fresh conditional one-bit fit; fresh8064/2016/8064 calibration/check/report',
            'policy':'stage1 check-selected policy; if either frozen reference infeasible across5 new calibration seeds, common f1_8064 fallback; exclude individually infeasible trained methods',
            'selection':'check Exact among frozen-new/natural/pair1/pair2, subject to same weak/FPR/tail safeguards relative frozen-new; no eligible gain retains frozen-new',
            'secondary_reports':'fresh10--40dB natural report and gap-stratified held-out pair edges, never used for selection'},
        'formal_Exact_success':{'gain_at_least_pp':.5,'positive_seeds_at_least':4,'paired95_lower_positive':True,'weak_drop_max_pp':1.,'class_FPR_rise_max_pp':1.,'tail_F1_drop_max_pp':.3},
        'secondary_weak_tradeoff_success':{'Saw_MFT_mean_weak_recall_gain_at_least_pp':2.,'weak_paired95_lower_positive':True,'Exact_drop_max_pp':.2,'per_class_FPR_rise_max_pp':.5,'each_target_weak_drop_max_pp':1.},
        'isolation':'stage1/2 separate identities, calibration-only thresholds; check-only candidate/policy selection; freeze before report; stage1 report never drives stage2',
        'limits':['Same Python physical-equation PCG64 simulator, no MATLAB bitwise parity or real measurements.',
                  'Pair edges held out relative to pair fit; pretrained natural backbone has seen all63 combinations.',
                  'Five reused model seeds; fixed-pool conditional intervals; secondary comparisons exploratory, no multiplicity correction.',
                  'Targeted fit remains within original10--30dB and gaps below20dB; no claim of solving20--30dB gap OOD.']}
    plan = {'created':now(),'root':str(ROOT),'previous_suite':str(previous),'data_root':str(data),'data':ready,
            'seeds':SEEDS,'slots':SLOTS,'references':references,'data_hashes':hashes,'source_manifest':source,
            'report_output':str(report),'protocol':protocol,
            'preflight':{'real_IQ_two_steps_passed':True,'head_copy_parameters':sum(p.numel() for p in copy.parameters()),
                         'source_local_unchanged':True,'new_data_no_old_identity_overlap':True}}
    write_json(suite/'plan.json',plan)
    write_json(suite/'preflight.json',plan['preflight'])
    html = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>大校准池与小比例定向配对验证方案</title>'+STYLE+'<h1>固定新旧候选：大校准池 → 小比例分层配对</h1><p>先固定全部权重比较SawFM/MFT误报与漏检；然后只训练新局部头副本，其他四类logits和两个参考候选保持固定。CPU五种子、同预算、报告前冻结。</p><pre>'+escape(json.dumps(protocol,ensure_ascii=False,indent=2))+'</pre></html>'
    (report/'plan.html').write_text(html)
    render(suite)
    print(json.dumps({'prepared':str(suite),'preflight':plan['preflight']},ensure_ascii=False),flush=True)


def worker(suite,stage,seed,report=False):
    from longtail.io import read_json,write_json
    from longtail.targeted_validation import worker as run_worker
    import torch
    plan = read_json(suite/'plan.json')
    cpus = plan['slots'][plan['seeds'].index(seed)]
    os.sched_setaffinity(0,set(cpus))
    os.nice(5)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    file = suite/'phases'/(stage+('_report' if report else '_dev')+f'_seed{seed}.json')
    if file.exists() and read_json(file)['status'] == 'completed':
        return
    started = now()
    write_json(file,{'status':'running','started':started,'pid':os.getpid(),'cpus':cpus})
    try:
        run_worker(suite,stage,seed,report)
        write_json(file,{'status':'completed','started':started,'finished':now(),'pid':os.getpid()})
    except BaseException as error:
        write_json(file,{'status':'failed','started':started,'failed':now(),'error':repr(error)})
        raise


def launch(suite,stage,report=False):
    from longtail.io import read_json,write_json
    plan = read_json(suite/'plan.json')
    tag = stage+('_report' if report else '_dev')
    (suite/'logs').mkdir(exist_ok=True)
    jobs = []
    for seed in plan['seeds']:
        log = (suite/'logs'/f'{tag}_seed{seed}.log').open('a')
        cmd = [sys.executable,'-u',str(Path(__file__).resolve()),'--suite',str(suite),'--worker',stage,'--seed',str(seed)]
        if report:
            cmd.append('--report')
        proc = subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
        jobs.append((seed,proc,log))
    while True:
        rows = []
        for seed,proc,_ in jobs:
            code = proc.poll()
            row = {'seed':seed,'pid':proc.pid,'state':'running' if code is None else 'completed' if code == 0 else 'failed'}
            phase = suite/'phases'/f'{tag}_seed{seed}.json'
            if phase.exists():
                row['phase'] = read_json(phase)
            row['training'] = []
            for file in (suite/stage/f'seed{seed}').glob('history_*.json'):
                history = read_json(file)
                if history:
                    row['training'].append({'method':file.stem.removeprefix('history_'),**history[-1]})
            rows.append(row)
        progress = {'updated':now(),'stage':stage,'role':'report' if report else 'development','jobs':rows,
                    'running_jobs':sum(r['state'] == 'running' for r in rows),'completed_jobs':sum(r['state'] == 'completed' for r in rows),
                    'failed_jobs':sum(r['state'] == 'failed' for r in rows)}
        write_json(suite/'progress.json',progress)
        if not progress['running_jobs']:
            break
        time.sleep(4)
    for _,_,log in jobs:
        log.close()
    if progress['failed_jobs']:
        raise RuntimeError('Worker failure: '+str([r for r in rows if r['state'] == 'failed']))


def render(suite):
    from longtail.io import read_json
    plan = read_json(suite/'plan.json')
    html = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>大校准池与小比例分层配对结果</title>'+STYLE+'<h1>大校准池与小比例定向配对CPU验证</h1><p><a href="plan.html">冻结方案</a></p><p>Exact为整组标签完全正确率；五种子均值，非模型集成。其他四类阈值始终固定，头副本只修改SawFM/MFT logits。</p>'
    if (suite/'pipeline_status.json').exists():
        html += '<p>流水线状态：'+escape(read_json(suite/'pipeline_status.json')['status'])+'</p>'
    for stage in STAGES:
        file = suite/stage/'summary.json'
        html += '<h2>'+stage+'</h2>'
        if not file.exists():
            html += '<p>等待完成。</p>'
            continue
        result = read_json(file)
        selection = result['selection']
        html += '<p>check选择：'+escape(selection['selected'])+'；共同决策规则：'+escape(selection['selected_rule'])+'</p>'
        rows = []
        for model,rules in result['report_means'].items():
            for rule,m in rules.items():
                if stage == 'targeted' and rule not in [selection['selected_rule'],'f1_8064']:
                    continue
                rows.append([model,rule]+[f'{100*m[k]:.3f}' for k in ['exact','macro_f1','saw_weak_recall','mft_weak_recall','focus_weak_aupr','focus_fpr']])
        html += table(['模型','决策','Exact%','MacroF1%','Saw弱召回%','MFT弱召回%','两目标弱AP%','两目标FPR%'],rows)
        for name,comp in result['comparisons'].items():
            d = comp['deltas']['exact']
            w = comp['deltas']['focus_weak_recall']
            html += '<p>'+escape(name)+f'：Exact差{100*d["mean"]:+.3f}pp，条件95%区间[{100*d["paired95"][0]:+.3f},{100*d["paired95"][1]:+.3f}]；两目标弱召回差{100*w["mean"]:+.3f}pp。整体Exact门槛'+('通过' if comp['exact_passed'] else '未全部通过')+'，弱召回权衡门槛'+('通过' if comp['weak_tradeoff_passed'] else '未全部通过')+'。</p>'
        for role,models in result.get('supplementary',{}).items():
            rule = selection['selected_rule']
            html += '<h3>'+role+'（未参与选型）</h3>'+table(['模型','Exact%','Saw弱召回%','MFT弱召回%'],[[m]+[f'{100*v[rule][k]:.3f}' for k in ['exact','saw_weak_recall','mft_weak_recall']] for m,v in models.items() if rule in v])
    html += '<p>误报预算只在calibration拟合，报告上的实际FPR独立测量；不可视为严格等测试误报率的召回比较。统计区间仅为固定数据池上的五种子条件区间，次要比较未校正多重检验。数据来自同一Python物理仿真器，尚无实测验证。</p></html>'
    (Path(plan['report_output'])/'analysis.html').write_text(html)


def audit(suite):
    import numpy as np
    import h5py
    from longtail.io import read_json,write_json,sha256
    from longtail.targeted_validation import check_report_gate,metric,KEYS
    from longtail.calibration_tradeoff import fit_policies,TARGETS
    from longtail.data.dataset import read_metadata
    from longtail.data.power_truth import read_power_truth
    from longtail.data.one_bit import pair
    from longtail.data.signals import Simulator
    plan = read_json(suite/'plan.json')
    for source in plan['source_manifest']:
        assert sha256(suite/'source_snapshot'/source['path']) == source['sha256']
    for path,digest in plan['data_hashes'].items():
        assert sha256(path) == digest
    metrics_count,decisions_count,models = 0,0,[]
    for stage in STAGES:
        check_report_gate(suite,stage)
        frozen = read_json(suite/stage/'frozen.json')
        for seed in plan['seeds']:
            dev = read_json(suite/'phases'/f'{stage}_dev_seed{seed}.json')
            report = read_json(suite/'phases'/f'{stage}_report_seed{seed}.json')
            assert dev['status'] == report['status'] == 'completed'
            assert dev['finished'] <= frozen['frozen_at'] <= report['started']
            output = suite/stage/f'seed{seed}'
            cal = dict(np.load(output/'predictions_calibration.npz'))
            decisions = read_json(output/'decisions.json')
            logits = {m:cal['logits_'+m] for m in decisions['models']}
            refs = plan['references'][str(seed)]
            locked = {m:refs['old' if m == 'old' else 'new']['thresholds'] for m in logits}
            replay = fit_policies(cal['targets'],logits,cal['weak_mask'],locked)
            assert replay == decisions
            decisions_count += 1
            for file in output.glob('metrics_*.json'):
                stored = read_json(file)
                role = stored['role']
                pred = dict(np.load(output/('predictions_'+role+'.npz')))
                if role == 'pair_report':
                    info = read_json(plan['data']['pair_manifest'])['files']['development_test']
                    data_file,indices = Path(info['path']),np.arange(info['samples'])
                elif role == 'ood_wide':
                    info = plan['data']['shifts']['ood_wide']
                    data_file,indices = Path(info['path']),np.arange(info['samples'])
                else:
                    manifest = Path(plan['data']['manifests'][stage])
                    data_file = manifest.parent/'development.h5'
                    indices = np.asarray(read_json(suite/'splits'/(stage+'.json'))['roles'][role])
                y,ids,_ = read_metadata(data_file)
                truth = read_power_truth(data_file,y)
                np.testing.assert_array_equal(pred['targets'],y[indices])
                np.testing.assert_array_equal(pred['identities'],ids[indices])
                np.testing.assert_array_equal(pred['weak_mask'],truth['weak_mask'][indices])
                np.testing.assert_allclose(pred['relative_power_db'],truth['relative_power_db'][indices],rtol=0,atol=0,equal_nan=True)
                for model,rules in stored['metrics'].items():
                    if model not in ['new','old']:
                        np.testing.assert_array_equal(pred['logits_'+model][:,[0,1,3,5]],pred['logits_new'][:,[0,1,3,5]])
                    for rule,expected in rules.items():
                        measured = metric(pred,pred['logits_'+model],decisions['models'][model]['thresholds'][rule])
                        assert all(np.isclose(measured[k],expected[k],rtol=0,atol=1e-12) for k in KEYS)
                        assert measured['per_label'] == expected['per_label']
                        metrics_count += 1
            if stage == 'targeted':
                for method in ['natural','pair1','pair2']:
                    state = read_json(output/(method+'_status.json'))
                    assert state['status'] == 'completed' and state['optimizer_steps'] == 6000
                    assert sha256(output/(method+'.pt')) == state['checkpoint_sha256']
                    history = read_json(output/('history_'+method+'.json'))
                    assert len(history) == 24
                    assert all(h['optimizer_steps'] == (i+1)*250 and h['natural_views']+h['paired_views'] == 8000 for i,h in enumerate(history))
                    dose = {'natural':0,'pair1':1,'pair2':2}[method]
                    np.testing.assert_array_equal(np.sum([h['pair_stratum_counts'] for h in history],axis=0),np.full(6,1000*dose))
                    models.append({'seed':seed,'method':method,'steps':6000})
    # Independently regenerate one waveform per target/gap/role, not only metadata.
    pair_manifest = read_json(plan['data']['pair_manifest'])
    generator = read_json(Path(plan['data_root'])/'generation_plan.json')
    simulator = Simulator(generator['physics'])
    backgrounds, regenerated = [],0
    for tag,path in plan['data']['manifests'].items():
        with h5py.File(Path(path).parent/'development.h5') as f:
            backgrounds.extend(f['receiver_noise_seed'][:].ravel().tolist())
    with h5py.File(plan['data']['shifts']['ood_wide']['path']) as f:
        backgrounds.extend(f['receiver_noise_seed'][:].ravel().tolist())
    for role,info in pair_manifest['files'].items():
        with h5py.File(info['path']) as f:
            y = f['label_selected'][:].T
            targets = f['pair_target'][:].ravel()
            bins = f['target_gap_bin'][:].ravel()
            seed = f['sample_seed'][:].ravel()
            backgrounds.extend(f['receiver_noise_seed'][0,::2].tolist())
            for target in [2,4]:
                for gap_bin in range(3):
                    row = int(np.flatnonzero((targets[::2] == target)&(bins[::2] == gap_bin))[0]*2)
                    code = int(y[row].astype(int)@(1 << np.arange(6)))
                    _,positive,negative,_,error = pair(simulator,code,target,int(seed[row]),generator['classes'])
                    np.testing.assert_array_equal(negative,f['iq_complex'][:,row])
                    np.testing.assert_array_equal(positive['iq'],f['iq_complex'][:,row+1])
                    assert error < 1e-12
                    regenerated += 1
    assert len(np.unique(backgrounds)) == len(backgrounds)
    for path in plan['data']['checked_identity_files']:
        with h5py.File(path) as f:
            assert not np.intersect1d(backgrounds,f['receiver_noise_seed'][:].ravel()).size
    result = {'status':'passed','new_models':len(models),'actual_optimizer_steps':sum(m['steps'] for m in models),
              'metrics_replayed':metrics_count,'calibration_bundles_replayed':decisions_count,
              'raw_waveform_pairs_regenerated':regenerated,'source_data_weights_verified':True,
              'other_four_logits_and_thresholds_preserved':True,'report_barriers_passed':True,'models':models}
    write_json(suite/'completion_audit.json',result)
    write_json(Path(plan['report_output'])/'completion_audit.json',result)
    return result


def export_selection(suite):
    from longtail.io import read_json,write_json,sha256
    plan = read_json(suite/'plan.json')
    selection = read_json(suite/'targeted/selection.json')
    candidate,rule = selection['selected'],selection['selected_rule']
    models = []
    for seed in plan['seeds']:
        ref = plan['references'][str(seed)]['new']
        item = {'seed':seed,'frozen_new_reference':ref,'selected':candidate,'decision_rule':rule,
                'thresholds':read_json(suite/'targeted'/f'seed{seed}/decisions.json')['models'][candidate]['thresholds'][rule]}
        if candidate != 'new':
            path = suite/'targeted'/f'seed{seed}'/(candidate+'.pt')
            item['target_head_copy'] = {'path':str(path),'sha256':sha256(path),'replaces_columns':[2,4]}
        models.append(item)
    result = {'status':'check-selected; inspect independent report acceptance','models':models,
              'five_seeds_are_separate_models_not_ensemble':True,'reference_weight_files_modified':False}
    write_json(suite/'selected_models.json',result)
    write_json(Path(plan['report_output'])/'selected_models.json',result)


def run(suite):
    from longtail.io import read_json,write_json
    from longtail.targeted_validation import select,freeze,summarize
    import fcntl
    lock = (suite/'coordinator.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    started = now()
    write_json(suite/'pipeline_status.json',{'status':'running','started':started,'pid':os.getpid()})
    try:
        for stage in STAGES:
            if (suite/stage/'summary.json').exists():
                continue
            launch(suite,stage)
            select(suite,stage)
            freeze(suite,stage)
            launch(suite,stage,report=True)
            summary = summarize(suite,stage)
            write_json(Path(read_json(suite/'plan.json')['report_output'])/(stage+'_results.json'),summary)
            render(suite)
        export_selection(suite)
        result = audit(suite)
        write_json(suite/'pipeline_completed.json',{'status':'completed','started':started,'finished':now(),'audit':result})
        write_json(suite/'pipeline_status.json',{'status':'completed','started':started,'finished':now(),'pid':os.getpid()})
        render(suite)
    except BaseException as error:
        write_json(suite/'pipeline_status.json',{'status':'failed','started':started,'failed':now(),'error':repr(error)})
        render(suite)
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite',type=Path,required=True)
    p.add_argument('--data',type=Path)
    p.add_argument('--output',type=Path)
    p.add_argument('--prepare',action='store_true')
    p.add_argument('--worker',choices=STAGES)
    p.add_argument('--seed',type=int)
    p.add_argument('--report',action='store_true')
    a = p.parse_args()
    if a.prepare:
        prepare(a.suite.resolve(),a.data.resolve(),a.output.resolve())
    elif a.worker:
        worker(a.suite.resolve(),a.worker,a.seed,a.report)
    else:
        run(a.suite.resolve())
