"""Check completed CPU-run provenance, paired identities and frozen decisions."""
import argparse
from pathlib import Path
import numpy as np
from longtail.io import read_json, write_json, sha256
from longtail.data.dataset import read_metadata
from longtail.data.splits import validate_split


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', type=Path, required=True)
    args = parser.parse_args();suite = args.suite.resolve()
    root = Path(__file__).resolve().parents[1]
    plan = read_json(suite / 'plan.json')
    for entry in plan['source_manifest']:
        assert sha256(root / entry['path']) == entry['sha256'], entry['path']
        assert sha256(suite / 'source_snapshot' / entry['path']) == entry['sha256']
    addendum = read_json(suite / 'evaluation_addendum.json')
    for path, digest in addendum['code_sha256'].items():
        assert sha256(root / path) == digest, path
    reference = {};checked = [];hashes = {}
    paths = ['check', 'calibration', 'development_test'] + [
        f'development_test/gap_{g}' for g in (0,6,12,18,24)]
    for job in plan['jobs']:
        output = Path(job['output']);phase = read_json(suite / 'phases' / (job['id']+'.json'))
        status = read_json(output / 'status.json');history = read_json(output / 'history.json')
        assert phase['phase'] == status['status'] == 'completed'
        assert status['device'] == 'cpu' and status['world_size'] == 1
        assert status['parameters'] == job['parameters']
        assert [r['epoch'] for r in history] == list(range(1,9))
        assert all(r['samples_seen']==8000 and r['sampler_padding']==0 for r in history)
        assert all(r['selection_split']=='check' for r in history)
        digest = sha256(output / 'best.pt');hashes[job['id']] = digest
        assert phase['checkpoint_sha256'] == digest
        split = read_json(output / 'split.json');manifest = read_json(output / 'dataset.json')
        validate_split(split, manifest)
        assert split['sha256'] == plan['protocol']['data_split_sha256']
        config = read_json(job['config']);data_root = Path(config['data']['manifest']).parent
        train_y, train_ids, _ = read_metadata(data_root / manifest['files']['train']['path'])
        dev_y, dev_ids, _ = read_metadata(data_root / manifest['files']['development']['path'])
        assert len(train_y)==8000 and not np.intersect1d(train_ids,dev_ids).size
        decision = read_json(output / 'decision.json')
        assert decision['fit_role'] == 'calibration'
        for path in paths:
            e = read_json(output / 'evaluation' / path / 'metrics.json')
            assert e['checkpoint_sha256'] == digest and e['decision'] == decision
            assert e['split_sha256'] == split['sha256']
            z = dict(np.load(output / 'evaluation' / path / 'predictions.npz'))
            role = path.split('/')[0]
            np.testing.assert_array_equal(z['indices'],split['roles'][role])
            np.testing.assert_array_equal(z['identities'],dev_ids[z['indices']])
            np.testing.assert_array_equal(z['targets'],dev_y[z['indices']])
            assert np.isfinite(z['logits']).all()
            if path not in reference:reference[path] = z
            for key in z:
                if key != 'logits':np.testing.assert_array_equal(z[key], reference[path][key])
        fpr = read_json(output / 'evaluation/fpr_controlled/metrics.json')
        assert fpr['checkpoint_sha256']==digest and fpr['threshold_fit_role']=='calibration'
        for d in read_json(output / 'evaluation/fpr_controlled/decisions.json').values():
            assert d['checkpoint_sha256']==digest and d['fit_role']=='calibration'
            np.testing.assert_array_equal(d['calibration_identities'],reference['calibration']['identities'])
            assert max(d['achieved_calibration_fpr']) <= d['target_negative_fpr']
        if job['arm'] != 'baseline':
            d = read_json(output / 'query_diagnostics.json')
            assert d['checkpoint_sha256']==digest and d['role']=='check'
        checked.append(job['id'])
    for seed in (2050,2051):
        d = read_json(suite / 'diagnostics' / f'memory_ablation_seed{seed}.json')
        assert d['checkpoint_sha256']==hashes[f'multiscale_seed{seed}']
        assert d['role']=='check' and d['global_path_unchanged'] and not d['thresholds_refitted']
    completion = read_json(suite / 'completion.json')
    assert completion['jobs_completed']==8 and not completion['failures']
    result = {'passed':True,'jobs':checked,'frozen_core_source_files':len(plan['source_manifest']),
              'frozen_evaluations_checked':len(paths)*len(checked),'checkpoints':hashes,
              'identity_overlap_train_development':0,'all_prediction_metadata_paired':True,
              'all_primary_thresholds_unchanged':True,'all_checkpoints_unchanged':True,
              'fpr_fit_role':'calibration','memory_diagnostic_role':'check'}
    write_json(suite / 'quality_checks.json',result)
    print(result)


if __name__=='__main__':main()
