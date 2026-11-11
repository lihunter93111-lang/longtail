"""Materialize randomized quota identities and audit matrix pools before training."""
from common import *
from prepare_roles import identities

def main():
    config=read(HERE/'matrix_protocol.json');assert sha(config['source_quota_acceptance'])==config['source_quota_sha256']
    all_sources=set()
    for root in [PREVIOUS.parent.parent,MARGINAL,ROOT]:
        for p in root.rglob('generation_receipt.json'):
            candidate=Path(read(p).get('path',''))
            if candidate.is_file() and candidate.suffix=='.mat':all_sources.add(candidate.resolve())
    for split in CONFIG['splits']:
        for row in read(ROOT/split/'development/preflight.json')['audit']:all_sources.add(Path(row['path']).resolve())
    audited={}
    for split in CONFIG['splits']:
        candidate=Path(config['candidate_paths'][split]);selection=ROOT/'matrix'/split/'data/selection_pool/closedset_6class_train.mat'
        for path in [candidate,selection]:
            sample,noise,comp=identities(path);rows=[]
            physical=base.candidates(path.parent.parent,path.parent.name,LABELS)
            np.testing.assert_array_equal(sample,physical['seeds'])
            registry=np.array([1,2,3,4,6,8])
            for i,seeds in enumerate(physical['components']):
                ids=registry[np.flatnonzero(physical['y'][i])]
                np.testing.assert_array_equal(seeds,(sample[i]+2246822519+ids*2654435761)%(1<<32))
            for other in sorted(all_sources):
                if path.resolve()==other:continue
                old_s,old_n,old_c=identities(other)
                overlaps=[len(np.intersect1d(sample,old_s)),len(np.intersect1d(noise,old_n)),sum(len(np.intersect1d(v,old_c.get(k,[]))) for k,v in comp.items())]
                assert overlaps==[0,0,0],(str(path),str(other),overlaps)
                rows.append({'path':str(other),'overlaps':overlaps})
            audited[str(path)]={'sha256':sha(path),'rows':len(sample),'actual_components_verified':True,'audit':rows}
        with h5py.File(candidate,'r') as f:labels=np.asarray(f['known_label'],dtype=np.float32).T
        with h5py.File(selection,'r') as f:select_y=np.asarray(f['known_label'],dtype=np.float32).T
        codes=labels.astype(int)@(1<<np.arange(6));select_codes=select_y.astype(int)@(1<<np.arange(6))
        assert len(labels)==30240 and len(select_y)==2016
        rng=np.random.default_rng(config['fit_permutation_seeds'][split]);priority=rng.permutation(len(labels))
        early=[];cal=[]
        for code in range(1,64):
            rows=np.flatnonzero(select_codes==code);assert len(rows)==32
            early.extend(rows[:16].tolist());cal.extend(rows[16:].tolist())
        for name,cell in config['cells'].items():
            chosen=[]
            for code,n in enumerate(cell['counts'],1):chosen.extend(priority[codes[priority]==code][:n].tolist())
            chosen_set=set(chosen);indices=[int(i) for i in priority if int(i) in chosen_set]
            actual=np.bincount(codes[indices],minlength=64)[1:]
            np.testing.assert_array_equal(actual,cell['counts']);np.testing.assert_array_equal(labels[indices].sum(0),cell['label_margins'])
            assert len(indices)==8000 and len(set(indices))==8000
            manifest={'role':'matrix_fit_and_selection','fit_path':str(candidate),'fit_sha256':sha(candidate),'fit_A':indices,
                'dev_path':str(selection),'dev_sha256':sha(selection),'early_stop':early,'calibration':cal,
                'development_manifest':str(ROOT/split/'development/manifest.json'),'counts':actual.tolist(),
                'label_margins':cell['label_margins'],'permutation_seed':config['fit_permutation_seeds'][split],'matrix_protocol_sha256':sha(HERE/'matrix_protocol.json')}
            write(ROOT/'matrix'/split/name/'data/manifest.json',manifest)
    write(ROOT/'matrix/data_acceptance.json',{'passed':True,'pools':audited,'cells':list(config['cells']),'split_count':2,'role_note':'Each cohort shares candidate pool among four cells; selection and evaluation identities are separate','time':stamp()})
    print('MATRIX_DATA_ACCEPTED',flush=True)

if __name__=='__main__':main()
