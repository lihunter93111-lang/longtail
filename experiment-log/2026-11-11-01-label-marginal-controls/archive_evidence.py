"""Copy lightweight evidence only; preserve external raw data and models."""
from common import *
import shutil

def main():
    output=HERE/'results';copied=[]
    def copy(path,relative=None):
        dest=output/(relative or path.relative_to(ROOT));dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,dest);assert sha(path)==sha(dest)
        copied.append({'original':str(path),'archive':dest.relative_to(HERE).as_posix(),'sha256':sha(dest)})
    for path in ROOT.glob('*.json'):copy(path)
    for path in ROOT.glob('*.log'):copy(path,Path('logs')/path.name)
    for top in ['analysis','future_matrix','split1','split2']:
        for path in (ROOT/top).rglob('*.json'):
            if path.name=='generation_manifest.json':continue
            copy(path)
    for split in CONFIG['splits']:
        for group in ['ordinary_expert','weighted_expert']:
            folder=PREVIOUS/split/'expert_training'/group
            for path in folder.rglob('*.json'):
                copy(path,Path('reference_dependencies')/split/group/path.relative_to(folder))
        for name in ['data/manifest.json','w32/protocol.json']:
            path=BASELINES/split/name
            if path.exists():copy(path,Path('reference_dependencies')/split/'main'/name)
        for seed in CONFIG['main_seeds']:
            for name in ['receipt.json','history.json']:
                path=BASELINES/split/'w32'/f'M_{seed}'/name
                if path.exists():copy(path,Path('reference_dependencies')/split/'main'/f'M_{seed}'/name)
        for name in ['manifest.json','preflight.json']:
            path=PREVIOUS/split/'fusion_fit'/name
            if path.exists():copy(path,Path('reference_dependencies')/split/'fusion_fit'/name)
    write(output/'archive_provenance.json',{'files':copied,'excluded':'MAT, PTH, NPZ, recovery states, verbose generation manifests; external replay ZIP has weights and new confirmation data',
        'time':stamp(),'source_sha256':sha(Path(__file__))})
    print('LIGHTWEIGHT_ARCHIVE_COPIED',len(copied),flush=True)

if __name__=='__main__':main()
