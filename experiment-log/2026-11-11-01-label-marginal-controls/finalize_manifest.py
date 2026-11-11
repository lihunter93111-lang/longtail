"""Freeze the lightweight delivery bytes, excluding only self-referential receipts."""
from common import *

if __name__=='__main__':
    repo=Path(CONFIG['repository']);paths=[]
    for p in HERE.rglob('*'):
        if not p.is_file() or '__pycache__' in p.parts:continue
        if p.name in ['delivery_manifest.json','delivery_verification.json']:continue
        assert p.suffix not in ['.mat','.pth','.pt','.npz','.npy','.h5','.hdf5'],p
        paths.append(p)
    paths.append(HERE.parent/'README.md')
    files=[{'path':p.relative_to(repo).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(paths)]
    write(HERE/'delivery_manifest.json',{'date':'2026-11-11','base_commit':CONFIG['base_commit'],'branch':'codex/data-marginal-20261111',
        'files':files,'excluded':'delivery manifest itself, post-verification receipt, bytecode; raw data/weights/logits external',
        'time':stamp(),'scientific_completion_audit_sha256':sha(HERE/'results/completion_audit.json')})
    print('DELIVERY_MANIFEST',len(files),sum(r['bytes'] for r in files),flush=True)
