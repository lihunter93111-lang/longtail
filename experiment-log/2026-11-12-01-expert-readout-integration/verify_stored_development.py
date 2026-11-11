"""Check existing development identities against all later audited raw pools."""
from common import *
from audit_identity_registry import REGISTRY,overlap


def verify():
    registry=read(REGISTRY/'registry.json');assert registry['passed']
    assert registry['source_sha256']==sha(HERE/'audit_identity_registry.py')
    by_hash={record['sha256']:record for record in registry['records']};fresh={}
    for split in CONFIG['splits']:
        manifest=role_manifest(split,'development');record=by_hash[manifest['sha256']]
        assert sha(manifest['path'])==record['sha256'] and sha(record['index_path'])==record['index_sha256']
        assert record['identities']==4032;fresh[split]=(record,np.load(record['index_path']))
    excluded={record['sha256'] for record,index in fresh.values()};old=[]
    for record in registry['records']:
        if record['sha256'] in excluded:continue
        assert sha(record['index_path'])==record['index_sha256'];old.append((record,np.load(record['index_path'])))
    rows=[]
    for split,(record,indices) in fresh.items():
        for previous,other in old:rows.append({'split':split,'reference':previous['path'],**overlap(indices,other)})
    splits=list(CONFIG['splits']);between=overlap(fresh[splits[0]][1],fresh[splits[1]][1])
    result={'passed':True,'role':'development','registry_sha256':sha(REGISTRY/'registry.json'),'old_pool_count':len(old),
            'checks':rows,'between_development_splits':between,'source_sha256':sha(Path(__file__)),
            'scope':'Directly verified stored-seed indexes include pools generated after initial development acceptance; no models changed.',
            'time':stamp()}
    write(ROOT/'development_stored_identity_verification.json',result)
    print('DEVELOPMENT_STORED_IDENTITIES_VERIFIED',len(rows),'plus_cross_split',flush=True)


if __name__=='__main__':verify()
