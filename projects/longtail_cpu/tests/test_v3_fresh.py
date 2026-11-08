import numpy as np
from longtail.data.signals import Simulator,substream_seed,REGISTRY_IDS
from longtail.data.v3_fresh import compose_v3


def test_v3_fresh_independent_jnr_truth_and_identity_are_reproducible():
    sim=Simulator();codes=['FMB','AMA','SawFM','SinFM','MFT','ISRJ']
    a=compose_v3(sim,codes,3400000123);b=compose_v3(sim,codes,3400000123)
    np.testing.assert_array_equal(a['iq'],b['iq'])
    np.testing.assert_allclose(a['power'],a['jnr'],atol=1e-10)
    assert a['iq'].shape==(3000,) and np.isfinite(a['iq']).all()
    assert ((a['jnr']>=10)&(a['jnr']<=30)).all() and len(np.unique(a['jnr']))==6
    assert a['noise_seed']==substream_seed(3400000123,2)
    assert a['component_seeds']==[substream_seed(3400000123,1,REGISTRY_IDS[c]) for c in codes]
    c=compose_v3(sim,codes,3400000124)
    assert not np.array_equal(a['iq'],c['iq'])
