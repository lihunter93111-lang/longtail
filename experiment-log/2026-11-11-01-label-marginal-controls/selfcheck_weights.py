"""Regression checks for scalar versus per-label weighting and matched marginals."""
from common import *
from torch.nn import functional as F

bits=((np.arange(1,64)[:,None]>>np.arange(6))&1).astype(np.float32)
y=np.repeat(bits,QUOTAS['A_counts'],axis=0)
marginal,combo,design=objective_weights(y)
zero_m,zero_c,_=objective_weights(y,0)
np.testing.assert_allclose(zero_m,1,atol=1e-7)
np.testing.assert_allclose(zero_c,1,atol=1e-7)
truth=torch.tensor(y,dtype=torch.float64)
torch.manual_seed(19)
z=torch.randn(y.shape,dtype=torch.float64,requires_grad=True)
raw=F.binary_cross_entropy_with_logits(z,truth,reduction='none')
old=(raw.mean(1)*torch.tensor(combo[:,0],dtype=torch.float64)).mean()
new=(raw*torch.tensor(combo,dtype=torch.float64)).mean()
torch.testing.assert_close(old,new,atol=1e-12,rtol=1e-12)
g1=torch.autograd.grad(old,z,retain_graph=True)[0]
g2=torch.autograd.grad(new,z)[0]
torch.testing.assert_close(g1,g2,atol=1e-12,rtol=1e-12)
# At common zero logits, total label bias gradients match by construction.
gm=((.5-y)*marginal).mean(0,dtype=np.float64)
gc=((.5-y)*combo).mean(0,dtype=np.float64)
np.testing.assert_allclose(gm,gc,atol=3e-8)
write(ROOT/'weight_selfcheck.json',{'passed':True,'identity_count':len(y),'scalar_vs_matrix_loss_error':float(abs(old-new)),'maximum_gradient_error':float(abs(g1-g2).max()),'label_bias_gradient_error':float(abs(gm-gc).max()),'zero_mixture_unit_weights':True,'source_sha256':sha(Path(__file__))})
print('WEIGHT_TESTS_PASSED')
