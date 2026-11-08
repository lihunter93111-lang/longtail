"""Small trainable components over a frozen DFMS feature vector."""
import torch
from torch import nn


class FiveLabelHead(nn.Module):
    def __init__(self,weight,bias,dropout=.2):
        super().__init__();self.dropout=nn.Dropout(dropout);self.linear=nn.Linear(weight.shape[1],5)
        with torch.no_grad():self.linear.weight.copy_(weight[1:]);self.linear.bias.copy_(bias[1:])

    def forward(self,x):return self.linear(self.dropout(x))


class LocalResidual(nn.Module):
    def __init__(self,dimension=128,active=(True,True,True)):
        super().__init__();self.hidden=nn.Linear(dimension,32);self.output=nn.Linear(32,3)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
        self.register_buffer('active',torch.tensor(active,dtype=torch.float32))

    def forward(self,x):return self.output(torch.nn.functional.gelu(self.hidden(x)))*self.active


def residual_logits(base,residual):
    result=base.clone();result[:,[2,4,5]]+=residual
    return result


def pair_margin_loss(logits,targets,margin=.5):
    if len(logits)!=2*len(targets):raise ValueError('Expected consecutive negative/positive pairs')
    rows=torch.arange(len(targets),device=logits.device)
    delta=logits[2*rows+1,targets]-logits[2*rows,targets]
    return torch.relu(margin-delta).mean()
