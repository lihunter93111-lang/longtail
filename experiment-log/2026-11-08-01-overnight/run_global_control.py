"""Parameter-matched mean/std control for the learned top8 residual readout."""
import argparse
import sys
from pathlib import Path

import torch

import run_variant
from cjlt.models.phase_multiscale import LocalEvidenceStatisticsBypass1d

base = run_variant.base
original_builder = base.build_backbone


class GlobalStatisticsBypass(LocalEvidenceStatisticsBypass1d):
    def forward_local(self,temporal):
        mean = temporal.mean(dim=-1)
        standard_deviation = torch.sqrt(torch.clamp(temporal.square().mean(dim=-1)-mean.square(),min=1e-5))
        return self.local_projection(torch.cat([mean,standard_deviation],dim=1))


def build_control(model_name,num_classes,width=16,dropout=.2):
    if model_name!="dfms_iq_global_statistics_bypass":
        return original_builder(model_name,num_classes=num_classes,width=width,dropout=dropout)
    model = original_builder("dfms_iq_local_topk8",num_classes=num_classes,width=width,dropout=dropout)
    current = model.local_statistics_bypass
    random_state = torch.get_rng_state()
    replacement = GlobalStatisticsBypass(width*8,width*8,topk=8,dropout=dropout,residual_bound=current.residual_bound)
    replacement.load_state_dict(current.state_dict())
    torch.set_rng_state(random_state)
    model.local_statistics_bypass = replacement
    return model


if __name__=="__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output",type=Path,required=True)
    args,_ = parser.parse_known_args()
    args.output.mkdir(parents=True,exist_ok=True)
    protocol = {"wrapper_sha256":base.sha(Path(__file__)),
        "prompt_sha256":base.sha(Path(__file__).with_name("04B_PARAMETER_CONTROL_GOAL.txt")),
        "replacement":"maximum/top8 mean to mean/std; same trainable tensors and dropout calls"}
    path = args.output/"wrapper_protocol.json"
    if path.exists():
        assert run_variant.read(path)==protocol
    else:
        base.write(path,protocol)
    base.build_backbone = build_control
    run_variant.main()
