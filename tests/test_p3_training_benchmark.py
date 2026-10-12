from dataclasses import dataclass,replace
import importlib.util
from pathlib import Path
from types import MethodType
import pytest
import torch

spec=importlib.util.spec_from_file_location('benchmark',Path(__file__).resolve().parents[1]/'scripts/pipeline3_training_benchmark.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


@dataclass
class Modality:
    x:torch.Tensor
    context:torch.Tensor


class Block(torch.nn.Module):
    def forward(self,video,audio,perturbations):
        v=video.x+.1*torch.tanh(video.x+video.context+.3*audio.x)
        a=audio.x+.1*torch.tanh(audio.x+audio.context+.2*video.x)
        return replace(video,x=v),replace(audio,x=a)


@pytest.mark.parametrize('group,every,offload',[(1,1,False),(4,1,False),(4,1,True),(1,2,False)])
def test_checkpoint_candidates_preserve_context_gradients(group,every,offload):
    torch.manual_seed(9)
    base=[torch.randn(1,3,4,dtype=torch.float64) for _ in range(4)]
    results=[]
    for candidate in [False,True]:
        values=[x.clone().requires_grad_() for x in base]
        video=Modality(values[0],values[2]);audio=Modality(values[1],values[3])
        renderer=torch.nn.Module();renderer.transformer_blocks=torch.nn.ModuleList([Block() for _ in range(7)])
        if candidate:
            renderer._benchmark_config=dict(group=group,checkpoint_every=every,offload=offload)
            renderer.forward=MethodType(module.checkpoint_blocks,renderer)
            v,a=renderer(video,audio,None)
        else:
            v,a=video,audio
            for block in renderer.transformer_blocks:v,a=block(v,a,None)
        loss=v.x.square().mean()+a.x.square().mean()
        results.append((v.x.detach(),a.x.detach(),*torch.autograd.grad(loss,values)))
    for ref,actual in zip(*results):torch.testing.assert_close(ref,actual,rtol=1e-10,atol=1e-10)
