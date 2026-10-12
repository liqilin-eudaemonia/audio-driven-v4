import importlib.util
from pathlib import Path
import pytest
import torch

spec=importlib.util.spec_from_file_location('diagnose',Path(__file__).resolve().parents[1]/'scripts/pipeline3_diagnose.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_separate_gradients_reconstruct_combined_backward_and_conflict():
    v=torch.nn.Parameter(torch.tensor([1.,2.]));a=torch.nn.Parameter(torch.tensor([3.,4.]))
    names=[('video.weight',v),('audio.weight',a)]
    lv=(v+a).square().sum();la=(v-2*a).square().sum()
    vg=torch.autograd.grad(lv,[v,a],retain_graph=True)
    ag=torch.autograd.grad(la,[v,a],retain_graph=True)
    combined=torch.autograd.grad(lv+la,[v,a],retain_graph=True)
    weighted=torch.autograd.grad(2*lv+la,[v,a])
    report=module.gradient_summary(names,vg,ag)
    for i,branch in enumerate(['video','audio']):
        assert report[branch]['combined_grad_norm']==pytest.approx(float(combined[i].norm()),rel=1e-6)
        assert report[branch]['video_weight2_grad_norm']==pytest.approx(float(weighted[i].norm()),rel=1e-6)
    assert report['video']['cosine']<0
    assert report['audio']['cosine']>0


def test_wrong_condition_preserves_token_count():
    wrong={'video':torch.randn(1,3,4),'audio':torch.randn(1,8,2)}
    ref={'video':torch.randn(1,5,4),'audio':torch.randn(1,8,2)}
    matched=module.match_tokens(wrong,ref)
    assert matched['video'].shape==ref['video'].shape
    assert matched['audio'] is wrong['audio']
