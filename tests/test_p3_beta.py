import copy
from dataclasses import replace
import pytest
import torch
from torch import nn
from omni_av_p3_online.model import OnlineConfig, OnlinePlanner, OnlineTraining, sample_mask
from omni_av_p3_online.checkpoint import save, resume, initialize
from omni_av_p3_online.runtime import update_learning_rates
from omni_av_p2.checkpoint import capture_rng
from omni_av_p3.distributed import sha
from test_p3 import tiny
from test_p3_online import make, inputs, has_grad


def planner_model():
    base=tiny(); planner=OnlinePlanner(base.thinker,base.cfg,'lora')
    model=OnlineTraining(planner,nn.Identity(),None,'P',OnlineConfig())
    with torch.no_grad():
        for d in planner.decoders:
            d.net.final_layer.linear.weight.normal_(0,.02)
            d.net.final_layer.adaLN_modulation[-1].weight.normal_(0,.02)
    return model


def test_beta_distribution_no_uniform_no_forced_visible_floor():
    layout,_=inputs()
    large=replace(layout,valid=torch.ones(1,2000,dtype=torch.bool),modalities=torch.tensor([[0]*1000+[1]*1000]))
    rates=[];high=0
    for seed in range(1000):
        mask=sample_mask(large,OnlineConfig(),seed)
        nv,na=int(mask[:,:1000].sum()),int(mask[:,1000:].sum())
        assert nv==na and nv>0
        rates.append(nv/1000);high+=nv/1000>.95
    assert .895 < sum(rates)/len(rates) < .925
    assert .35 < high/1000 < .45
    # A small sample can be all masked without an extra full-mask branch.
    assert any(sample_mask(layout,OnlineConfig(),i).all() for i in range(10))
    for invalid in [0,-1,float('nan'),float('inf')]:
        with pytest.raises(ValueError):OnlineConfig(mask_alpha=invalid)
    with pytest.raises(TypeError):OnlineConfig(mask_min=.5)


def test_p_updates_all_feature_groups_without_renderer_or_connector():
    model=planner_model();layout,target=inputs(); frozen=[(p,p.detach().clone()) for p in model.planner.parameters() if not p.requires_grad]
    opt=torch.optim.AdamW(model.optimizer_groups(.001,.001,.001,.001))
    assert {g['name'] for g in opt.param_groups}=={'omni_lora','feature_heads'}
    assert set(model.state_dict())=={k for k in model.state_dict() if k.startswith('planner_parameters.')}
    loss=model(layout,target,None,7,1,layout.valid.clone());loss['loss'].backward()
    assert loss['av']==0 and loss['feature_video']>0 and loss['feature_audio']>0
    assert all(has_grad(g['params']) for g in opt.param_groups)
    opt.step();model.assert_ownership()
    for p,old in frozen:torch.testing.assert_close(p,old,rtol=0,atol=0)
    recipe=dict(planner_lr=1e-5,heads_lr=1e-5,online=dict(planner_warmup_steps=100))
    update_learning_rates(opt,recipe,1,False,stage='P')
    assert all(g['lr']==pytest.approx(1e-5/50) for g in opt.param_groups)


def test_p_resume_and_exact_p_c0_a_transfer(tmp_path):
    model=planner_model();layout,target=inputs(); opt=torch.optim.AdamW(model.optimizer_groups(.001,.001,.001,.001))
    def update(step):
        opt.zero_grad(set_to_none=True);model(layout,target,None,17+step,step)['loss'].backward();opt.step()
    update(1); identity=dict(world_size=1,manifest_sha256='manifest',planner_sha256='planner')
    cp=tmp_path/'p.pt';save(cp,model,opt,1,identity,[capture_rng()]);expected_p=copy.deepcopy(model.state_dict())
    update(2);expected=copy.deepcopy(model.state_dict());resume(cp,model,opt,identity);update(2)
    for k,v in model.state_dict().items():torch.testing.assert_close(v,expected[k],rtol=0,atol=0)
    c0,_=make('C0');assert initialize(cp,c0,'manifest',sha(cp),'planner')==1
    for k,v in expected_p.items():torch.testing.assert_close(c0.state_dict()[k],v,rtol=0,atol=0)
    cp0=tmp_path/'c0.pt';save(cp0,c0,torch.optim.AdamW(c0.optimizer_groups(.001,.001,.001,.001)),2,identity,[capture_rng()])
    a,_=make('A');initialize(cp0,a,'manifest',sha(cp0),'planner')
    for k,v in c0.state_dict().items():torch.testing.assert_close(a.state_dict()[k],v,rtol=0,atol=0)
    assert all(not mod.b.weight.any() for mod in a.adapters)
    with pytest.raises(ValueError):resume(cp,model,opt,dict(identity,recipe='uniform'))


def test_allmask_joint_and_warmup_gradient_boundary():
    model,prepared=make('A',warmup=100);layout,target=inputs()
    for step in [100,101]:
        model.zero_grad(set_to_none=True)
        values=model(layout,target,prepared,19,step,layout.valid.clone());values['loss'].backward()
        assert has_grad(model.connector.parameters()) and has_grad(model.adapters.parameters())
        assert has_grad(model.planner_parameters)==(step==101)
        assert bool(values['feature_video']>0)==(step==101)
