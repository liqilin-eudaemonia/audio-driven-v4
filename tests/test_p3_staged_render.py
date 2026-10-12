import copy
from pathlib import Path
import json
import pytest
import torch
from omni_av_p3_staged.model import GenerationConnector, RenderTraining
from omni_av_p3_staged.checkpoint import save, resume, initialize
from omni_av_p3_staged.data import join_rows, condition_mode, condition_seed
from omni_av_p21.config import LoRAConfig
from omni_av_p3.distributed import GlobalOrder, sha
from omni_av_p2.checkpoint import capture_rng
from omni_av_p3_staged.quality import QUALITY_VERSION
from test_p2 import native_ltx


def model(stage):
    renderer,strategy,prepared=native_ltx()
    return RenderTraining(renderer,strategy,stage,GenerationConnector(8,32,32),LoRAConfig(rank=2,alpha=2)),prepared


def hidden():
    return {k:torch.randn(1,n,8,requires_grad=True) for k,n in [('video',3),('audio',2)]}


@pytest.mark.parametrize('stage',['C0','A'])
def test_real_ltx_gradients_and_freezing(stage):
    torch.manual_seed(10);m,p=model(stage);h=hidden()
    frozen=[(v,v.detach().clone()) for v in m.renderer.parameters() if not v.requires_grad]
    optimizer=torch.optim.AdamW(m.optimizer_groups(.001,.001))
    for i in range(3):
        optimizer.zero_grad();loss,lv,la=m(h,p);loss.backward()
        assert torch.isfinite(loss) and torch.isfinite(lv) and torch.isfinite(la)
        assert all(v.grad is None for v in h.values())
        if i:assert m.connector.video[0].weight.grad.abs().sum()>0
        assert m.connector.video[-1].weight.grad.abs().sum()>0
        if stage=='A':assert any(a.b.weight.grad.abs().sum()>0 for a in m.adapters)
        optimizer.step();m.assert_ownership()
    for v,original in frozen:
        assert v.grad is None
        torch.testing.assert_close(v,original,rtol=0,atol=0)
    assert len(m.targets)==(8 if stage=='A' else 0)
    assert not any(k.startswith('renderer.') for k in m.state_dict())


def test_strict_resume_matches_next_optimizer_update(tmp_path):
    torch.manual_seed(11);m,p=model('A');h=hidden();opt=torch.optim.AdamW(m.optimizer_groups(.001,.001))
    opt.zero_grad();m(h,p)[0].backward();opt.step()
    identity=dict(world_size=1,manifest_sha256='test');path=tmp_path/'step.pt'
    save(path,m,opt,1,identity,[capture_rng()])
    opt.zero_grad();m(h,p)[0].backward();opt.step();expected=copy.deepcopy(m.state_dict())
    d=resume(path,m,opt,identity);assert d['step']==1
    opt.zero_grad();m(h,p)[0].backward();opt.step()
    for k,v in m.state_dict().items():torch.testing.assert_close(v,expected[k],rtol=0,atol=0)
    with pytest.raises(ValueError,match='Resume'):resume(path,m,opt,dict(world_size=2,manifest_sha256='test'))
    c0,_=model('C0')
    with pytest.raises(ValueError,match='Resume'):resume(path,c0,opt,identity)


def test_stage_transition_needs_bound_gate_and_zero_lora(tmp_path):
    c0,_=model('C0');opt=torch.optim.AdamW(c0.optimizer_groups(.001,.001));cp=tmp_path/'c0.pt'
    save(cp,c0,opt,100,dict(world_size=1,manifest_sha256='data'),[capture_rng()])
    a,_=model('A')
    with pytest.raises(ValueError,match='passing evaluation'):initialize(cp,a,'data')
    gate=tmp_path/'gate.json';gate.write_text(json.dumps(dict(quality_version=QUALITY_VERSION,passed=True,manifest_sha256='data',checkpoint_sha256=sha(cp))))
    assert initialize(cp,a,'data',gate)==100
    for adapter in a.adapters:assert not adapter.b.weight.any()
    for k,v in c0.connector.state_dict().items():torch.testing.assert_close(a.connector.state_dict()[k],v)
    gate.write_text(json.dumps(dict(passed=True,manifest_sha256='wrong',checkpoint_sha256=sha(cp))))
    with pytest.raises(ValueError):initialize(cp,a,'data',gate)


def test_explicit_source_transition_preserves_optimizer_and_rejects_recipe_change(tmp_path):
    torch.manual_seed(11);m,p=model('A');h=hidden();opt=torch.optim.AdamW(m.optimizer_groups(.001,.001))
    opt.zero_grad();m(h,p)[0].backward();opt.step()
    old=dict(world_size=1,manifest_sha256='test',code={'old':'hash'},recipe={'steps':1000})
    cp=tmp_path/'step.pt';save(cp,m,opt,1,old,[capture_rng()])
    seal=tmp_path/'inputs.json';seal.write_text('{}')
    new=dict(old,code={'new':'hash'},input_integrity_sha256=sha(seal))
    certificate=tmp_path/'transition.json'
    certificate.write_text(json.dumps(dict(version='p3.audit-fix-transition.v1',from_checkpoint_sha256=sha(cp),
        from_identity=old,target_code=new['code'],input_integrity_seal=str(seal))))
    opt.zero_grad();m(h,p)[0].backward();opt.step();expected=copy.deepcopy(m.state_dict())
    with pytest.raises(ValueError,match='Resume'):resume(cp,m,opt,new)
    assert resume(cp,m,opt,new,certificate)['step']==1
    opt.zero_grad();m(h,p)[0].backward();opt.step()
    for k,v in m.state_dict().items():torch.testing.assert_close(v,expected[k],rtol=0,atol=0)
    with pytest.raises(ValueError,match='Resume'):resume(cp,m,opt,dict(new,world_size=2),certificate)
    with pytest.raises(ValueError,match='Resume'):resume(cp,m,opt,dict(new,recipe={'steps':500}),certificate)


def rows():
    n=[dict(sample_id='a',split='train',duration_s=1.)]
    t=[dict(sample_id='a',split='train',text='a.pt')]
    e=[dict(observation=dict(sample_id='a'),split='train',duration_s=1.,target_height=640,target_width=1440)]
    m=[dict(sample_id='a',split='train',target_height=640,target_width=1440,target_frames=25,fps=30.)]
    return n,t,e,m


def test_full_manifest_checks_split_identity_and_original_aspect():
    n,t,e,m=rows();out=join_rows(n,t,e,m,'train',1)
    assert out[0]['metadata']['target_width']==1440
    with pytest.raises(ValueError):join_rows(n,t+t,e,m,'train',1)
    with pytest.raises(ValueError):join_rows(n,t,e,m,'train',2)
    t[0]['split']='dev'
    with pytest.raises(ValueError):join_rows(n,t,e,m,'train',1)
    n,t,e,m=rows();m[0]['target_width']=640
    with pytest.raises(ValueError,match='geometry'):join_rows(n,t,e,m,'train',1)
    n,t,e,m=rows();m[0]['target_frames']=1
    with pytest.raises(ValueError,match='window'):join_rows(n,t,e,m,'train',1)


def test_condition_selection_and_full_epoch_resume():
    order=GlobalOrder(91009,42)
    ids=[order[i] for i in range(91009)]
    assert len(set(ids))==91009
    assert sum(condition_mode(42,i)=='pred25' for i in range(144000))==72000
    restarted=GlobalOrder(91009,42)
    for i in (72,91008,91009,143999):assert order[i]==restarted[i]
    assert condition_seed(42,'a')==condition_seed(42,'a')!=condition_seed(42,'b')


def test_no_silent_joint_training_or_padded_readout():
    with pytest.raises(ValueError,match='Only C0/A'):model('B')
    c=GenerationConnector(8,32,32);h=hidden();h['audio']=torch.zeros(2,3,8)
    with pytest.raises(ValueError,match='single-sample'):c(h)


def _ddp_worker(rank, rendezvous, output, stage):
    from contextlib import nullcontext
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method='file://'+rendezvous,rank=rank,world_size=2)
    try:
        torch.manual_seed(101);m,p=model(stage);wrapped=DDP(m,broadcast_buffers=False)
        opt=torch.optim.SGD(m.optimizer_groups(.001,.001))
        for step in range(2):
            opt.zero_grad()
            for micro in range(2):
                torch.manual_seed(200+step*4+micro*2+rank)
                h=hidden()
                with wrapped.no_sync() if micro==0 else nullcontext():
                    (wrapped(h,p)[0]/2).backward()
            opt.step()
        torch.save(m.state_dict(),Path(output)/f'rank{rank}.pt')
    finally:dist.destroy_process_group()


@pytest.mark.parametrize('stage',['C0','A'])
def test_two_cpu_rank_accumulation_matches_serial(tmp_path,stage):
    import torch.multiprocessing as mp
    torch.set_num_threads(1)
    mp.spawn(_ddp_worker,args=(str(tmp_path/'rendezvous'),str(tmp_path),stage),nprocs=2,join=True)
    rank0=torch.load(tmp_path/'rank0.pt',weights_only=True);rank1=torch.load(tmp_path/'rank1.pt',weights_only=True)
    for k in rank0:torch.testing.assert_close(rank0[k],rank1[k],rtol=0,atol=0)
    torch.manual_seed(101);m,p=model(stage);opt=torch.optim.SGD(m.optimizer_groups(.001,.001))
    for step in range(2):
        opt.zero_grad()
        for ordinal in range(4):
            torch.manual_seed(200+step*4+ordinal)
            (m(hidden(),p)[0]/4).backward()
        opt.step()
    for k,v in m.state_dict().items():torch.testing.assert_close(rank0[k],v,rtol=1e-5,atol=1e-7)
