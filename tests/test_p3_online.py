"""Real tiny Omni + native LTX: live gradients, no leakage, resume, DDP parity."""
import copy
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from omni_av_p21.config import LoRAConfig
from omni_av_p2.checkpoint import capture_rng
from omni_av_p3_online.model import (OnlinePlanner, OnlineTraining, OnlineConfig,
    GenerationConnector, sample_mask, select_contexts)
from omni_av_p3_online.checkpoint import save, resume, initialize
from omni_av_p3.distributed import sha
from test_p3 import tiny, layout
from test_p2 import native_ltx


def make(stage="A", warmup=0, opened=True):
    base = tiny()
    planner = OnlinePlanner(base.thinker, base.cfg, "lora")
    planner.thinker.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    renderer, strategy, prepared = native_ltx()
    model = OnlineTraining(planner, renderer, strategy, stage,
        OnlineConfig(planner_warmup_steps=warmup), GenerationConnector(32, 32, 32), LoRAConfig(rank=2, alpha=2))
    if opened:
        with torch.no_grad():
            for mod in (model.connector.video, model.connector.audio):
                mod[-1].weight.normal_(0, .02)
            for decoder in planner.decoders:
                decoder.net.final_layer.linear.weight.normal_(0, .02)
                decoder.net.final_layer.adaLN_modulation[-1].weight.normal_(0, .02)
    return model, prepared


def inputs(seed=11):
    torch.manual_seed(seed)
    return layout(), torch.randn(1, 6, 32)


def has_grad(parameters):
    return any(p.grad is not None and p.grad.abs().sum() > 0 for p in parameters)


def test_exact_context_selection_prefix_and_visible_modality_only():
    data, _ = inputs()
    prefix = torch.randn(1, 4, 32, requires_grad=True)
    targets = torch.randn(1, 6, 32, requires_grad=True)
    mask = torch.tensor([[False, True, True, True, False, True]])
    c = select_contexts(prefix, targets, data, mask)
    torch.testing.assert_close(c.video, torch.cat([prefix, targets[:, :1]], 1))
    torch.testing.assert_close(c.audio, torch.cat([prefix, targets[:, 4:5]], 1))
    (c.video.sum() + c.audio.sum()).backward()
    assert prefix.grad.eq(2).all()
    assert targets.grad[0, [1, 2, 3, 5]].eq(0).all()
    assert targets.grad[0, [0, 4]].eq(1).all()
    all_mask = select_contexts(prefix, targets, data, data.valid)
    torch.testing.assert_close(all_mask.video, prefix)
    torch.testing.assert_close(all_mask.audio, prefix)


def test_masks_reproducible_and_handle_padding_and_singletons():
    data, _ = inputs()
    cfg = OnlineConfig()
    for seed in range(10):
        mask = sample_mask(data, cfg, seed)
        assert mask[:, :3].any()
        assert mask[:, 3:].any()
        torch.rand(20)
        assert torch.equal(mask, sample_mask(data, cfg, seed))
    assert any(torch.equal(sample_mask(data, cfg, seed), data.valid) for seed in range(10))
    valid = torch.tensor([[True, False, False, True, False, False]])
    small = replace(data, valid=valid)
    assert torch.equal(sample_mask(small, cfg, 7), valid)


def test_masked_gt_cannot_leak_into_hidden_or_av_loss():
    model, prepared = make()
    data, target = inputs()
    mask = torch.tensor([[False, True, True, True, False, True]])
    changed = target.clone()
    changed[mask] += 1000
    a = model.planner.readout(data, mask, target)
    b = model.planner.readout(data, mask, changed)
    for x, y in zip(a, b):
        torch.testing.assert_close(x, y, rtol=0, atol=0)
    av1 = model(data, target, prepared, 7, 1, mask)["av"]
    av2 = model(data, changed, prepared, 7, 1, mask)["av"]
    torch.testing.assert_close(av1, av2, rtol=0, atol=0)
    # Prefix is causal and cannot absorb GT target tokens even when visible.
    prefix_changed = model.planner.readout(data, mask, target + 20)[0]
    torch.testing.assert_close(a[0], prefix_changed, rtol=0, atol=0)


def test_one_omni_forward_and_separate_av_fm_gradient_paths():
    model, prepared = make()
    data, target = inputs()
    mask = torch.tensor([[False, True, True, True, False, True]])
    calls = []
    hook = model.planner.thinker.model.register_forward_hook(lambda *args: calls.append(1))
    values = model(data, target, prepared, 7, 1, mask)
    hook.remove()
    assert len(calls) == 1
    values["av"].backward()
    lora = [p for n, p in model.planner.named_parameters() if "lora_" in n]
    assert has_grad(lora) and has_grad(model.connector.parameters()) and has_grad(model.adapters.parameters())
    assert not has_grad(model.planner.decoders.parameters())
    assert not has_grad(model.planner.connectors.parameters())
    model.zero_grad(set_to_none=True)
    values = model(data, target, prepared, 7, 1, mask)
    (values["feature_video"] + values["feature_audio"]).backward()
    assert has_grad(lora) and has_grad(model.planner.decoders.parameters())
    assert has_grad(model.planner.connectors.parameters())
    assert not has_grad(model.connector.parameters()) and not has_grad(model.adapters.parameters())


@pytest.mark.parametrize("stage,warmup,step", [("C0", 0, 1), ("A", 100, 100), ("A", 100, 101)])
def test_stage_freezing_and_base_immutability(stage, warmup, step):
    model, prepared = make(stage, warmup)
    data, target = inputs()
    original = {n: p.detach().clone() for n, p in model.planner.named_parameters()}
    bases = [(p, p.detach().clone()) for m in (model.planner, model.renderer)
             for p in m.parameters() if not p.requires_grad]
    opt = torch.optim.AdamW(model.optimizer_groups(.001, .001, .001, .001))
    opt.zero_grad(set_to_none=True)
    values = model(data, target, prepared, 83, step)
    values["loss"].backward()
    opt.step()
    if not model.joint(step):
        assert not has_grad(model.planner.parameters())
        for n, p in model.planner.named_parameters():
            torch.testing.assert_close(p, original[n], rtol=0, atol=0)
        assert values["feature_video"] == 0 and values["feature_audio"] == 0
    else:
        assert has_grad(model.planner_parameters)
    for p, initial in bases:
        assert p.grad is None
        torch.testing.assert_close(p, initial, rtol=0, atol=0)
    model.assert_ownership()
    assert not any(k.startswith(("planner.", "renderer.")) for k in model.state_dict())


def test_zero_initialized_connector_opens_on_first_update():
    model, prepared = make("A", opened=False)
    data, target = inputs()
    model(data, target, prepared, 7, 1)["loss"].backward()
    assert has_grad(model.connector.video[-1].parameters())
    assert has_grad(model.planner.decoders[0].net.final_layer.linear.parameters())


def test_strict_resume_reproduces_update_and_transition(tmp_path):
    model, prepared = make()
    data, target = inputs()
    opt = torch.optim.AdamW(model.optimizer_groups(.001, .001, .001, .001))
    def update(step):
        opt.zero_grad(set_to_none=True)
        model(data, target, prepared, 7, step)["loss"].backward()
        opt.step()
    update(1)
    ident = dict(world_size=1, manifest_sha256="manifest", planner_sha256="planner")
    cp = tmp_path / "a.pt"
    save(cp, model, opt, 1, ident, [capture_rng()])
    update(2)
    expected = copy.deepcopy(model.state_dict())
    assert resume(cp, model, opt, ident)["step"] == 1
    update(2)
    for k, v in model.state_dict().items():
        torch.testing.assert_close(v, expected[k], atol=0, rtol=0)
    with pytest.raises(ValueError, match="identity"):
        resume(cp, model, opt, dict(ident, world_size=2))
    c0, _ = make("C0")
    cp0 = tmp_path / "c0.pt"
    save(cp0, c0, torch.optim.AdamW(c0.optimizer_groups(.001, .001, .001, .001)), 2, ident, [capture_rng()])
    a, _ = make("A")
    assert initialize(cp0, a, "manifest", sha(cp0), "planner") == 2
    for k, v in c0.state_dict().items():
        torch.testing.assert_close(v, a.state_dict()[k], rtol=0, atol=0)
    assert all(not adapter.b.weight.any() for adapter in a.adapters)
    with pytest.raises(ValueError, match="identity"):
        resume(cp0, a, opt, dict(ident, stage="A"))


def test_generation_reads_current_planner_without_gt_argument():
    model, _ = make()
    model.eval()
    data, _ = inputs()
    condition = model.predicted_conditions(data, 17, planning_steps=2)
    assert condition.video.shape == (1, 7, 32) and condition.audio.shape == (1, 7, 32)
    assert torch.isfinite(condition.video).all()
    with pytest.raises(TypeError):
        model.predicted_conditions(data, 17, targets=torch.randn(1, 6, 32))
    with torch.no_grad():
        model.planner.modality_embedding.add_(1)
    changed = model.predicted_conditions(data, 17, planning_steps=2)
    assert not torch.equal(changed.video, condition.video)


def test_online_conditions_work_with_cfg_and_no_gt_vae_tail():
    from omni_av_p2.ltx import append_conditions
    from omni_av_p3_staged.sampling import guided_sample
    model, prepared = make()
    model.eval()
    data, _ = inputs()
    negative = dict(video=torch.zeros_like(prepared.video.context),
                    audio=torch.zeros_like(prepared.audio.context), mask=prepared.video.context_mask)
    prepared = append_conditions(prepared, model.predicted_conditions(data, 17, planning_steps=2))
    keep = ~prepared.video_loss_mask
    altered = replace(prepared,
        video=replace(prepared.video, latent=torch.where(keep[..., None], prepared.video.latent,
                                                        prepared.video.latent + 1000)),
        audio=replace(prepared.audio, latent=prepared.audio.latent + 1000))
    sigmas = torch.tensor([1., .5, 0.])
    torch.manual_seed(91)
    a = guided_sample(model.renderer, prepared, negative, steps=2, sigmas=sigmas)
    torch.manual_seed(91)
    b = guided_sample(model.renderer, altered, negative, steps=2, sigmas=sigmas)
    for name in ("video", "audio"):
        assert torch.isfinite(a[name]).all()
        torch.testing.assert_close(a[name], b[name], rtol=0, atol=0)
    torch.testing.assert_close(a["video"][keep], prepared.video.latent[keep], rtol=0, atol=0)


def _worker(rank, path, output):
    from contextlib import nullcontext
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method="file://" + path, rank=rank, world_size=2)
    try:
        model, prepared = make(warmup=1)
        wrapped = DDP(model, find_unused_parameters=True, broadcast_buffers=False)
        opt = torch.optim.SGD(model.optimizer_groups(.0001, .0001, .0001, .0001))
        for step in (1, 2):
            opt.zero_grad(set_to_none=True)
            for micro in range(2):
                seed = step * 100 + micro * 2 + rank
                data, target = inputs(seed)
                with wrapped.no_sync() if micro == 0 else nullcontext():
                    (wrapped(data, target, prepared, seed, step)["loss"] / 2).backward()
            opt.step()
        torch.save(model.state_dict(), Path(output) / f"rank{rank}.pt")
    finally:
        dist.destroy_process_group()


def test_two_rank_warmup_to_joint_accumulation_matches_serial(tmp_path):
    import torch.multiprocessing as mp
    torch.set_num_threads(1)
    mp.spawn(_worker, args=(str(tmp_path / "rendezvous"), str(tmp_path)), nprocs=2, join=True)
    r0 = torch.load(tmp_path / "rank0.pt", weights_only=True)
    r1 = torch.load(tmp_path / "rank1.pt", weights_only=True)
    for k in r0:
        torch.testing.assert_close(r0[k], r1[k], rtol=0, atol=0)
    model, prepared = make(warmup=1)
    opt = torch.optim.SGD(model.optimizer_groups(.0001, .0001, .0001, .0001))
    for step in (1, 2):
        opt.zero_grad(set_to_none=True)
        for ordinal in range(4):
            seed = step * 100 + ordinal
            data, target = inputs(seed)
            (model(data, target, prepared, seed, step)["loss"] / 4).backward()
        opt.step()
    for k, v in model.state_dict().items():
        torch.testing.assert_close(v, r0[k], rtol=1e-5, atol=1e-7)
