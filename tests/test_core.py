from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from omni_av.checkpoint import initialize_delta, restore_checkpoint, save_checkpoint
from omni_av.config import PlanConfig
from omni_av.data import Observation, TextPlan, gt_text_plan
from omni_av.flow import FeatureFlow
from omni_av.lora import PlanningLinear, inject_planning_lora, native_reply_mode
from omni_av.masking import fill_schedule, hybrid_attention, sample_target_mask
from omni_av.planner import AVPlanner, PlanningBatch
from omni_av.renderer import PlanAttention


def tiny_planner():
    from transformers.models.qwen2_5_omni.configuration_qwen2_5_omni import Qwen2_5OmniTextConfig
    from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import Qwen2_5OmniThinkerTextModel
    config = Qwen2_5OmniTextConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
        rope_scaling={"rope_type": "default", "mrope_section": [1, 1, 2]}, attention_dropout=0.0)
    config._attn_implementation = "sdpa"
    thinker = nn.Module()
    thinker.model = Qwen2_5OmniThinkerTextModel(config)
    thinker.visual, thinker.audio_tower = nn.Identity(), nn.Identity()
    inject_planning_lora(thinker, rank=2, alpha=4)
    return AVPlanner(thinker, PlanConfig(hidden_size=32, flow_width=32, flow_depth=2,
        diffusion_batch_mul=2, lora_rank=2, lora_alpha=4, gradient_checkpointing=False))


def batch():
    return PlanningBatch(torch.randn(1, 4, 32), torch.arange(4)[None, None].expand(3, 1, -1),
        torch.ones(1, 4, dtype=torch.bool), torch.tensor([[False, False, True, True]]),
        torch.randn(1, 6, 32), torch.tensor([[[0., 0, 0], [0., 0, 1], [1., 0, 0],
            [0., 0, 0], [.5, 0, 0], [1., 0, 0]]]), torch.tensor([[0, 0, 0, 1, 1, 1]]),
        torch.ones(1, 6, dtype=torch.bool))


def test_hybrid_attention_and_padding():
    mask = hybrid_attention(torch.tensor([[True, True, False]]), torch.tensor([[True, False]]), torch.float32)
    assert mask[0, 0, 0, 1] < -1e20
    assert mask[0, 0, :3, 3:].max() < -1e20
    assert mask[0, 0, 3, 0] == 0
    assert mask[0, 0, 3, 2] < -1e20
    assert mask[0, 0, 4, 4] == 0


def test_native_lora_after_update_and_exception():
    torch.manual_seed(7)
    layer = PlanningLinear(nn.Linear(8, 8).requires_grad_(False), 2, 4)
    x = torch.randn(3, 8)
    baseline = layer.base(x).detach()
    opt = torch.optim.AdamW([layer.lora_a, layer.lora_b], lr=.1)
    layer(x).square().mean().backward()
    opt.step()
    assert not torch.equal(layer(x), baseline)
    with native_reply_mode(layer):
        assert torch.equal(layer(x), baseline)
    with pytest.raises(RuntimeError), native_reply_mode(layer):
        raise RuntimeError("test restore")
    assert layer.enabled and layer.training


def test_real_text_backbone_no_target_leak_and_gradients():
    torch.manual_seed(4)
    model, data = tiny_planner(), batch()
    mask = torch.tensor([[True, False, True, True, False, True]])
    result = model(data, mask)
    changed = model.hidden(replace(data, targets=data.targets + 100), mask)
    assert torch.allclose(result["hidden"][:, :4], changed[:, :4], atol=1e-6)
    result["loss"].backward()
    assert model.mask_token.grad.abs().sum() > 0
    assert all(p.grad is None for n, p in model.thinker.named_parameters() if not p.requires_grad)
    grads = [p.grad for n, p in model.named_parameters() if "lora_b" in n]
    assert all(g is not None and g.abs().sum() > 0 for g in grads)
    assert not result["renderer_mask"][0, 4]
    assert result["renderer_mask"][0, 5]


@pytest.mark.parametrize("n,steps", [(0, 4), (1, 8), (13, 8), (100, 3)])
def test_schedule(n, steps):
    visits = [i for start, end in fill_schedule(n, steps) for i in range(start, end)]
    assert visits == list(range(n))


def test_all_mask_inference_and_teacher_rejection():
    model, data = tiny_planner().eval(), batch()
    with pytest.raises(ValueError, match="empty target"):
        model.plan(data)
    empty = replace(data, targets=torch.zeros_like(data.targets))
    out = model.plan(empty, steps=3, denoising_steps=2)
    assert out["remaining"][-1] == 0
    assert torch.isfinite(out["features"]).all()
    assert out["renderer_mask"][0, 4:].all()


def test_flow_sign_with_exact_velocity():
    flow = FeatureFlow(3, 3, 8, 1)
    target = torch.tensor([[2., -1., 3.]])
    # Constant exact velocity derived from the identical initial noise draw.
    g = torch.Generator().manual_seed(9)
    noise = torch.randn(1, 3, generator=g)
    flow.forward = lambda x, sigma, c: noise - target
    out = flow.sample(target, steps=5, generator=torch.Generator().manual_seed(9))
    assert torch.allclose(out, target, atol=1e-6)


def test_mask_covers_modalities_and_valid_only():
    data = batch()
    data.valid[0, -1] = False
    for _ in range(10):
        mask = sample_target_mask(data.valid, data.coordinates[..., 0], data.modalities)
        assert not mask[0, -1]
        assert mask[0, :3].any() and mask[0, 3:5].any()


def test_plan_attention_zero_init_and_upstream_grad():
    torch.manual_seed(6)
    branch = PlanAttention(16, 8, 8, 2)
    context = torch.randn(1, 4, 8, requires_grad=True)
    x = torch.randn(1, 3, 16)
    mask = torch.tensor([[True, True, False, False]])
    assert torch.count_nonzero(branch(x, context, mask)) == 0
    opt = torch.optim.AdamW(branch.parameters(), lr=.01)
    for _ in range(2):
        opt.zero_grad()
        (branch(x, context, mask) - 1).square().mean().backward()
        opt.step()
    assert context.grad[:, :2].abs().sum() > 0
    assert context.grad[:, 2:].abs().sum() == 0


def test_checkpoint_restore_rng_and_new_file(tmp_path):
    model = nn.Linear(3, 3)
    opt = torch.optim.AdamW(model.parameters())
    model(torch.randn(2, 3)).sum().backward()
    opt.step()
    path = tmp_path / "step.pt"
    save_checkpoint(path, model, opt, 1, {"model": "tiny"})
    expected = torch.rand(5)
    with torch.no_grad():
        model.weight.add_(100)
    assert restore_checkpoint(path, model, opt, {"model": "tiny"}) == 1
    assert torch.equal(torch.rand(5), expected)
    with pytest.raises(FileExistsError):
        save_checkpoint(path, model, opt, 2, {})
    with pytest.raises(ValueError, match="contract"):
        restore_checkpoint(path, model, opt, {})


def test_text_and_observation_contracts():
    text = TextPlan('He said "hello".', "calm", "nods", "hmm")
    assert TextPlan.parse(text.serialize()) == text
    with pytest.raises(ValueError):
        TextPlan.parse("Reply: hi")
    with pytest.raises(ValueError):
        Observation("id", "audio", "image", 0, 5, 4)
    with pytest.raises(TypeError):
        Observation("id", "audio", "image", 0, 1, 4, target_video="bad")
    gt = {"avatar_plan": {"sample_id": "id", "response": {"present": True, "text": "hi"}}}
    plan = gt_text_plan(gt, "id")
    assert plan.behavior == "" and plan.delivery == ""


def test_stage_initialization_requires_complete_parameter_set(tmp_path):
    model = nn.Linear(3, 3)
    opt = torch.optim.AdamW(model.parameters())
    path = tmp_path / "stage.pt"
    save_checkpoint(path, model, opt, 1, {"model": "tiny"})
    target = nn.Linear(3, 3)
    initialize_delta(path, target, "tiny")
    assert torch.equal(model.weight, target.weight)
    with pytest.raises(ValueError, match="parameter set"):
        initialize_delta(path, target, "tiny", expected_names={"weight"})
