"""Real LTX classes at small dimensions; CPU engineering tests, not pretrained quality."""
from dataclasses import replace

import torch

from omni_av.ltx_runtime import RendererBundle, render_loss, sample_latents, with_text
from omni_av.renderer import PlanningConnector, attach_plan, configure_renderer_lora, install_plan_attention


def tiny_ltx():
    from ltx_core.model.transformer.attention import AttentionFunction
    from ltx_core.model.transformer.model import LTXModel
    from ltx_trainer.timestep_samplers import UniformTimestepSampler
    from ltx_trainer.training_strategies.text_to_video import TextToVideoConfig, TextToVideoStrategy
    model = LTXModel(num_attention_heads=2, attention_head_dim=16, num_layers=2,
        cross_attention_dim=32, audio_num_attention_heads=2, audio_attention_head_dim=16,
        audio_cross_attention_dim=32, attention_type=AttentionFunction.PYTORCH)
    # LTX constructor leaves pretrained scale/shift tables uninitialized intentionally.
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "scale_shift_table" in name:
                p.normal_(0, .02)
    install_plan_attention(model, context_dim=8, width=8)
    model = configure_renderer_lora(model, rank=2, alpha=2)
    connector = PlanningConnector(32, 8, 8)
    bundle = RendererBundle(model, connector)
    strategy = TextToVideoStrategy(TextToVideoConfig(with_audio=True, i2av_only=True,
                                                    first_frame_conditioning_p=1))
    video = dict(latents=torch.randn(1, 128, 3, 2, 2), num_frames=torch.tensor([3]),
                 height=torch.tensor([2]), width=torch.tensor([2]), fps=torch.tensor([30.]))
    audio = dict(latents=torch.randn(1, 8, 8, 16), num_time_steps=torch.tensor([8]))
    text = dict(video=torch.randn(1, 4, 32), audio=torch.randn(1, 4, 32),
                mask=torch.ones(1, 4, dtype=torch.bool))
    prepared = strategy.prepare_training_inputs(with_text(dict(latents=video, audio_latents=audio), text),
                                                UniformTimestepSampler(.2, .8))
    return bundle, strategy, prepared


def test_real_ltx_zero_branch_gradient_and_mask():
    torch.manual_seed(9)
    bundle, strategy, prepared = tiny_ltx()
    bundle.train()
    bundle.renderer.get_base_model().set_gradient_checkpointing(True)
    hidden = torch.randn(1, 6, 32, requires_grad=True)
    valid = torch.tensor([[True, True, True, False, False, False]])
    with torch.no_grad():
        original = bundle.renderer(video=prepared.video, audio=prepared.audio, perturbations=None)
        condition = attach_plan(prepared, bundle.connector, hidden, valid)
        after = bundle.renderer(video=condition.video, audio=condition.audio, perturbations=None)
    assert all(torch.equal(a, b) for a, b in zip(original, after))
    opt = torch.optim.AdamW([p for p in bundle.parameters() if p.requires_grad], lr=.01)
    for _ in range(2):
        opt.zero_grad(set_to_none=True)
        hidden.grad = None
        condition = attach_plan(prepared, bundle.connector, hidden, valid)
        loss, _ = render_loss(bundle, strategy, condition)
        loss.backward()
        assert torch.isfinite(loss)
        opt.step()
    assert hidden.grad[:, :3].abs().sum() > 0
    assert hidden.grad[:, 3:].abs().sum() == 0
    assert all(p.grad is None for p in bundle.parameters() if not p.requires_grad)


def test_joint_small_omni_ltx_backward():
    from test_core import batch, tiny_planner
    torch.manual_seed(8)
    bundle, strategy, prepared = tiny_ltx()
    planner, data = tiny_planner(), batch()
    bundle.planner = planner
    bundle.train()
    opt = torch.optim.AdamW([p for p in bundle.parameters() if p.requires_grad], lr=.01)
    for step in range(2):
        opt.zero_grad(set_to_none=True)
        mask = torch.tensor([[True, False, True, True, False, True]])
        planned = planner(data, mask)
        condition = attach_plan(prepared, bundle.connector, planned["hidden"], planned["renderer_mask"])
        loss, _ = render_loss(bundle, strategy, condition)
        loss.backward()  # AV only: proves the link independently of planning loss.
        if step == 1:
            grads = [p.grad for n, p in planner.named_parameters() if "lora_b" in n]
            assert all(g is not None and g.abs().sum() > 0 for g in grads)
        opt.step()


def test_sampler_preserves_reference_and_rejects_target_tail_dependency():
    torch.manual_seed(2)
    bundle, _, prepared = tiny_ltx()
    first = sample_latents(bundle, prepared, steps=2)
    edited = prepared.video.latent.clone()
    edited[prepared.video_loss_mask] += 1000
    torch.manual_seed(2)
    # Reset seed immediately before each sampler so only unused target tail differs.
    first = sample_latents(bundle, prepared, steps=2)
    torch.manual_seed(2)
    second = sample_latents(bundle, replace(prepared, video=replace(prepared.video, latent=edited)), steps=2)
    assert all(torch.equal(first[k], second[k]) for k in first)
