"""Real tiny Omni backbone tests for masks, gradients, paired RNG and FM sampling."""
from dataclasses import replace

import pytest
import torch
from torch import nn

from omni_av_p3.experiment import digest, mean_teachers
from omni_av_p3.model import Config, Layout, Planner, content_mask


def tiny(arm="frozen"):
    from transformers.models.qwen2_5_omni.configuration_qwen2_5_omni import Qwen2_5OmniTextConfig
    from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import Qwen2_5OmniThinkerTextModel
    cfg = Qwen2_5OmniTextConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
        rope_scaling={"rope_type": "default", "mrope_section": [1, 1, 2]}, attention_dropout=0.)
    cfg._attn_implementation = "sdpa"
    torch.manual_seed(7)
    thinker = nn.Module()
    thinker.model = Qwen2_5OmniThinkerTextModel(cfg)
    model = Planner(thinker, Config(hidden_size=32, decoder_width=32, decoder_depth=2,
        diffusion_batch_mul=2, lora_rank=2, lora_alpha=4, planning_steps=5, denoising_steps=3), arm)
    return model


def layout():
    return Layout(torch.randn(1, 4, 32), torch.arange(4)[None, None].expand(3, 1, -1),
        torch.ones(1, 4, dtype=torch.bool), torch.tensor([[[0., 0, 0], [0., 0, 1], [1., 0, 0],
        [.02, 0, 0], [.06, 0, 0], [.10, 0, 0]]]), torch.tensor([[0, 0, 0, 1, 1, 1]]),
        torch.ones(1, 6, dtype=torch.bool))


def test_p3_common_initialization_and_mask_rng_are_paired():
    frozen, lora = tiny(), tiny("lora")
    common = lambda m: {k: v for k, v in m.trainable_state().items() if not k.startswith("thinker.")}
    assert digest(common(frozen)) == digest(common(lora))
    data = layout()
    for seed in range(10):
        a = content_mask(data, frozen.cfg, seed)
        torch.rand(1000)
        b = content_mask(data, lora.cfg, seed)
        assert torch.equal(a, b) and a[0, :3].any() and a[0, 3:].any()
    values = torch.randn(1, 6, 32)
    a = frozen.hidden(data, data.valid, values)
    b = lora.hidden(data, data.valid, values)
    torch.testing.assert_close(a, b)


@pytest.mark.parametrize("arm", ["frozen", "lora"])
def test_p3_adaln_zero_init_opens_gradients_and_preserves_frozen_weights(arm):
    model, data = tiny(arm), layout()
    model.thinker.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    frozen = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.01)
    targets = torch.randn(1, 6, 32)
    for _ in range(4):
        optimizer.zero_grad()
        sum(model.loss(data, targets, 83, all_mask=True).values()).backward()
        optimizer.step()
    assert model.mask_token.grad.abs().sum() > 0
    assert model.connectors[0][0].weight.grad.abs().sum() > 0
    if arm == "lora":
        assert any(p.grad is not None and p.grad.abs().sum() > 0
                   for n, p in model.named_parameters() if "lora_b" in n)
    for n, p in model.named_parameters():
        if n in frozen:
            assert p.grad is None
            torch.testing.assert_close(p, frozen[n], atol=0, rtol=0)


def test_p3_all_mask_hidden_cannot_read_gt_and_sampler_has_no_teacher_argument():
    model, data = tiny().eval(), layout()
    targets = torch.randn(1, 6, 32)
    torch.testing.assert_close(model.hidden(data, data.valid, targets),
                               model.hidden(data, data.valid, targets + 100))
    with pytest.raises(TypeError):
        Layout(**vars(data), targets=targets)
    with pytest.raises(TypeError):
        model.sample(data, 7, targets=targets)
    one, _ = model.sample(data, 7, steps=1)
    many, remaining = model.sample(data, 7)
    # Initially zero velocity: output should be the same position-specific noise.
    torch.testing.assert_close(one, many, atol=0, rtol=0)
    assert remaining[-1] == 0
    assert all(a >= b for a, b in zip(remaining, remaining[1:]))


def test_p3_prefix_cannot_attend_targets_and_target_attention_is_bidirectional():
    from omni_av.masking import hybrid_attention
    data = layout()
    mask = hybrid_attention(data.prefix_valid, data.valid, torch.float32)
    assert (mask[:, :, :4, 4:] < -1e20).all()
    assert (mask[:, :, 4:, 4:] == 0).all()
    assert (mask[:, :, 4:, :4] == 0).all()


def test_p3_euler_sign_with_exact_constant_velocity():
    model, data = tiny().eval(), layout()
    seed = 17
    noise = torch.randn(1, 6, 32, generator=torch.Generator().manual_seed(seed))
    target = torch.randn(1, 6, 32)
    class Velocity(nn.Module):
        def __init__(self, velocity):
            super().__init__()
            self.velocity = velocity
        def forward(self, x, t, c):
            return self.velocity
    for mod in (0, 1):
        model.decoders[mod].net = Velocity((noise - target)[data.modalities == mod])
    actual, _ = model.sample(data, seed, steps=1)
    torch.testing.assert_close(actual, target, atol=1e-6, rtol=1e-6)


def test_p3_mean_baseline_only_uses_supplied_train_samples():
    data = layout()
    targets = torch.zeros(1, 6, 32)
    targets[:, :3] = 2
    targets[:, 3:] = 5
    means = mean_teachers([dict(layout=vars(data), targets=targets)])
    assert torch.equal(means[0], torch.full((32,), 2.))
    assert torch.equal(means[1], torch.full((32,), 5.))


@pytest.mark.parametrize("arm", ["frozen", "lora"])
def test_p3_checkpoint_restoration_exact_and_rejects_missing_or_extra_parameters(arm):
    from omni_av_p3.diagnostics import restore
    model = tiny(arm)
    state = model.trainable_state()
    checkpoint = dict(arm=arm, state=state)
    with torch.no_grad():
        model.mask_token.add_(7)
    restore(model, checkpoint)
    assert digest(model.trainable_state()) == digest(state)
    with pytest.raises(ValueError, match="parameter set"):
        restore(model, dict(arm=arm, state={k: v for k, v in state.items() if k != "mask_token"}))
    with pytest.raises(ValueError, match="parameter set"):
        restore(model, dict(arm=arm, state={**state, "thinker.model.fake.weight": torch.zeros(1)}))
    with pytest.raises(ValueError, match="arm"):
        restore(model, dict(arm="invalid", state=state))
