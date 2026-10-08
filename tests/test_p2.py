"""Pipeline 2: reference parity, shared final conditions and frozen-renderer gradients."""
from dataclasses import replace
import importlib.util
from pathlib import Path
import json

import pytest
import torch

from omni_av.ltx_runtime import with_text
from omni_av_p2.checkpoint import export_connector, load, save
from omni_av_p2.config import Config
from omni_av_p2.connector import ConditionInputs, FusionConnector
from omni_av_p2.decoder import SimpleMLPAdaLN
from omni_av_p2.ltx import append_conditions, av_loss
from omni_av_p2.model import Extractor


def config():
    return Config(hidden_size=32, target_dim=16, video_dim=32, audio_dim=32,
                  fusion_width=32, fusion_depth=1, fusion_heads=4, decoder_width=32,
                  decoder_depth=2, diffusion_batch_mul=2)


def condition(layer_ids=(8, 18, 28, 35)):
    return ConditionInputs(torch.randn(1, len(layer_ids), 7, 32), torch.tensor([[True] * 5 + [False] * 2]),
        torch.tensor([[[.25, 0, 0], [.25, 0, 1], [1.25, 0, 0]]]),
        torch.tensor([[[.125, 0, 0], [.375, 0, 0]]]),
        torch.ones(1, 3, dtype=torch.bool), torch.ones(1, 2, dtype=torch.bool), torch.tensor(layer_ids))


def native_ltx():
    from ltx_core.model.transformer.model import LTXModel
    from ltx_core.model.transformer.attention import AttentionFunction
    from ltx_trainer.timestep_samplers import UniformTimestepSampler
    from ltx_trainer.training_strategies.text_to_video import TextToVideoConfig, TextToVideoStrategy
    model = LTXModel(num_attention_heads=2, attention_head_dim=16, num_layers=1,
        cross_attention_dim=32, audio_num_attention_heads=2, audio_attention_head_dim=16,
        audio_cross_attention_dim=32, attention_type=AttentionFunction.PYTORCH)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "scale_shift_table" in name:
                p.normal_(0, .02)
    model.requires_grad_(False)
    model.set_gradient_checkpointing(True)
    strategy = TextToVideoStrategy(TextToVideoConfig(with_audio=True, i2av_only=True, first_frame_conditioning_p=1))
    av = dict(latents=dict(latents=torch.randn(1, 128, 2, 2, 2), num_frames=torch.tensor([2]),
              height=torch.tensor([2]), width=torch.tensor([2]), fps=torch.tensor([30.])),
              audio_latents=dict(latents=torch.randn(1, 8, 8, 16), num_time_steps=torch.tensor([8])))
    text = dict(video=torch.randn(1, 4, 32), audio=torch.randn(1, 4, 32), mask=torch.ones(1, 4, dtype=torch.bool))
    prepared = strategy.prepare_training_inputs(with_text(av, text), UniformTimestepSampler(.2, .8))
    return model, strategy, prepared


def test_p2_decoder_matches_bernini_forward_and_backward(monkeypatch):
    path = Path("/private/liqi_workspace/project_code/Bernini/bernini/models/diffloss_fm.py")
    if not path.exists():
        pytest.skip("Local upstream Bernini reference unavailable")
    monkeypatch.setenv("USE_MLP_NORM_IN_RESBLOCK_OF_FM", "False")
    spec = importlib.util.spec_from_file_location("p2_reference_bernini", path)
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    reference = original.SimpleMLPAdaLN(16, 32, 16, 24, 2).eval()
    # Nonzero weights ensure comparison exercises conditions/gates, not trivial zero output.
    with torch.no_grad():
        for param in reference.parameters():
            param.normal_(0, .1)
    actual = SimpleMLPAdaLN(16, 32, 16, 24, 2).eval()
    actual.load_state_dict(reference.state_dict(), strict=True)
    x, t = torch.randn(5, 16), torch.rand(5) * 1000
    a, b = torch.randn(5, 24, requires_grad=True), None
    b = a.detach().clone().requires_grad_(True)
    expected, found = reference(x, t, a), actual(x, t, b)
    torch.testing.assert_close(expected, found, atol=1e-6, rtol=1e-6)
    expected.sum().backward()
    found.sum().backward()
    torch.testing.assert_close(a.grad, b.grad, atol=1e-6, rtol=1e-6)


def test_p2_connector_padding_and_zero_init():
    cfg, data = config(), condition()
    model = FusionConnector(cfg)
    out = model(data)
    assert out.video.count_nonzero() == 0 and out.audio.count_nonzero() == 0
    model = FusionConnector(replace(cfg, zero_init_connector=False)).eval()
    original = model(data)
    changed = data.hidden.clone()
    changed[:, :, 5:] += 10000
    after = model(replace(data, hidden=changed))
    torch.testing.assert_close(original.video, after.video)
    torch.testing.assert_close(original.audio, after.audio)
    with pytest.raises(TypeError):
        ConditionInputs(**vars(data), targets=torch.randn(1))


def test_p2_frozen_ltx_and_auxiliary_update_same_final_condition():
    torch.manual_seed(12)
    model, data = Extractor(config()), condition()
    renderer, strategy, prepared = native_ltx()
    targets = dict(video=torch.randn(1, 3, 16), audio=torch.randn(1, 2, 16),
                   video_valid=torch.tensor([[True, True, False]]), audio_valid=torch.ones(1, 2, dtype=torch.bool))
    opt = torch.optim.AdamW(model.parameters(), lr=.01)
    versions = [p._version for p in renderer.parameters()]
    for step in range(3):
        opt.zero_grad()
        u = model(data)
        inputs = append_conditions(prepared, u)
        assert torch.equal(inputs.video.context[:, -3:], u.video)
        assert torch.equal(inputs.audio.context[:, -2:], u.audio)
        assert inputs.video.context_mask[:, -3:].all()  # GT availability must not change U mask.
        lav, _ = av_loss(renderer, strategy, inputs)
        losses = model.feature_losses(u, targets)
        for mod in ("video", "audio"):
            ga = torch.autograd.grad(lav, getattr(u, mod), retain_graph=True)[0]
            gf = torch.autograd.grad(losses[mod], getattr(u, mod), retain_graph=True)[0]
            assert ga.abs().sum() > 0
            if step > 1:  # Bernini zeros BOTH output and conditioning modulation: two updates to open path.
                assert gf.abs().sum() > 0
        (lav + sum(losses.values())).backward()
        if step > 0:
            assert model.connector.memory.projection.weight.grad.abs().sum() > 0
        opt.step()
    assert all(not p.requires_grad and p.grad is None and p._version == v for p, v in zip(renderer.parameters(), versions))


def test_p2_checkpoint_connector_only_and_rng(tmp_path):
    model = Extractor(config()).eval()
    opt = torch.optim.AdamW(model.parameters())
    path = tmp_path / "p2.pt"
    save(path, model, opt, 3, {"config": "test"})
    expected_rng = torch.rand(6)
    load(path, model, {"config": "test"}, opt)
    assert torch.equal(expected_rng, torch.rand(6))
    deploy = FusionConnector(config()).eval()
    load(path, deploy, {"config": "test"}, connector_only=True)
    deployment = tmp_path / "connector.pt"
    export_connector(path, deployment)
    deployed = load(deployment, deploy, {"config": "test"}, connector_only=True)
    assert "optimizer" not in deployed
    assert all(k.startswith("connector.") for k in deployed["model"])
    data = condition()
    torch.testing.assert_close(model(data).video, deploy(data).video)
    with pytest.raises(ValueError):
        load(path, deploy, {"config": "wrong"}, connector_only=True)
    with pytest.raises(FileExistsError):
        save(path, model, opt, 4, {})


def test_p2_feature_targets_never_change_condition():
    model, data = Extractor(replace(config(), zero_init_connector=False)).eval(), condition()
    u = model(data)
    targets = dict(video=torch.randn(1, 3, 16), audio=torch.randn(1, 2, 16),
                   video_valid=data.video_valid, audio_valid=data.audio_valid)
    model.feature_losses(u, targets)
    targets["video"] += 100
    model.feature_losses(u, targets)
    after = model(data)
    assert torch.equal(u.video, after.video) and torch.equal(u.audio, after.audio)


def test_p2_output_geometry_uses_external_budget_only():
    from omni_av_p2.inference import output_geometry
    shape = output_geometry(8, 30, 256, 256)
    assert shape["frames"] == 233
    assert (shape["frames"] - 1) % 8 == 0
    assert shape["duration_s"] <= 8
    with pytest.raises(ValueError):
        output_geometry(8, 30, 255, 256)


def test_p2_native_packing_roundtrip():
    from ltx_core.components.patchifiers import AudioPatchifier, VideoLatentPatchifier
    from omni_av_p2.ltx import unpack_latents
    video, audio = torch.randn(1, 128, 3, 2, 4), torch.randn(1, 8, 12, 16)
    sampled = dict(video=VideoLatentPatchifier(1).patchify(video), audio=AudioPatchifier(1).patchify(audio))
    unpacked = unpack_latents(sampled, dict(latents=dict(latents=video), audio_latents=dict(latents=audio)))
    assert torch.equal(video, unpacked[0]) and torch.equal(audio, unpacked[1])


def test_p2_native_hidden_indices_match_completed_blocks():
    from transformers.models.qwen2_5_omni.configuration_qwen2_5_omni import Qwen2_5OmniTextConfig
    from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import Qwen2_5OmniThinkerTextModel
    from omni_av_p2.data import select_hidden_layers
    cfg = Qwen2_5OmniTextConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
        rope_scaling={"rope_type": "default", "mrope_section": [1, 1, 2]})
    cfg._attn_implementation = "sdpa"
    model = Qwen2_5OmniThinkerTextModel(cfg).eval()
    captured = {}
    def capture(index):
        def hook(module, args, output):
            captured[index] = (output[0] if isinstance(output, tuple) else output).detach().clone()
        return hook
    hooks = [block.register_forward_hook(capture(i + 1)) for i, block in enumerate(model.layers)]
    try:
        result = model(input_ids=torch.tensor([[1, 2, 3, 4]]), output_hidden_states=True, use_cache=False)
    finally:
        for hook in hooks:
            hook.remove()
    selected = select_hidden_layers(result.hidden_states, (1, 2), 3)
    for i in (1, 2):
        torch.testing.assert_close(selected[:, i - 1], captured[i])
    torch.testing.assert_close(selected[:, -1], result.hidden_states[-2])
    assert not selected.requires_grad
    with pytest.raises(ValueError):
        select_hidden_layers(result.hidden_states, (3,), 3)


@pytest.mark.parametrize("mode,layers", [("attention", (8, 18, 28, 35)),
    ("weighted_sum", (8, 18, 28, 35)), ("single", (35,))])
def test_p2_layer_modes_keep_shared_shapes_and_train_every_selected_layer(mode, layers):
    torch.manual_seed(2026)
    cfg = replace(config(), hidden_layers=layers, layer_fusion=mode, zero_init_connector=False)
    model, data = FusionConnector(cfg), condition(layers)
    data.hidden.requires_grad_(True)
    memory, mask = model.memory(data)
    copies = len(layers) if mode == "attention" else 1
    assert memory.shape == (1, 7 * copies, cfg.fusion_width)
    assert torch.equal(mask, data.hidden_valid.repeat(1, copies))
    out = model(data)
    assert out.video.shape == (1, 3, cfg.video_dim) and out.audio.shape == (1, 2, cfg.audio_dim)
    (out.video.square().mean() + out.audio.square().mean()).backward()
    assert data.hidden.grad is None  # frozen Omni boundary
    for norm in model.memory.norms:
        assert norm.weight.grad.isfinite().all() and norm.weight.grad.abs().sum() > 0
    if mode == "weighted_sum":
        assert model.memory.layer_logits.grad.abs().sum() > 0
    if mode == "attention":
        assert (model.memory.layer_embedding.grad.abs().sum(-1) > 0).all()
    for i in range(len(layers)):
        changed = data.hidden.detach().clone()
        changed[:, i, :5] = torch.randn_like(changed[:, i, :5])
        assert not torch.allclose(out.video, model(replace(data, hidden=changed)).video)


def test_p2_layer_identity_and_old_cache_rejection():
    data, cfg = condition(), config()
    assert Config(**json.loads(json.dumps(cfg.to_dict()))) == cfg
    with pytest.raises(ValueError, match="layer IDs differ"):
        FusionConnector(cfg)(replace(data, layer_ids=torch.tensor([7, 18, 28, 35])))
    with pytest.raises(ValueError, match="regenerate"):
        FusionConnector(cfg)(replace(data, hidden=data.hidden[:, -1]))
    for fields in (dict(hidden_layers=[18, 8]), dict(hidden_layers=[8, 8]),
                   dict(hidden_layers=[0]), dict(layer_fusion="single")):
        with pytest.raises(ValueError):
            replace(cfg, **fields)


def test_p2_rejects_previous_checkpoint_version(tmp_path):
    path = tmp_path / "old.pt"
    torch.save(dict(version="p2_shared_condition_v1", identity={}), path)
    with pytest.raises(ValueError, match="identity mismatch"):
        load(path, FusionConnector(config()), {})
