"""Reference and regression checks added by the Bernini/eight-GPU audit."""
import ast
import copy
import json
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch import nn

from omni_av.data import file_hash, gt_text_plan
from omni_av_p2.connector import RMSNorm, bernini_projection
from omni_av_p2.data import validate_cached_example
from omni_av_p2.decoder import EmbeddingDecoder
from omni_av_p2.ltx import append_conditions, av_loss
from omni_av_p2.model import Extractor
from omni_av_p2.performance import configure_checkpointing, global_sample_index
from test_p2 import condition, config, native_ltx


def test_bernini_generation_projection_forward_and_all_gradients():
    path = Path("/private/liqi_workspace/project_code/Bernini/bernini/models/bernini.py")
    if not path.exists():
        pytest.skip("Local upstream Bernini reference unavailable")
    tree = ast.parse(path.read_text())
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef)
               and node.name in {"RMSNorm", "MLPConnector"}]
    namespace = {"torch": torch, "nn": nn}
    exec(compile(ast.Module(body=classes, type_ignores=[]), str(path), "exec"), namespace)
    reference = namespace["MLPConnector"](16, out_dim_for_gen=24, enable_vit_branch=False)
    actual = bernini_projection(16, 24, zero_init=False)
    actual.load_state_dict(reference.proj_gen.state_dict())
    x = torch.randn(2, 3, 16, requires_grad=True)
    y = x.detach().clone().requires_grad_(True)
    a, b = reference.for_gen(x), actual(y)
    torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-6)
    a.square().mean().backward()
    b.square().mean().backward()
    torch.testing.assert_close(x.grad, y.grad)
    for p, q in zip(reference.proj_gen.parameters(), actual.parameters()):
        torch.testing.assert_close(p.grad, q.grad)


def test_feature_fm_noise_sign_repeats_mask_and_condition_gradient():
    torch.manual_seed(123)
    decoder = EmbeddingDecoder(8, 12, 16, 2, repeats=3)
    # Open all branches so sign/order errors cannot hide behind zero initialization.
    with torch.no_grad():
        for p in decoder.parameters():
            p.normal_(0, .1)
    target = torch.randn(1, 4, 8, requires_grad=True)
    u = torch.randn(1, 4, 12, requires_grad=True)
    valid = torch.tensor([[True, False, True, False]])
    actual = decoder.loss(target, u, valid, torch.Generator().manual_seed(73))
    generator = torch.Generator().manual_seed(73)
    z = target[valid].detach().repeat_interleave(3, 0)
    c = u[valid].repeat_interleave(3, 0)
    noise = torch.randn(z.shape, generator=generator)
    sigma = torch.rand(z.shape[0], generator=generator)
    x = (1 - sigma[:, None]) * z + sigma[:, None] * noise
    expected = (decoder.net(x, sigma * 1000, c) - (noise - z)).square().mean()
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert target.grad is None
    assert u.grad[valid].abs().sum() > 0
    assert u.grad[~valid].count_nonzero() == 0
    bad_target = target.detach().clone()
    bad_target[~valid] = float("nan")
    torch.testing.assert_close(actual.detach(), decoder.loss(bad_target, u.detach(), valid,
                                                           torch.Generator().manual_seed(73)))


def test_each_loss_reaches_shared_connector_independently():
    torch.manual_seed(17)
    model = Extractor(config())
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(0, .1)
    renderer, strategy, prepared = native_ltx()
    data = condition()
    targets = dict(video=torch.randn(1, 3, 16), audio=torch.randn(1, 2, 16),
                   video_valid=data.video_valid, audio_valid=data.audio_valid)
    u, features = model(data, targets)
    lav, _ = av_loss(renderer, strategy, append_conditions(prepared, u))
    weight = model.connector.memory.projection.weight
    for loss in (lav, features["video"], features["audio"]):
        grad = torch.autograd.grad(loss, weight, retain_graph=True)[0]
        assert grad.isfinite().all() and grad.abs().sum() > 0
    assert all(p.grad is None and not p.requires_grad for p in renderer.parameters())


@pytest.mark.parametrize("every", [0, 1, 2])
def test_renderer_recompute_policy_preserves_outputs_and_gradients(every):
    torch.manual_seed(3)
    renderer, strategy, prepared = native_ltx()
    alternative = copy.deepcopy(renderer)
    raw = Extractor(config())
    configure_checkpointing(alternative, raw, every=every, decoder=False)
    renderer.train()
    alternative.train()
    u = raw(condition())
    p = append_conditions(prepared, u)
    a, _ = av_loss(renderer, strategy, p)
    b, _ = av_loss(alternative, strategy, p)
    torch.testing.assert_close(a, b)
    ga = torch.autograd.grad(a, (u.video, u.audio), retain_graph=True)
    gb = torch.autograd.grad(b, (u.video, u.audio))
    for first, second in zip(ga, gb):
        torch.testing.assert_close(first, second)


def test_ddp_and_accumulation_assign_identical_samples_and_noise():
    for step in (1, 2, 7):
        parallel = [global_sample_index(step, micro, rank, 8, 2)
                    for micro in range(2) for rank in range(8)]
        serial = [global_sample_index(step, micro, 0, 1, 16) for micro in range(16)]
        assert parallel == serial


def test_changed_gt_and_cached_sample_identity_are_rejected(tmp_path):
    gt = dict(avatar_plan=dict(sample_id="sample", response=dict(present=True, text="hello")))
    path = tmp_path / "gt.json"
    path.write_text(json.dumps(gt))
    plan = vars(gt_text_plan(gt, "sample"))
    example = dict(observation=dict(sample_id="sample", speaker_audio="speaker.wav", listener_image="image.jpg",
                   speaker_start_s=0, speaker_end_s=2, output_duration_s=8),
                   split="train", duration_s=8, text_plan=plan, gt_path=str(path))
    payload = dict(example=example, sample_id="sample", gt_sha256=file_hash(path))
    validate_cached_example(payload)
    with pytest.raises(ValueError, match="identity"):
        validate_cached_example(dict(payload, sample_id="another"))
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="GT changed"):
        validate_cached_example(payload)


def _distributed_gradient_worker(rank, rendezvous):
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method="file://" + rendezvous, rank=rank, world_size=2)
    try:
        torch.manual_seed(29)
        raw = Extractor(replace(config(), zero_init_connector=False))
        # Exercise auxiliary gradients immediately, not only zero-initialized branches.
        with torch.no_grad():
            for p in raw.parameters():
                p.normal_(0, .1)
        serial = copy.deepcopy(raw)
        parallel = DistributedDataParallel(raw)
        opt = torch.optim.SGD(raw.parameters(), lr=.01)
        serial_opt = torch.optim.SGD(serial.parameters(), lr=.01)

        def objective(model, ordinal):
            torch.manual_seed(1000 + ordinal)
            data = condition()
            targets = dict(video=torch.randn(1, 3, 16), audio=torch.randn(1, 2, 16),
                           video_valid=data.video_valid, audio_valid=data.audio_valid)
            u, feature = model(data, targets, torch.Generator().manual_seed(2000 + ordinal))
            return u.video.square().mean() + u.audio.square().mean() + sum(feature.values())

        for step in (1, 2):
            opt.zero_grad()
            serial_opt.zero_grad()
            for micro in range(2):
                with parallel.no_sync() if micro == 0 else nullcontext():
                    (objective(parallel, global_sample_index(step, micro, rank, 2, 2)) / 2).backward()
            for micro in range(4):
                (objective(serial, global_sample_index(step, micro, 0, 1, 4)) / 4).backward()
            for a, b in zip(raw.parameters(), serial.parameters()):
                torch.testing.assert_close(a.grad, b.grad, atol=2e-6, rtol=2e-5)
            opt.step()
            serial_opt.step()
            for a, b in zip(raw.parameters(), serial.parameters()):
                torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)
    finally:
        dist.destroy_process_group()


def test_ddp_gradients_and_updates_match_serial_accumulation(tmp_path):
    torch.multiprocessing.spawn(_distributed_gradient_worker,
                                args=(str(tmp_path / "rendezvous"),), nprocs=2, join=True)
