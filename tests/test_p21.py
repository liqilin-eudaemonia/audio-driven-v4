"""P2.1 native LTX gradients, exact target boundaries, DDP and deployment parity."""
import copy
from contextlib import nullcontext

import pytest
import torch

from omni_av_p2.ltx import append_conditions, av_loss
from omni_av_p2.performance import configure_checkpointing
from omni_av_p21 import VERSION
from omni_av_p21.checkpoint import (export_deployment, initialize_from_p2, load, save,
                                    validate_state)
from omni_av_p21.config import LoRAConfig, read_config
from omni_av_p21.lora import AdaptedLinear, inject_text_lora
from omni_av_p21.model import TrainingModel
from omni_av_p21 import paired_validation as paired
from test_p2 import condition, config, native_ltx


def make_model(auxiliary=True):
    renderer, strategy, prepared = native_ltx()
    base = copy.deepcopy(renderer)
    model = TrainingModel(config(), LoRAConfig(rank=2, alpha=2), renderer, strategy, auxiliary)
    return model, base, prepared


def test_shared_extractor_initialization_matches_p2_before_renderer():
    from omni_av_p2.model import Extractor
    torch.manual_seed(45)
    p2 = Extractor(config())
    torch.manual_seed(45)
    fresh = Extractor(config())
    renderer, strategy, _ = native_ltx()  # Deliberately consumes RNG.
    p21 = TrainingModel(config(), LoRAConfig(rank=2), renderer, strategy, extractor=fresh)
    for name, value in p2.state_dict().items():
        torch.testing.assert_close(value, p21.extractor.state_dict()[name], atol=0, rtol=0)


def targets_for(data):
    return dict(video=torch.randn(1, 3, 16), audio=torch.randn(1, 2, 16),
                video_valid=data.video_valid, audio_valid=data.audio_valid)


def test_only_native_text_attention_and_zero_delta():
    torch.manual_seed(41)
    model, base, prepared = make_model()
    model.assert_parameter_ownership()
    assert len(model.targets) == 8
    assert all(name.split('.')[2] in ('attn2', 'audio_attn2') for name in model.targets)
    assert all(name.endswith(('to_q', 'to_k', 'to_v', 'to_out.0')) for name in model.targets)
    assert not any('base.' in key or 'renderer.' in key for key in model.state_dict())
    assert len([m for m in model.renderer.modules() if isinstance(m, AdaptedLinear)]) == 8
    u = model.extractor(condition())
    inputs = append_conditions(prepared, u)
    a, _ = av_loss(base, model.strategy, inputs)
    b, _ = av_loss(model.renderer, model.strategy, inputs)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    with pytest.raises(ValueError, match='already'):
        inject_text_lora(model.renderer, model.lora)


@pytest.mark.parametrize('every', [0, 1, 2])
def test_av_gradients_and_frozen_base_after_updates(every):
    torch.manual_seed(42)
    model, _, prepared = make_model(auxiliary=False)
    configure_checkpointing(model.renderer, model.extractor, every, False)
    model.train()
    frozen = [(p, p.detach().clone()) for p in model.renderer.parameters() if not p.requires_grad]
    opt = torch.optim.AdamW(model.optimizer_groups())
    for step in range(2):
        opt.zero_grad(set_to_none=True)
        values = model(condition(), None, prepared)
        values[0].backward()
        for adapter in model.adapters:
            assert adapter.b.weight.grad.isfinite().all()
            assert adapter.b.weight.grad.abs().sum() > 0
            if step:
                assert adapter.a.weight.grad.abs().sum() > 0
        assert model.extractor.connector.video_output[-1].weight.grad.abs().sum() > 0
        assert model.extractor.connector.audio_output[-1].weight.grad.abs().sum() > 0
        opt.step()
    for param, initial in frozen:
        assert param.grad is None
        torch.testing.assert_close(param, initial, atol=0, rtol=0)
    model.assert_parameter_ownership()


def test_checkpoint_resume_and_deployment_restore_both_branches(tmp_path, monkeypatch):
    torch.manual_seed(43)
    model, base, prepared = make_model()
    data = condition()
    opt = torch.optim.AdamW(model.optimizer_groups())
    model(data, targets_for(data), prepared)[0].backward()
    opt.step()
    identity = dict(contract={'test': 'cache'}, entries=[{'sample_id': 'train'}], recipe='joint')
    full, deploy = tmp_path / 'state.pt', tmp_path / 'deploy.pt'
    save(full, model, opt, 7, identity)
    expected = copy.deepcopy(model.state_dict())
    expected_rng = torch.get_rng_state().clone()
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1)
    load(full, model, identity, opt)
    assert torch.equal(expected_rng, torch.get_rng_state())
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, expected[name], atol=0, rtol=0)
    export_deployment(full, deploy)
    saved = torch.load(deploy, weights_only=True)
    assert saved['version'] == VERSION
    assert all(k.startswith(('extractor.connector.', 'adapters.')) for k in saved['model'])
    assert not any('decoder' in k for k in saved['model'])
    suite = dict(config=model.cfg.to_dict(), contract=identity['contract'], cases=[{'sample_id': 'dev'}])
    bundle, step = paired.checkpoint_state(deploy, suite)
    assert step == 7
    monkeypatch.setattr(paired, 'load_renderer', lambda cfg, device: (copy.deepcopy(base), model.strategy, None))
    renderer, strategy, _ = paired.load_adapted_renderer(model.cfg, bundle, device='cpu')
    assert not any(p.requires_grad for p in renderer.parameters())
    u = model.extractor(data)
    inputs = append_conditions(prepared, u)
    a, _ = av_loss(model.renderer, strategy, inputs)
    b, _ = av_loss(renderer, strategy, inputs)
    torch.testing.assert_close(a, b, atol=0, rtol=0)
    with pytest.raises(ValueError, match='Deployment'):
        load(deploy, model, identity, opt)
    broken = copy.deepcopy(saved)
    broken['model'] = {k: v for k, v in broken['model'].items() if not k.startswith('adapters.')}
    torch.save(broken, tmp_path / 'missing.pt')
    with pytest.raises(ValueError, match='AND text LoRA'):
        paired.checkpoint_state(tmp_path / 'missing.pt', suite)
    with pytest.raises(ValueError, match='target mismatch'):
        paired.load_adapted_renderer(model.cfg, dict(bundle, architecture=dict(bundle['architecture'], targets=[])), 'cpu')
    with pytest.raises(ValueError, match='not completely'):
        paired.checkpoint_state(tmp_path / 'extractor_step_000100_deployment.pt', suite)


def test_p2_initialization_is_explicit_and_does_not_invalidate_cache(tmp_path):
    from omni_av_p2.checkpoint import save as save_p2
    from omni_av_p2.data import contract
    from omni_av_p2.config import Config
    cfg, lora = read_config()
    assert contract(cfg) == contract(Config())
    assert 'lora' not in contract(cfg)
    model, _, _ = make_model()
    identity = dict(contract={'same': 'cache'}, recipe='joint')
    path = tmp_path / 'p2.pt'
    save_p2(path, model.extractor, torch.optim.AdamW(model.extractor.parameters()), 20, identity)
    assert initialize_from_p2(path, model, identity['contract'], 'joint') == 20
    assert all(update.b.weight.count_nonzero() == 0 for update in model.adapters)
    with pytest.raises(ValueError, match='identity/architecture'):
        load(path, model)
    with pytest.raises(ValueError, match='matching'):
        initialize_from_p2(path, model, {'different': 'cache'}, 'joint')
    with pytest.raises(ValueError):
        LoRAConfig(rank=0)
    with pytest.raises(ValueError):
        LoRAConfig(learning_rate=float('nan'))
    with pytest.raises(ValueError, match='keys'):
        validate_state(model.adapters, {})


def test_bfloat16_base_fp32_adapters_and_validation_rng(monkeypatch):
    from types import SimpleNamespace
    from omni_av_p21 import validation
    torch.manual_seed(45)
    renderer, strategy, prepared = native_ltx()
    renderer.to(torch.bfloat16)
    model = TrainingModel(config(), LoRAConfig(rank=2), renderer, strategy)
    assert all(p.dtype == torch.float32 for p in model.parameters())
    data = condition()
    targets = targets_for(data)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        value = model(data, targets, prepared)[0]
    value.backward()
    assert all(p.grad is not None and p.grad.isfinite().all() for p in model.parameters())
    # Validation uses the same composite model and restores RNG/training mode.
    renderer.float()
    class Samples:
        entries = [1, 2, 3]
        def __getitem__(self, i):
            return data, targets, {}, {}, {}
    monkeypatch.setattr(validation, 'with_text', lambda av, text: None)
    monkeypatch.setattr(validation, 'distributed_mean', lambda sums, n, device: ([s / n for s in sums], n))
    fake_strategy = SimpleNamespace(prepare_training_inputs=lambda av, sampler: prepared)
    state = torch.get_rng_state().clone()
    first = validation.evaluate(model, renderer, fake_strategy, None, Samples(), model.cfg,
                                'joint', 0, 1, torch.device('cpu'), 1045)
    second = validation.evaluate(model, renderer, fake_strategy, None, Samples(), model.cfg,
                                 'joint', 0, 1, torch.device('cpu'), 1045)
    assert first == second and first['samples'] == 3
    assert model.training and renderer.training
    assert torch.equal(state, torch.get_rng_state())


def test_ltx23_gated_attention_and_text_adaln():
    from ltx_core.model.transformer.model import LTXModel
    from ltx_core.model.transformer.attention import AttentionFunction
    torch.manual_seed(46)
    _, strategy, prepared = native_ltx()
    renderer = LTXModel(num_attention_heads=2, attention_head_dim=16, num_layers=2,
        cross_attention_dim=32, audio_num_attention_heads=2, audio_attention_head_dim=16,
        audio_cross_attention_dim=32, attention_type=AttentionFunction.PYTORCH,
        apply_gated_attention=True, cross_attention_adaln=True,
        av_ca_timestep_scale_multiplier=1000, double_precision_rope=True)
    with torch.no_grad():
        for name, p in renderer.named_parameters():
            if 'scale_shift_table' in name:
                p.normal_(0, .02)
    renderer.requires_grad_(False)
    model = TrainingModel(config(), LoRAConfig(rank=2), renderer, strategy, auxiliary=False)
    configure_checkpointing(renderer, model.extractor, every=2, decoder=False)
    data = condition()
    values = model(data, None, prepared)
    values[0].backward()
    assert len(model.targets) == 16
    for p in model.adapters.parameters():
        assert p.grad is not None and p.grad.isfinite().all()
    assert all(update.b.weight.grad.abs().sum() > 0 for update in model.adapters)
    # New LTX2.3 gates and AdaLN tables remain frozen as specified.
    assert all(not p.requires_grad for name, p in renderer.named_parameters()
               if 'to_gate_logits' in name or 'scale_shift_table' in name)


def test_multinode_entry_and_busy_gpu_guard(tmp_path, monkeypatch, capsys):
    import json
    from pathlib import Path
    import runpy
    import sys
    import omni_av_p2.launch as launcher
    script = Path(__file__).resolve().parents[1] / 'scripts/pipeline21_parallel.py'
    cache = tmp_path / 'cache.json'
    cache.write_text('{}')
    output = tmp_path / 'output'
    argv = [str(script), '--cache', str(cache), '--output', str(output), '--preflight-only',
            '--nnodes', '3', '--node-rank', '1', '--master-addr', '127.0.0.1', '--gpus', '0,1']
    monkeypatch.setattr(sys, 'argv', argv)
    checked = []
    monkeypatch.setattr(launcher, 'idle_gpus', lambda gpus: checked.append(gpus))
    runpy.run_path(str(script), run_name='__main__')
    result = json.loads(capsys.readouterr().out)
    assert str(script) in result['command'] and '--nnodes=3' in result['command']
    assert '--node_rank=1' in result['command'] and '--worker' in result['command']
    assert checked == [[0, 1]] and not output.exists()
    def busy(gpus):
        raise RuntimeError('GPU occupied')
    monkeypatch.setattr(launcher, 'idle_gpus', busy)
    with pytest.raises(RuntimeError, match='occupied'):
        runpy.run_path(str(script), run_name='__main__')
    assert not output.exists()


def _ddp_worker(rank, rendezvous):
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method='file://' + rendezvous, rank=rank, world_size=2)
    try:
        torch.manual_seed(44)
        raw, _, prepared = make_model()
        serial = copy.deepcopy(raw)
        serial.assert_parameter_ownership()
        parallel = DistributedDataParallel(raw, broadcast_buffers=False)
        optimizer = torch.optim.SGD(raw.parameters(), lr=.001)
        serial_optimizer = torch.optim.SGD(serial.parameters(), lr=.001)

        def objective(model, ordinal):
            torch.manual_seed(1000 + ordinal)
            data = condition()
            return model(data, targets_for(data), prepared, torch.Generator().manual_seed(2000 + ordinal))[0]

        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            serial_optimizer.zero_grad(set_to_none=True)
            for micro in range(2):
                with parallel.no_sync() if micro == 0 else nullcontext():
                    (objective(parallel, step * 4 + micro * 2 + rank) / 2).backward()
            for ordinal in range(step * 4, (step + 1) * 4):
                (objective(serial, ordinal) / 4).backward()
            for p, q in zip(raw.parameters(), serial.parameters()):
                assert p.grad is not None and q.grad is not None
                torch.testing.assert_close(p.grad, q.grad, atol=3e-6, rtol=3e-5)
            optimizer.step()
            serial_optimizer.step()
            for p, q in zip(raw.parameters(), serial.parameters()):
                torch.testing.assert_close(p, q, atol=3e-6, rtol=3e-5)
            # Test per-rank RNG and adapter-inclusive optimizer restore after DDP updates.
            path = rendezvous + f'.rank{rank}.pt'
            identity = dict(rank=rank)
            from omni_av_p2.checkpoint import capture_rng
            states = [None, None]
            dist.all_gather_object(states, capture_rng())
            if step == 1:
                save(path, raw, optimizer, step + 1, identity, rank_rng=states)
                load(path, raw, identity, optimizer, rank=rank)
    finally:
        dist.destroy_process_group()


def test_two_rank_ddp_accumulation_matches_serial_and_resume(tmp_path):
    torch.multiprocessing.spawn(_ddp_worker, args=(str(tmp_path / 'rendezvous'),), nprocs=2, join=True)
