"""Regression tests for held-out evaluation, split isolation and multi-node state."""
import copy
import json
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from omni_av.data import gt_text_plan
from omni_av.ltx_runtime import load_av_targets
from omni_av_p2.checkpoint import atomic_save, capture_rng, load, save
from omni_av_p2.launch import launch_command, launch_environment
from omni_av_p2.validation import check_indexes, evaluate
import omni_av_p2.validation as validation


def test_null_prosody_is_valid_absence():
    plan = gt_text_plan({'avatar_plan': {'sample_id': 's', 'response': {'present': False, 'prosody': None}}}, 's')
    assert plan.reply == '' and plan.delivery == ''


def test_index_isolation_and_legacy_contract():
    def index(split, sid):
        return dict(contract={}, split=split, entries=[dict(sample_id=sid, split=split)])
    train, dev = index('train', 'a'), index('dev', 'b')
    check_indexes(train, dev, {})
    check_indexes(dict(contract={}, entries=[{}]), None, {})
    with pytest.raises(ValueError, match='overlap'):
        check_indexes(train, index('dev', 'a'), {})
    with pytest.raises(ValueError, match='Expected dev'):
        check_indexes(train, train, {})
    with pytest.raises(ValueError, match='Duplicate'):
        check_indexes(train, dict(dev, entries=dev['entries'] * 2), {})
    with pytest.raises(ValueError, match='reindex'):
        check_indexes(dict(contract={}, entries=[{}]), dev, {})


def test_vae_split_is_explicit(tmp_path):
    root = tmp_path / 'ltx23_precomputed_720p_area_qualified_20260920'
    root.mkdir()
    for split in ('train', 'dev'):
        (root / f'metadata_{split}.jsonl').write_text(json.dumps(dict(sample_id=split, split=split,
            target_frames=17, target_height=32, target_width=32, fps=30)) + '\n')
    for key, shape in (('latents', (128, 3, 1, 1)), ('audio_latents', (8, 14, 16))):
        folder = root / key / 'dev'
        folder.mkdir(parents=True)
        torch.save(dict(latents=torch.zeros(shape), duration=17/30), folder / 'dev.pt')
    result, meta = load_av_targets(tmp_path, 'dev', device='cpu', split='dev')
    assert meta['metadata']['split'] == 'dev' and result['latents']['latents'].shape == (1, 128, 3, 1, 1)
    with pytest.raises(ValueError, match='train'):
        load_av_targets(tmp_path, 'dev', device='cpu')


def test_atomic_checkpoint_does_not_replace_existing(tmp_path):
    path = tmp_path / 'state.pt'
    atomic_save(path, {'a': 1})
    with pytest.raises(FileExistsError):
        atomic_save(path, {'a': 2})
    assert torch.load(path, weights_only=True) == {'a': 1}
    assert list(tmp_path.iterdir()) == [path]


def test_checkpoint_restores_each_rank_rng(tmp_path):
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    states, expected = [], []
    for seed in (101, 202):
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        states.append(capture_rng())
        expected.append((torch.rand(3), random.random(), np.random.rand()))
    path = tmp_path / 'model.pt'
    save(path, model, optimizer, 2, {}, rank_rng=states)
    for rank in (0, 1):
        load(path, model, {}, optimizer, rank=rank)
        torch.testing.assert_close(torch.rand(3), expected[rank][0], rtol=0, atol=0)
        assert random.random() == expected[rank][1] and np.random.rand() == expected[rank][2]


def test_validation_repeatable_and_preserves_state(monkeypatch):
    raw, renderer = torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)
    raw.train()
    renderer.train()
    class Samples:
        entries = list(range(3))
        def __getitem__(self, index):
            return torch.ones(1, 2) * index, {}, {}, {}, {}
    class Strategy:
        def prepare_training_inputs(self, av, sampler):
            return torch.rand(1, 2)
    monkeypatch.setattr(validation, 'with_text', lambda av, text: av)
    monkeypatch.setattr(validation, 'append_conditions', lambda inputs, u: inputs + u)
    def loss(renderer, strategy, inputs):
        assert not torch.is_grad_enabled() and not renderer.training
        v = renderer(inputs).square().mean()
        return 2*v, dict(video_loss=v, audio_loss=v)
    monkeypatch.setattr(validation, 'av_loss', loss)
    monkeypatch.setattr(validation, 'distributed_mean', lambda sums, count, device: ([v/count for v in sums], count))
    cfg = SimpleNamespace(av_weight=1, feature_video_weight=1, feature_audio_weight=1)
    state = copy.deepcopy(raw.state_dict())
    rng = torch.get_rng_state()
    first = evaluate(raw, renderer, Strategy(), None, Samples(), cfg, 'av', 0, 1, torch.device('cpu'), 123)
    second = evaluate(raw, renderer, Strategy(), None, Samples(), cfg, 'av', 0, 1, torch.device('cpu'), 123)
    assert first == second and first['samples'] == 3
    assert raw.training and renderer.training and all(p.grad is None for p in raw.parameters())
    torch.testing.assert_close(rng, torch.get_rng_state(), rtol=0, atol=0)
    for key, value in raw.state_dict().items():
        torch.testing.assert_close(value, state[key], rtol=0, atol=0)


def test_multinode_command_and_network_profile(tmp_path):
    args = SimpleNamespace(nnodes=2, node_rank=1, master_addr='10.0.0.1', master_port=29518,
        run_id='experiment1', output=tmp_path/'run', probe='cpu', network_config=None)
    command = launch_command(args, 2, 'train.py', ['--steps', '1'])
    assert '--node_rank=1' in command and '--nnodes=2' in command and '--standalone' not in command
    assert any('experiment1/node_1' in x for x in command)
    args.network_config = tmp_path / 'network.json'
    args.network_config.write_text('{"NCCL_IB_GID_INDEX": "3"}')
    assert launch_environment(args, [0, 1])['NCCL_IB_GID_INDEX'] == '3'
    args.network_config.write_text('{"LD_PRELOAD": "bad"}')
    with pytest.raises(ValueError):
        launch_environment(args, [0, 1])


def test_merge_shards_exact_coverage_and_order(tmp_path):
    import runpy
    from pathlib import Path
    merge = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'scripts/pipeline2_merge_cache.py'))['merge']
    entries = []
    for sid in ('a', 'b', 'c'):
        asset = tmp_path / f'{sid}.pt'
        asset.touch()
        entries.append(dict(sample_id=sid, split='train', features=str(asset), text=str(asset)))
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps([dict(split='train', observation=dict(sample_id=sid)) for sid in ('a', 'b', 'c')]))
    paths = [tmp_path / f'{i}.json' for i in (0, 1)]
    for i, path in enumerate(paths):
        path.write_text(json.dumps(dict(contract={}, split='train', entries=entries[i::2])))
    assert merge(paths, manifest)['entries'] == entries
    with pytest.raises(ValueError, match='missing'):
        merge(paths[:1], manifest)
    with pytest.raises(ValueError, match='Duplicate'):
        merge(paths + paths, manifest)


def test_teacher_does_not_supervise_frame_rounded_audio_tail(monkeypatch):
    import omni_av_p2.data as data
    model = torch.nn.Linear(1, 1)
    model.get_video_features = lambda *a: torch.zeros(2, 2048)
    model.get_audio_features = lambda *a: torch.zeros(2, 2048)
    cfg = SimpleNamespace(vision_fps=2, audio_hz=4, target_dim=2048)
    runtime = SimpleNamespace(config=cfg, thinker=model)
    processor = SimpleNamespace(
        image_processor=SimpleNamespace(patch_size=14, merge_size=2),
        video_processor=lambda **k: dict(pixel_values_videos=torch.zeros(1), video_grid_thw=torch.ones(1)),
        feature_extractor=lambda *a, **k: dict(input_features=torch.zeros(1), attention_mask=torch.ones(1)))
    condition = SimpleNamespace(video_coordinates=torch.tensor([[[.25,0,0],[7.25,0,0]]]),
        audio_coordinates=torch.tensor([[[.125,0,0],[7.375,0,0]]]),
        video_valid=torch.ones(1,2,dtype=torch.bool), audio_valid=torch.ones(1,2,dtype=torch.bool))
    example = dict(observation=dict(output_duration_s=8), duration_s=7.472, target_video='video', target_audio='audio')
    def read(path, end_s):
        assert end_s == 7.472  # 7.5-second VAE window used to overrun this real audio.
        return np.zeros(round(end_s * 16000), dtype=np.float32)
    monkeypatch.setattr(data, 'read_audio', read)
    monkeypatch.setattr(data, 'sample_target_frames', lambda *a: np.zeros((16,28,28,3)))
    targets = data.extract_teacher(runtime, processor, condition, example, 7.5)
    assert targets['video_valid'].tolist() == [[True,False]]
    assert targets['audio_valid'].tolist() == [[True,False]]


def test_balancing_keeps_global_samples_and_noise_identity():
    from omni_av_p2.performance import balanced_sample_indices, global_sample_index
    costs = [1] * 8 + [4] * 8 + [9] * 8
    plan = balanced_sample_indices(2, 24, 3, costs)
    assert sorted(x for rank in plan for x in rank) == list(range(72, 144))
    assert all(len(rank) == 3 for rank in plan)
    balanced = [sum(costs[x % 24] for x in rank) for rank in plan]
    previous = [sum(costs[global_sample_index(2, m, r, 24, 3) % 24] for m in range(3)) for r in range(24)]
    assert max(balanced) == 14 and max(previous) == 27
    # Noise is seeded by the preserved ordinal, independent of rank placement.
    assert sorted(45 + x + 1 for rank in plan for x in rank) == list(range(118, 190))
