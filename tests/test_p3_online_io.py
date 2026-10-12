import importlib.util
import json
from pathlib import Path

import pytest
import torch

from omni_av_p2.checkpoint import atomic_save, atomic_json_new, capture_rng
from omni_av_p3.distributed import sha
from omni_av_p3_online.checkpoint import initialize, resume
from omni_av_p3_online.data import read_native
from omni_av_p3_online.model import select_contexts
from omni_av_p3_online.runtime import update_learning_rates
from test_p3_online import make, inputs


def test_native_integrity_and_prediction_excludes_teacher(tmp_path):
    layout, target = inputs()
    path = tmp_path / "native.pt"
    atomic_save(path, dict(layout=vars(layout), targets=target, sample_id="sample", split="dev"))
    entry = dict(sample_id="sample", split="dev", native=dict(path=str(path), sha256=sha(path)))
    found, values = read_native(entry)
    torch.testing.assert_close(values, target)
    prediction_layout = read_native(entry, include_targets=False)
    assert not hasattr(prediction_layout, "targets")
    torch.testing.assert_close(prediction_layout.prefix, found.prefix)
    with path.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="hash"):
        read_native(entry)


def test_context_padding_preserves_gradients_and_valid_masks():
    from dataclasses import replace
    data, target = inputs()
    data = replace(data, **{k: (v.repeat(1, 2, 1) if k == "prefix_positions" else
                               v.repeat(2, *([1] * (v.ndim - 1)))) for k, v in vars(data).items()})
    data.prefix_valid[1, -1] = False
    prefix = torch.randn(2, 4, 32, requires_grad=True)
    hidden = torch.randn(2, 6, 32, requires_grad=True)
    mask = torch.tensor([[False, True, True, True, False, True], [True] * 6])
    c = select_contexts(prefix, hidden, data, mask)
    assert c.video_valid.tolist() == [[True] * 5, [True] * 3 + [False] * 2]
    assert c.audio_valid.tolist() == c.video_valid.tolist()
    c.video.sum().backward()
    assert prefix.grad[1, -1].eq(0).all() and hidden.grad[1].eq(0).all()


def test_offline_checkpoint_only_explicit_c0_warm_start(tmp_path):
    from omni_av_p3_staged.model import RenderTraining, GenerationConnector
    from omni_av_p3_staged.checkpoint import save as offline_save
    from test_p2 import native_ltx
    renderer, strategy, _ = native_ltx()
    old = RenderTraining(renderer, strategy, "C0", GenerationConnector(32, 32, 32))
    cp = tmp_path / "offline.pt"
    offline_save(cp, old, torch.optim.AdamW(old.optimizer_groups(.001, .001)), 100,
                 dict(world_size=1, manifest_sha256="manifest"), [capture_rng()])
    c0, _ = make("C0")
    assert initialize(cp, c0, "manifest", sha(cp), "planner") == 100
    for k, v in c0.connector.state_dict().items():
        torch.testing.assert_close(v, old.connector.state_dict()[k], rtol=0, atol=0)
    with pytest.raises(ValueError, match="identity"):
        resume(cp, c0, torch.optim.AdamW(c0.optimizer_groups(.001, .001, .001, .001)), {})
    a, _ = make("A")
    with pytest.raises(ValueError, match="online C0"):
        initialize(cp, a, "manifest", sha(cp), "planner")


def test_planner_lr_only_starts_after_warmup():
    model, _ = make()
    opt = torch.optim.AdamW(model.optimizer_groups(.001, .001, .001, .001))
    recipe = dict(connector_lr=3e-5, dit_lr=5e-6, planner_lr=1e-5, heads_lr=1e-5,
                  online=dict(planner_warmup_steps=100))
    update_learning_rates(opt, recipe, 100, False)
    rates = {g["name"]: g["lr"] for g in opt.param_groups}
    assert rates["omni_lora"] == rates["feature_heads"] == 0
    assert rates["dit_text_lora"] == 5e-6
    update_learning_rates(opt, recipe, 101, True)
    assert {g["name"]: g["lr"] for g in opt.param_groups}["omni_lora"] == pytest.approx(1e-5 / 50)


def test_launcher_only_accepts_network_environment():
    path = Path(__file__).resolve().parents[1] / "scripts/pipeline3_online_cluster.py"
    spec = importlib.util.spec_from_file_location("online_cluster", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.network_environment({"NCCL_IB_DISABLE": 0}) == {"NCCL_IB_DISABLE": "0"}
    with pytest.raises(ValueError):
        module.network_environment({"PYTHONPATH": "/wrong/source"})
