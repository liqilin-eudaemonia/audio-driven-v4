import importlib.util
import json
from pathlib import Path

import pytest
import torch

from omni_av_p2.checkpoint import atomic_save, atomic_json_new, capture_rng
from omni_av_p3.distributed import sha
from omni_av_p3_online.checkpoint import initialize, initialize_connector, resume
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
    with pytest.raises(ValueError, match="Beta P"):
        initialize(cp, c0, "manifest", sha(cp), "planner")
    initialize_connector(cp, c0, sha(cp))
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


def training_entry_ast():
    import ast
    source = Path(__file__).resolve().parents[1] / "scripts/pipeline3_online_train.py"
    main = next(n for n in ast.parse(source.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "main")
    return next(n for n in main.body if isinstance(n, ast.Try))


def test_dev_beta_follows_training_distribution_and_allmask_is_explicit():
    """Beta evaluation may legitimately ceil to all-mask, unlike old partial mode."""
    import ast
    from dataclasses import replace
    from types import SimpleNamespace
    from omni_av_p3_online.model import OnlineConfig, sample_mask
    block = training_entry_ast()
    evaluate = next(n for n in block.body if isinstance(n, ast.FunctionDef) and n.name == "evaluate")
    assignment = next(n for n in ast.walk(evaluate) if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == "mask" for t in n.targets))
    expression = compile(ast.Expression(assignment.value), "<actual-dev-mask>", "eval")
    layout, _ = inputs()
    for i in range(48):
        scope = dict(layout=layout, raw=SimpleNamespace(config=OnlineConfig()), seed=20261012+100000+i,
                     sample_mask=sample_mask, replace=replace)
        partial = eval(expression, dict(scope, mode="beta"))
        assert torch.equal(partial, sample_mask(layout, OnlineConfig(), scope["seed"]))
        for mod in (0, 1):
            selected = layout.valid & (layout.modalities == mod)
            assert partial[selected].any()
        assert torch.equal(eval(expression, dict(scope, mode="all_mask")), layout.valid)


@pytest.mark.parametrize("evaluation_fails", [False, True])
def test_final_checkpoint_resume_evaluates_before_completion(tmp_path, evaluation_fails):
    """Run the unchanged CLI control flow with its optimizer loop already exhausted."""
    import ast
    from types import SimpleNamespace
    block = training_entry_ast()
    boundary = next(i for i, n in enumerate(block.body) if isinstance(n, ast.Assign)
                    and isinstance(n.targets[0], ast.Tuple)
                    and [x.id for x in n.targets[0].elts] == ["last", "saved"])
    code = compile(ast.Module(body=block.body[boundary-1:], type_ignores=[]), "<actual-resume-tail>", "exec")
    events = []
    def evaluate(step):
        events.append(("evaluate", step))
        if evaluation_fails:
            raise RuntimeError("Dev failed")
    scope = dict(a=SimpleNamespace(output=tmp_path, stage="A"), rank=0, start=1000, steps=1000,
                 engineering=False, recipe={"validate_every":100}, evaluate=evaluate,
                 checkpoint=lambda step:events.append(("checkpoint",step)),
                 master_action=lambda action, rank:action(), atomic_json_new=atomic_json_new)
    if evaluation_fails:
        with pytest.raises(RuntimeError, match="Dev failed"):
            exec(code, scope)
        assert not (tmp_path / "completed.json").exists()
    else:
        exec(code, scope)
        assert json.loads((tmp_path / "completed.json").read_text())["stopped"] is False
    assert events[0] == ("evaluate", 1000)
