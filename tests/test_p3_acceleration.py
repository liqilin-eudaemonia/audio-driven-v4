"""Acceleration preserves per-position mask/noise, gradients and continuation order."""
from copy import deepcopy

import pytest
import torch

from test_p3 import tiny, layout
from omni_av_p3.accelerated import FastPlanner, audit_gradients, execution_mode, optimizer_for, prepare_mask, resident
from omni_av_p3.extended import example_order
from omni_av_p3.model import content_mask


@pytest.mark.parametrize("arm", ["frozen", "lora"])
def test_prepared_forward_and_gradients_match_reference_with_active_conditioning(arm):
    reference = tiny(arm)
    with torch.no_grad():
        for decoder in reference.decoders:
            for p in decoder.parameters():
                p.normal_(0, .04)
    actual = deepcopy(reference)
    actual.__class__ = FastPlanner
    actual._geometry = {}
    execution_mode(reference, False)
    execution_mode(actual, True)
    data = layout()
    values = torch.randn(1, 6, 32)
    item = resident(dict(layout=vars(data), targets=values, sample_id="test"), "cpu")
    masked, ids = prepare_mask(item, actual.cfg, 111)
    assert torch.equal(masked, content_mask(data, reference.cfg, 111))
    first = reference.loss(data, values, 111)
    second = actual.prepared_loss(item, masked, ids, 111)
    for key in first:
        torch.testing.assert_close(first[key], second[key], atol=1e-6, rtol=1e-6)
    sum(first.values()).backward()
    sum(second.values()).backward()
    audit = audit_gradients(actual)
    assert set(audit) == {"mask", "connector", "decoder", "lora"}
    assert audit["connector"] > 0 and audit["decoder"] > 0
    assert audit["lora"] == 0 if arm == "frozen" else audit["lora"] > 0
    for (n, a), (m, b) in zip(reference.named_parameters(), actual.named_parameters()):
        assert n == m
        if a.grad is not None:
            torch.testing.assert_close(a.grad, b.grad, atol=2e-6, rtol=2e-5)
        else:
            assert b.grad is None


def test_continuation_schedule_is_prefix_consistent():
    assert example_order(16, 200, 15) == example_order(16, 2000, 15)[:200]
    assert sorted(example_order(16, 16, 15)) == list(range(16))


def test_optimizer_restore_does_not_mutate_checkpoint_step_tensors():
    model = tiny()
    optimizer, _ = optimizer_for(model, fused=False)
    data = layout()
    values = torch.randn(1, 6, 32)
    sum(model.loss(data, values, 2).values()).backward()
    optimizer.step()
    snapshot = deepcopy(optimizer.state_dict())
    saved = deepcopy(snapshot)
    resumed, _ = optimizer_for(model, dict(optimizer=snapshot), fused=False)
    resumed.step()
    for key in snapshot["state"]:
        for name, value in snapshot["state"][key].items():
            if torch.is_tensor(value):
                torch.testing.assert_close(value, saved["state"][key][name], atol=0, rtol=0)


@pytest.mark.parametrize('arm',['frozen','lora'])
def test_padded_microbatch_preserves_per_example_loss_gradients_and_noise(arm):
    from omni_av_p3.distributed import tiny_rows
    model=tiny(arm);model.__class__=FastPlanner;model._geometry={}
    execution_mode(model,True)
    with torch.no_grad():
        for decoder in model.decoders:
            for p in decoder.parameters():p.normal_(0,.04)
    reference=deepcopy(model)
    data=[resident(row,'cpu') for row in tiny_rows(4,'train')]
    seeds=[101,202,303,404]
    prepared=[prepare_mask(row,model.cfg,seed) for row,seed in zip(data,seeds)]
    losses=[reference.prepared_loss(row,*prep,seed) for row,prep,seed in zip(data,prepared,seeds)]
    expected={k:torch.stack([loss[k] for loss in losses]).mean() for k in ('video','audio')}
    actual=model.prepared_batch_loss(data,[x[0] for x in prepared],[x[1] for x in prepared],seeds)
    for k in expected:torch.testing.assert_close(actual[k],expected[k],atol=1e-6,rtol=1e-6)
    sum(expected.values()).backward();sum(actual.values()).backward()
    for (n,a),(m,b) in zip(reference.named_parameters(),model.named_parameters()):
        assert n==m
        if a.grad is None:assert b.grad is None
        else:torch.testing.assert_close(a.grad,b.grad,atol=2e-6,rtol=2e-5)
