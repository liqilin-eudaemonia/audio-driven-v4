"""Local source parsing and real read-only data-contract checks."""
import ast
import json
from pathlib import Path

import pytest
import torch

from omni_av.config import PlanConfig
from omni_av.data import Observation, TextPlan
from omni_av.ltx_runtime import audit_target_window, load_av_targets
from omni_av.planner import PlanningBatch

ROOT = Path(__file__).resolve().parents[1]


def test_all_python_sources_parse():
    for directory in ("src", "scripts", "tests"):
        for path in (ROOT / directory).rglob("*.py"):
            ast.parse(path.read_text(), filename=str(path))


def test_real_cache_and_paired_vae_contract():
    path = ROOT / "outputs/omni_gate_003/features_0000.pt"
    if not path.exists():
        pytest.skip("Local GPU artifact not present")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    batch = PlanningBatch(**payload["tensors"])
    batch.validate()
    assert batch.targets.shape[-1] == PlanConfig().hidden_size
    observation = Observation(**payload["example"]["observation"])
    assert observation.sample_id == payload["sample_id"]
    plan = TextPlan(**payload["example"]["text_plan"])
    assert TextPlan.parse(plan.serialize()) == plan
    av, metadata = load_av_targets("/private/liqi_workspace/data/seamless/subset_300h_pure",
                                   payload["sample_id"], device="cpu")
    assert av["latents"]["latents"].shape[1] == 128
    assert not metadata["tiny"]
    audit_target_window(payload, metadata)
    inconsistent = dict(payload, example=dict(payload["example"], duration_s=20))
    with pytest.raises(ValueError, match="window"):
        audit_target_window(inconsistent, metadata)
