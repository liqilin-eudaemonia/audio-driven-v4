"""New checkpoint contract; stage transition is not optimizer resume."""
from pathlib import Path
import hashlib
import io
import json

import torch

from omni_av_p2.checkpoint import atomic_save, atomic_json_new
from omni_av_p21.checkpoint import validate_state
from omni_av_p3.distributed import sha
from . import VERSION


def checked(path):
    path = Path(path)
    marker = json.loads(path.with_suffix(".complete.json").read_text())
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != marker["sha256"]:
        raise ValueError("Checkpoint hash mismatch")
    data = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
    if data["step"] != marker["step"]:
        raise ValueError("Checkpoint step mismatch")
    return data


def save(path, model, optimizer, step, identity, rank_rng):
    model.assert_ownership()
    if len(rank_rng) != identity["world_size"]:
        raise ValueError("Missing rank RNG")
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    validate_state(model, state)
    atomic_save(path, dict(version=VERSION, stage=model.stage, step=step, identity=identity,
        planner_names=model.planner_names, model=state, optimizer=optimizer.state_dict(), rank_rng=rank_rng))
    atomic_json_new(Path(path).with_suffix(".complete.json"), dict(step=step, sha256=sha(path)))


def restore_model(data, model):
    if (data.get("version") != VERSION or data["stage"] != model.stage
            or tuple(data["planner_names"]) != model.planner_names):
        raise ValueError("Online checkpoint architecture mismatch; offline resume is forbidden")
    validate_state(model, data["model"])
    model.load_state_dict(data["model"], strict=True)


def resume(path, model, optimizer, identity):
    data = checked(path)
    if data.get("version") != VERSION or data["identity"] != identity:
        raise ValueError("Strict online resume identity mismatch")
    if len(data["rank_rng"]) != identity["world_size"]:
        raise ValueError("Resume world size mismatch")
    restore_model(data, model)
    optimizer.load_state_dict(data["optimizer"])
    return data


def initialize(path, model, manifest_sha, expected_sha, planner_sha):
    """Explicit hash-bound warm initialization; never carry old optimizer state."""
    if sha(path) != expected_sha:
        raise ValueError("Initialization checkpoint changed")
    data = checked(path)
    if model.stage == "C0":
        if data.get("version") == "p3.staged-render.v1" and data.get("stage") == "C0":
            state = {k.removeprefix("connector."): v for k, v in data["model"].items()
                     if k.startswith("connector.")}
        elif data.get("identity", {}).get("trainable") == "generation_connector_only":
            state = data["connector"]
        else:
            raise ValueError("C0 warm initialization requires an explicitly selected offline Connector")
        validate_state(model.connector, state)
        model.connector.load_state_dict(state, strict=True)
    else:
        if (data.get("version") != VERSION or data.get("stage") != "C0"
                or data["identity"]["manifest_sha256"] != manifest_sha
                or data["identity"]["planner_sha256"] != planner_sha
                or tuple(data["planner_names"]) != model.planner_names):
            raise ValueError("A needs this online C0/data/planner, not an offline checkpoint")
        state = {k: v for k, v in data["model"].items() if not k.startswith("adapters.")}
        expected = {k: v for k, v in model.state_dict().items() if not k.startswith("adapters.")}
        if set(state) != set(expected) or any(state[k].shape != expected[k].shape for k in state):
            raise ValueError("C0/A adaptation state mismatch")
        if any(a.b.weight.any() for a in model.adapters):
            raise ValueError("A must begin with zero-increment DiT LoRA")
        # Validate ALL incoming tensors before modifying the fresh A model.
        combined = dict(model.state_dict(), **state)
        validate_state(model, combined)
        model.load_state_dict(combined, strict=True)
    return data["step"]
