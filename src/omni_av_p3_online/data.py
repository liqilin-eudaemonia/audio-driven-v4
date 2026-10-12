"""Reuse native feature/text/VAE caches, never read offline generation hidden."""
import hashlib
import io
from pathlib import Path

import torch

from omni_av.data import TextPlan
from omni_av.ltx_runtime import load_av_targets, audit_target_window
from omni_av_p3.model import Layout
from omni_av_p3_staged.integrity import load_record, verified_load, VERSION as INPUT_VERSION


def read_native(entry, device="cpu", include_targets=True):
    payload = Path(entry["native"]["path"]).read_bytes()
    if hashlib.sha256(payload).hexdigest() != entry["native"]["sha256"]:
        raise ValueError("Native feature hash mismatch")
    row = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
    if row["sample_id"] != entry["sample_id"] or row["split"] != entry["split"]:
        raise ValueError("Native sample identity mismatch")
    layout = Layout(**row["layout"]).to(device)
    layout.validate()
    if not include_targets:
        # File integrity still covers the original native record; GT tensors are not
        # passed to the prediction API or moved onto the accelerator.
        return layout
    targets = row["targets"].to(device).detach()
    if targets.shape != (*layout.valid.shape, layout.prefix.shape[-1]) or not torch.isfinite(targets).all():
        raise ValueError("Invalid frozen perception targets")
    return layout, targets


def load_sample(entry, integrity_root, manifest_sha, data_root, device, allow_unsealed=False):
    layout, targets = read_native(entry, device)
    record = load_record(integrity_root, entry, manifest_sha, data_root, allow_unsealed)
    text = verified_load(entry["text"], record["files"], map_location=device)
    if text["prompt"] != TextPlan(**entry["example"]["text_plan"]).render_prompt():
        raise ValueError("Cached text plan differs")
    av, metadata = load_av_targets(data_root, entry["sample_id"], device=device, split=entry["split"],
        tensor_loader=lambda p: verified_load(p, record["files"]))
    if metadata["metadata"] != entry["metadata"]:
        raise ValueError("VAE metadata changed")
    audit_target_window(entry, metadata)
    return layout, targets, text, av


def validate_manifest(manifest, seal, digest):
    ids = []
    for split, count in (("train", 91009), ("dev", 48)):
        rows = manifest[split]
        keys = [r["sample_id"] for r in rows]
        if len(keys) != count or len(set(keys)) != count or any(r["split"] != split for r in rows):
            raise ValueError("Full split coverage mismatch")
        ids.append(set(keys))
        if any(Path(k).name != k or k in (".", "..") for k in keys):
            raise ValueError("Unsafe sample ID")
    if ids[0] & ids[1]:
        raise ValueError("Train/Dev overlap")
    expected = {r["split"] + ":" + r["sample_id"] for s in ("train", "dev") for r in manifest[s]}
    if (seal.get("version") != INPUT_VERSION or seal["manifest_sha256"] != digest
            or set(seal["entries"]) != expected):
        raise ValueError("Full input seal mismatch")
