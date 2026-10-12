"""One live Omni forward, separate visible generation and masked feature branches."""
from contextlib import nullcontext
from dataclasses import dataclass
import math

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from omni_av.masking import hybrid_attention
from omni_av_p2.connector import SharedCondition, bernini_projection
from omni_av_p2.ltx import append_conditions, av_loss
from omni_av_p21.config import LoRAConfig
from omni_av_p21.lora import inject_text_lora
from omni_av_p3.model import Planner


@dataclass(frozen=True)
class OnlineConfig:
    mask_min: float = .5
    mask_max: float = .95
    all_mask_probability: float = .2
    planner_warmup_steps: int = 100
    feature_video_weight: float = 1.
    feature_audio_weight: float = 1.

    def __post_init__(self):
        if not 0 < self.mask_min <= self.mask_max < 1:
            raise ValueError("Partial mask ratio must be in (0, 1)")
        if not 0 <= self.all_mask_probability <= 1 or self.planner_warmup_steps < 0:
            raise ValueError("Invalid mask probability or warmup")
        if any(not math.isfinite(x) or x <= 0 for x in
               (self.feature_video_weight, self.feature_audio_weight)):
            raise ValueError("Feature weights must be finite and positive")


def sample_mask(layout, config, seed):
    """CPU RNG independent of model/device; retain visible tokens on partial rows."""
    rng = np.random.default_rng(seed)
    mask = torch.zeros_like(layout.valid)
    for b in range(len(mask)):
        if rng.random() < config.all_mask_probability:
            mask[b] = layout.valid[b]
            continue
        ratio = rng.uniform(config.mask_min, config.mask_max)
        for modality in (0, 1):
            ids = torch.where(layout.valid[b] & (layout.modalities[b] == modality))[0]
            if not len(ids):
                raise ValueError("Each sample requires audio and video targets")
            count = min(max(1, math.ceil(ratio * len(ids))), max(1, len(ids) - 1))
            take = torch.as_tensor(rng.permutation(len(ids))[:count], device=ids.device)
            mask[b, ids[take]] = True
    return mask


class OnlinePlanner(Planner):
    def readout(self, layout, masked, values):
        """Return prefix AND target hidden without detaching the live planning graph."""
        layout.validate()
        if masked.dtype != torch.bool or masked.shape != layout.valid.shape or (masked & ~layout.valid).any():
            raise ValueError("Invalid target mask")
        if values.shape != (*layout.valid.shape, self.cfg.hidden_size) or not torch.isfinite(values).all():
            raise ValueError("Invalid target features")
        mod = layout.modalities
        tokens = torch.where(masked[..., None], self.mask_token[mod].to(values.dtype), values.detach())
        tokens = (tokens + self.modality_embedding[mod].to(values.dtype)) * layout.valid[..., None]
        embeds = torch.cat([layout.prefix.detach(), tokens], 1)
        coords = layout.coordinates.clone()
        coords[..., 0] = (coords[..., 0] * self.cfg.time_ticks_per_second).round()
        offset = layout.prefix_positions.amax(dim=(0, 2)) + 1
        positions = torch.cat([layout.prefix_positions,
            coords.long().permute(2, 0, 1) + offset[None, :, None]], -1)
        attention = hybrid_attention(layout.prefix_valid, layout.valid, embeds.dtype)
        outputs = self.thinker.model(inputs_embeds=embeds, position_ids=positions,
            attention_mask=attention, output_hidden_states=True, use_cache=False, return_dict=True)
        hidden = outputs.hidden_states[-2]
        return hidden[:, :layout.prefix.shape[1]], hidden[:, layout.prefix.shape[1]:]

    def hidden(self, layout, masked, values):
        # Planner.sample calls this: predictions use the same online attention/positions.
        return self.readout(layout, masked, values)[1]


def select_contexts(prefix, target, layout, masked):
    """Compact per modality; no masked target hidden can enter LTX's context."""
    values = {}
    for mod, name in enumerate(("video", "audio")):
        rows = []
        for b in range(len(layout.valid)):
            visible = layout.valid[b] & ~masked[b] & (layout.modalities[b] == mod)
            rows.append(torch.cat([prefix[b, layout.prefix_valid[b]], target[b, visible]], 0))
        padded = pad_sequence(rows, batch_first=True)
        valid = torch.arange(padded.shape[1], device=padded.device)[None] < torch.tensor(
            [len(row) for row in rows], device=padded.device)[:, None]
        values[name], values[name + "_valid"] = padded, valid
    return SharedCondition(**values)


class GenerationConnector(nn.Module):
    def __init__(self, hidden_size=2048, video_dim=4096, audio_dim=2048):
        super().__init__()
        self.video = bernini_projection(hidden_size, video_dim, zero_init=True)
        self.audio = bernini_projection(hidden_size, audio_dim, zero_init=True)

    def forward(self, context):
        return SharedCondition(video=self.video(context.video.float()),
            audio=self.audio(context.audio.float()), video_valid=context.video_valid,
            audio_valid=context.audio_valid)


class OnlineTraining(nn.Module):
    def __init__(self, planner, renderer, strategy, stage, config=None, connector=None, lora=None):
        super().__init__()
        if stage not in ("C0", "A") or planner.arm != "lora":
            raise ValueError("Online C0/A requires the selected LoRA planner")
        self.stage, self.config = stage, config or OnlineConfig()
        # Register adaptation tensors only: frozen base models must not enter DDP/checkpoints.
        self.planner_names = tuple(n for n, p in planner.named_parameters() if p.requires_grad)
        if not self.planner_names or not any("lora_" in n for n in self.planner_names):
            raise ValueError("Construct and load planner before freezing it")
        params = dict(planner.named_parameters())
        planner.requires_grad_(False)
        self.planner_parameters = nn.ParameterList([params[n] for n in self.planner_names])
        self.planner_parameters.requires_grad_(stage == "A")
        self.connector = connector if connector is not None else GenerationConnector()
        renderer.requires_grad_(False)
        self.adapters, self.targets = (inject_text_lora(renderer, lora or LoRAConfig())
            if stage == "A" else (nn.ModuleList(), []))
        object.__setattr__(self, "planner", planner)
        object.__setattr__(self, "renderer", renderer)
        object.__setattr__(self, "strategy", strategy)
        self.assert_ownership()

    def train(self, mode=True):
        super().train(mode)
        self.planner.train(mode)
        self.renderer.train(mode)
        return self

    def assert_ownership(self):
        planner_ids = {id(p) for p in self.planner_parameters if p.requires_grad}
        if planner_ids != {id(p) for p in self.planner.parameters() if p.requires_grad}:
            raise RuntimeError("Unexpected planner trainables")
        dit_ids = {id(p) for p in self.adapters.parameters()}
        if dit_ids != {id(p) for p in self.renderer.parameters() if p.requires_grad}:
            raise RuntimeError("Unexpected DiT trainables")
        expected = planner_ids | dit_ids | {id(p) for p in self.connector.parameters()}
        if expected != {id(p) for p in self.parameters() if p.requires_grad}:
            raise RuntimeError("DDP ownership mismatch")

    def joint(self, step):
        return self.stage == "A" and step > self.config.planner_warmup_steps

    def forward(self, layout, targets, prepared, seed, step, masked=None):
        if targets.requires_grad:
            raise ValueError("GT perception targets must be frozen")
        masked = sample_mask(layout, self.config, seed) if masked is None else masked
        joint = self.joint(step)
        with nullcontext() if joint else torch.no_grad():
            prefix, hidden = self.planner.readout(layout, masked, targets)
        context = select_contexts(prefix, hidden, layout, masked)
        loss_av, parts = av_loss(self.renderer, self.strategy,
            append_conditions(prepared, self.connector(context)))
        losses = dict(av=loss_av, video=parts["video_loss"], audio=parts["audio_loss"])
        for mod, name in enumerate(("video", "audio")):
            feature = loss_av.new_zeros(())
            if joint:
                per_sample = []
                for b in range(len(layout.valid)):
                    chosen = masked[b] & layout.valid[b] & (layout.modalities[b] == mod)
                    if not chosen.any():
                        raise ValueError("Both modalities need masked supervision")
                    c = self.planner.connectors[mod](hidden[b, chosen].float())[None]
                    z = targets[b, chosen][None]
                    generator = torch.Generator(device=z.device).manual_seed(seed + 100 + mod + b * 1000)
                    per_sample.append(self.planner.decoders[mod].loss(z, c,
                        torch.ones(z.shape[:2], dtype=torch.bool, device=z.device), generator))
                feature = torch.stack(per_sample).mean()
            losses["feature_" + name] = feature
        losses["loss"] = (loss_av + self.config.feature_video_weight * losses["feature_video"]
            + self.config.feature_audio_weight * losses["feature_audio"])
        # Values for logging only; do not leak additional graph outputs to DDP.
        losses["masked_fraction"] = (masked.sum() / layout.valid.sum()).detach()
        return losses

    def optimizer_groups(self, connector_lr, dit_lr, planner_lr, heads_lr):
        groups = [dict(params=list(self.connector.parameters()), lr=connector_lr, name="connector")]
        if self.stage == "A":
            groups.append(dict(params=list(self.adapters.parameters()), lr=dit_lr, name="dit_text_lora"))
            for name, is_lora, lr in (("omni_lora", True, planner_lr), ("feature_heads", False, heads_lr)):
                params = [p for n, p in zip(self.planner_names, self.planner_parameters)
                    if n.startswith("thinker.") == is_lora]
                groups.append(dict(params=params, lr=lr, name=name))
        return groups

    @torch.no_grad()
    def predicted_conditions(self, layout, seed, planning_steps=25):
        """No GT feature argument: use current planner, then re-read fully filled slots."""
        if self.training:
            raise ValueError("Generation requires eval mode")
        features, remaining = self.planner.sample(layout, seed, steps=planning_steps)
        if remaining[-1] != 0:
            raise RuntimeError("Incomplete planning")
        unmasked = torch.zeros_like(layout.valid)
        prefix, hidden = self.planner.readout(layout, unmasked, features)
        return self.connector(select_contexts(prefix, hidden, layout, unmasked))
