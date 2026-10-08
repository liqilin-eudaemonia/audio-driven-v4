# Projection topology adapted from Bernini; see THIRD_PARTY_NOTICES.md.
from dataclasses import dataclass
import math

import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        dtype = x.dtype
        normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps)
        return (normalized * self.weight).to(dtype)


def bernini_projection(in_dim, out_dim, zero_init=True):
    result = nn.Sequential(nn.Linear(in_dim, out_dim), nn.GELU(), RMSNorm(out_dim), nn.Linear(out_dim, out_dim))
    if zero_init:
        nn.init.zeros_(result[-1].weight)
        nn.init.zeros_(result[-1].bias)
    return result


@dataclass
class ConditionInputs:
    hidden: torch.Tensor
    hidden_valid: torch.Tensor
    video_coordinates: torch.Tensor
    audio_coordinates: torch.Tensor
    video_valid: torch.Tensor  # output-layout padding only, never GT availability
    audio_valid: torch.Tensor

    def validate(self):
        b, n, _ = self.hidden.shape
        if self.hidden_valid.shape != (b, n) or self.hidden_valid.dtype != torch.bool:
            raise ValueError("Invalid hidden mask")
        if not self.hidden_valid.any(-1).all() or not torch.isfinite(self.hidden).all():
            raise ValueError("Invalid hidden context")
        for mod in ("video", "audio"):
            coords, mask = getattr(self, mod + "_coordinates"), getattr(self, mod + "_valid")
            if coords.ndim != 3 or coords.shape[0] != b or coords.shape[-1] != 3:
                raise ValueError("Invalid coordinate shape")
            if mask.shape != coords.shape[:2] or mask.dtype != torch.bool or not mask.any(-1).all():
                raise ValueError("Invalid layout mask")
            if not torch.isfinite(coords).all() or (coords < 0).any():
                raise ValueError("Invalid coordinates")

    def to(self, device):
        return ConditionInputs(**{k: v.to(device) for k, v in vars(self).items()})


@dataclass
class SharedCondition:
    video: torch.Tensor
    audio: torch.Tensor
    video_valid: torch.Tensor
    audio_valid: torch.Tensor


class FusionBlock(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.cross_norm = nn.LayerNorm(width)
        self.cross = nn.MultiheadAttention(width, heads, dropout=0, batch_first=True)
        self.self_norm = nn.LayerNorm(width)
        self.self_attn = nn.MultiheadAttention(width, heads, dropout=0, batch_first=True)
        self.ff_norm = nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width))

    def forward(self, q, h, h_valid, q_valid):
        q = q + self.cross(self.cross_norm(q), h, h, key_padding_mask=~h_valid, need_weights=False)[0]
        n = self.self_norm(q)
        q = q + self.self_attn(n, n, n, key_padding_mask=~q_valid, need_weights=False)[0]
        return (q + self.ff(self.ff_norm(q))) * q_valid[..., None]


class FusionConnector(nn.Module):
    """Coordinate queries add fusion; final projections match Bernini for_gen topology."""
    def __init__(self, config):
        super().__init__()
        width = config.fusion_width
        self.input = nn.Sequential(nn.LayerNorm(config.hidden_size), nn.Linear(config.hidden_size, width))
        self.coordinate_projection = nn.Linear(3 * 2 * 16, width)
        self.modality = nn.Parameter(torch.randn(2, width) * .02)
        self.blocks = nn.ModuleList([FusionBlock(width, config.fusion_heads) for _ in range(config.fusion_depth)])
        self.video_output = bernini_projection(width, config.video_dim, config.zero_init_connector)
        self.audio_output = bernini_projection(width, config.audio_dim, config.zero_init_connector)

    def queries(self, coordinates, modality):
        frequencies = torch.exp(-math.log(10000) * torch.arange(16, device=coordinates.device).float() / 16)
        angles = coordinates.float()[..., None] * frequencies * (2 * math.pi)
        encoded = torch.cat([angles.sin(), angles.cos()], -1).flatten(-2)
        return self.coordinate_projection(encoded) + self.modality[modality]

    def forward(self, data: ConditionInputs):
        data.validate()
        h = self.input(data.hidden.detach())
        nv = data.video_coordinates.shape[1]
        q = torch.cat([self.queries(data.video_coordinates, 0), self.queries(data.audio_coordinates, 1)], 1)
        mask = torch.cat([data.video_valid, data.audio_valid], 1)
        for block in self.blocks:
            q = block(q, h, data.hidden_valid, mask)
        return SharedCondition(self.video_output(q[:, :nv]) * data.video_valid[..., None],
                               self.audio_output(q[:, nv:]) * data.audio_valid[..., None],
                               data.video_valid, data.audio_valid)
