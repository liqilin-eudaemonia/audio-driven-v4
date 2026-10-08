"""Shared U enters native frozen LTX text attention; no new DiT trainables."""
from dataclasses import replace
import math
from types import SimpleNamespace

import torch

from omni_av.ltx_runtime import render_loss, sample_latents


def append_conditions(prepared, u):
    values = {}
    for name in ("video", "audio"):
        modality = getattr(prepared, name)
        context = getattr(u, name)
        valid = getattr(u, name + "_valid")
        if modality.context.shape[-1] != context.shape[-1]:
            raise ValueError("Native context/U dimension mismatch")
        mask = modality.context_mask
        if mask is None:
            mask = torch.ones(modality.context.shape[:2], device=context.device, dtype=torch.bool)
        if mask.dtype != torch.bool or mask.ndim != 2:
            raise ValueError("Append U before native context mask conversion")
        values[name] = replace(modality, context=torch.cat([modality.context, context.to(modality.context.dtype)], 1),
                               context_mask=torch.cat([mask, valid], 1))
    return replace(prepared, **values)


def load_renderer(cfg, device="cuda:0"):
    from ltx_core.model.transformer.attention import Attention, PytorchAttention
    from ltx_trainer.model_loader import load_transformer
    from ltx_trainer.timestep_samplers import UniformTimestepSampler
    from ltx_trainer.training_strategies.text_to_video import TextToVideoConfig, TextToVideoStrategy
    renderer = load_transformer(cfg.ltx_path, device=device, dtype=torch.bfloat16).requires_grad_(False)
    renderer.set_gradient_checkpointing(True)
    for module in renderer.modules():
        if isinstance(module, Attention):
            module.attention_function = PytorchAttention()
    strategy = TextToVideoStrategy(TextToVideoConfig(with_audio=True, first_frame_conditioning_p=1, i2av_only=True))
    return renderer, strategy, UniformTimestepSampler()


def av_loss(renderer, strategy, inputs):
    return render_loss(SimpleNamespace(renderer=renderer), strategy, inputs)


def sample(renderer, prepared, steps):
    class Wrapper:
        def __init__(self, model):
            self.renderer = model
        def eval(self):
            self.renderer.eval()
    return sample_latents(Wrapper(renderer), prepared, steps)


def unpack_latents(sampled, av):
    video_shape = av["latents"]["latents"].shape
    audio_shape = av["audio_latents"]["latents"].shape
    b, c, f, h, w = video_shape
    video = sampled["video"].reshape(b, f, h, w, c).permute(0, 4, 1, 2, 3).contiguous()
    b, c, t, freq = audio_shape
    audio = sampled["audio"].reshape(b, t, c, freq).permute(0, 2, 1, 3).contiguous()
    return video, audio


@torch.no_grad()
def decode_and_save(cfg, video, audio, fps, path, device="cuda:0"):
    from ltx_trainer.model_loader import load_audio_vae_decoder, load_video_vae_decoder, load_vocoder
    from ltx_trainer.video_utils import save_video
    from .data import unload
    if path.exists():
        raise FileExistsError(path)
    decoder = load_video_vae_decoder(cfg.ltx_path, device, torch.bfloat16).requires_grad_(False).eval()
    pixels = decoder(video.to(device, torch.bfloat16))
    if not torch.isfinite(pixels).all():
        raise FloatingPointError("Invalid decoded video")
    pixels = ((pixels[0].float() + 1) / 2).clamp(0, 1).cpu()
    del decoder
    unload()
    decoder = load_audio_vae_decoder(cfg.ltx_path, device, torch.bfloat16).requires_grad_(False).eval()
    mel = decoder(audio.to(device, torch.bfloat16))
    del decoder
    unload()
    vocoder = load_vocoder(cfg.ltx_path, device, torch.bfloat16).requires_grad_(False).eval()
    wave = vocoder(mel).squeeze(0).float().cpu()
    if not torch.isfinite(wave).all():
        raise FloatingPointError("Invalid decoded audio")
    rate = int(vocoder.output_sampling_rate)
    del vocoder, mel
    unload()
    count = round(pixels.shape[1] / fps * rate)
    padding = max(0, count - wave.shape[-1])
    if padding > math.ceil(rate / 25):
        raise ValueError("Decoded audio shorter by more than one audio-latent tick")
    if padding:
        wave = torch.nn.functional.pad(wave, (0, padding))
    wave = wave[..., :count]
    save_video(pixels, path, fps=fps, audio=wave, audio_sample_rate=rate, video_format="CFHW")
    return dict(frames=pixels.shape[1], height=pixels.shape[2], width=pixels.shape[3],
                audio_samples=wave.shape[-1], audio_rate=rate, rounding_padding_samples=padding,
                duration_s=pixels.shape[1] / fps)
