"""Frozen causal extraction. Teacher tensors are separate from Connector inputs."""
import gc
import json
import math
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from omni_av.data import Observation, TextPlan, file_hash, gt_text_plan, read_audio, save_new, select_training_examples, write_json_new
from omni_av.omni import encode_prefix, generate_reply, make_layout, move_inputs, prepare_observation
from omni_av.ltx_runtime import NativeTextConditioner, audit_target_window, load_av_targets
from . import VERSION
from .connector import ConditionInputs


def contract(cfg):
    root = Path(cfg.omni_path)
    return dict(version=VERSION, config=cfg.to_dict(),
                omni_config_sha256=file_hash(root / "config.json"),
                omni_index_sha256=file_hash(root / "model.safetensors.index.json"))


def unload():
    gc.collect()
    torch.cuda.empty_cache()


def load_omni(cfg, device="cuda:0"):
    from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration
    processor = Qwen2_5OmniProcessor.from_pretrained(cfg.omni_path, local_files_only=True)
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(cfg.omni_path,
        dtype=torch.bfloat16, device_map={"": device}, local_files_only=True, attn_implementation="sdpa")
    model.requires_grad_(False).eval()
    if model.config.text_config.hidden_size != cfg.hidden_size:
        raise ValueError("Omni hidden dimension mismatch")
    return SimpleNamespace(thinker=model, config=cfg), processor


@torch.no_grad()
def extract_condition(runtime, processor, obs, plan):
    prepared = prepare_observation(processor, obs, runtime.config)
    prefix = encode_prefix(runtime, processor, prepared, plan)
    model = runtime.thinker
    # Original causal text model, no target sequence and no new trainable Omni parameters.
    out = model.model(inputs_embeds=prefix["prefix"], attention_mask=prefix["prefix_valid"].long(),
                      position_ids=prefix["prefix_positions"], use_cache=False,
                      output_hidden_states=True, return_dict=True)
    grid = prepared["image_grid_thw"][0]
    merge = processor.image_processor.merge_size
    layout = make_layout(prefix, obs.output_duration_s,
                         (int(grid[1]) // merge, int(grid[2]) // merge), runtime.config)
    coords = [layout.coordinates[:, layout.modalities[0] == mod] for mod in (0, 1)]
    data = ConditionInputs(out.hidden_states[-2].detach(), prefix["prefix_valid"], coords[0], coords[1],
                           torch.ones(coords[0].shape[:2], device=coords[0].device, dtype=torch.bool),
                           torch.ones(coords[1].shape[:2], device=coords[1].device, dtype=torch.bool))
    data.validate()
    return data


def sample_target_frames(path, budget, duration, fps, size):
    import av
    samples, next_i, last = [], 0, None
    count = math.ceil(budget * fps)
    with av.open(path) as container:
        for frame in container.decode(video=0):
            timestamp = float(frame.time or 0)
            if timestamp >= duration - 1e-6:
                break
            if timestamp + 1e-6 >= next_i / fps:
                last = np.asarray(frame.to_image().resize(size))
                while next_i < count and timestamp + 1e-6 >= next_i / fps:
                    samples.append(last)
                    next_i += 1
            if next_i == count:
                break
    if last is None:
        raise ValueError("No target frames in the aligned window")
    while len(samples) < count + count % 2:
        samples.append(last)
    return np.stack(samples)


@torch.no_grad()
def extract_teacher(runtime, processor, condition, example, aligned_duration):
    cfg, model = runtime.config, runtime.thinker
    device = next(model.parameters()).device
    budget = example["observation"]["output_duration_s"]
    if not 0 < aligned_duration <= budget:
        raise ValueError("Teacher must fit externally chosen output budget")
    coords = condition.video_coordinates[0]
    patch = processor.image_processor.patch_size * processor.image_processor.merge_size
    size = (int(coords[:, 2].max() + 1) * patch, int(coords[:, 1].max() + 1) * patch)
    frames = sample_target_frames(example["target_video"], budget, aligned_duration, cfg.vision_fps, size)
    inputs = processor.video_processor(videos=[frames], return_tensors="pt", do_sample_frames=False, do_resize=False)
    inputs = move_inputs(dict(inputs), device)
    video = model.get_video_features(inputs["pixel_values_videos"], inputs["video_grid_thw"])[None]
    wave = read_audio(example["target_audio"], end_s=aligned_duration)
    wave = np.pad(wave, (0, round(budget * 16000) - len(wave)))
    audio = processor.feature_extractor(wave, sampling_rate=16000, return_tensors="pt",
                                        return_attention_mask=True, padding="max_length")
    audio = model.get_audio_features(audio["input_features"].to(device, torch.bfloat16), audio["attention_mask"].to(device))
    audio = F.adaptive_avg_pool1d(audio.float().T[None], condition.audio_coordinates.shape[1]).transpose(1, 2).to(video.dtype)
    # Full temporal support must lie within the aligned GT. Layout validity stays independent.
    vvalid = condition.video_coordinates[..., 0] + .5 / cfg.vision_fps < aligned_duration
    avalid = condition.audio_coordinates[..., 0] + .5 / cfg.audio_hz <= aligned_duration
    if video.shape != (*condition.video_valid.shape, cfg.target_dim) or audio.shape != (*condition.audio_valid.shape, cfg.target_dim):
        raise ValueError("Teacher feature/query grid mismatch")
    if not vvalid.any() or not avalid.any():
        raise ValueError("No valid teacher support")
    return dict(video=video, audio=audio, video_valid=vvalid, audio_valid=avalid)


def prepare_cache(args, cfg):
    examples = json.loads(args.manifest.read_text()) if args.manifest else select_training_examples(args.data_root, args.samples, args.budget)
    if not examples or any(x["split"] != "train" for x in examples):
        raise ValueError("Preparation currently requires nonempty Train examples")
    ids = [x["observation"]["sample_id"] for x in examples]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate IDs")
    runtime, processor = load_omni(cfg)
    paths = []
    for i, example in enumerate(examples):
        obs, plan = Observation(**example["observation"]), TextPlan(**example["text_plan"])
        gt = json.loads(Path(example["gt_path"]).read_text())
        if plan != gt_text_plan(gt, obs.sample_id):
            raise ValueError("Text differs from paired GT")
        _, metadata = load_av_targets(args.data_root, obs.sample_id, device="cpu")
        window = audit_target_window(dict(example=example), metadata)
        condition = extract_condition(runtime, processor, obs, plan)
        targets = extract_teacher(runtime, processor, condition, example, window["renderer_target_duration_s"])
        path = args.output / f"features_{i:04d}.pt"
        save_new(path, dict(contract=contract(cfg), split="train", sample_id=obs.sample_id, example=example,
            condition={k: v.cpu() for k, v in vars(condition).items()},
            targets={k: v.cpu() for k, v in targets.items()}, window=window,
            gt_sha256=file_hash(example["gt_path"]), source="paired_gt_text_frozen_omni"))
        paths.append(dict(features=str(path.resolve())))
        print(json.dumps(dict(event="p2_features", sample_id=obs.sample_id,
                              hidden=list(condition.hidden.shape), video=list(targets["video"].shape),
                              audio=list(targets["audio"].shape), window=window)), flush=True)
    del runtime, processor, condition, targets
    unload()
    text_runtime = NativeTextConditioner(cfg)
    try:
        for i, (example, entry) in enumerate(zip(examples, paths)):
            text = text_runtime.encode(TextPlan(**example["text_plan"]))
            path = args.output / f"text_{i:04d}.pt"
            save_new(path, {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in text.items()})
            entry["text"] = str(path.resolve())
            print(json.dumps(dict(event="p2_text", index=i, video=list(text["video"].shape), audio=list(text["audio"].shape))), flush=True)
    finally:
        text_runtime.close()
    index = dict(contract=contract(cfg), entries=paths)
    write_json_new(args.output / "cache_index.json", index)
    return index


def read_sample(entry, cfg, device="cuda:0"):
    payload = torch.load(entry["features"], map_location="cpu", weights_only=True)
    if payload["contract"] != contract(cfg) or payload["split"] != "train":
        raise ValueError("P2 training cache identity mismatch")
    condition = ConditionInputs(**payload["condition"]).to(device)
    condition.validate()
    targets = {k: v.to(device) for k, v in payload["targets"].items()}
    text = torch.load(entry["text"], map_location=device, weights_only=True)
    if text["prompt"] != TextPlan(**payload["example"]["text_plan"]).render_prompt():
        raise ValueError("Text/hidden plan mismatch")
    return condition, targets, text, payload
