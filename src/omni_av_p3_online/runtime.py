"""Shared production construction for training and checkpoint-based generation."""
import torch
from omni_av_p3.model import Config as PlannerConfig
from omni_av_p3.data import load_native
from omni_av_p3.diagnostics import restore
from omni_av_p3.distributed import sha
from omni_av_p2.config import Config as RendererConfig
from omni_av_p2.ltx import load_renderer
from omni_av_p21.config import LoRAConfig
from omni_av_p21.residual_performance import configure_grouped_checkpointing
from .model import OnlineConfig, OnlinePlanner, OnlineTraining, GenerationConnector


def build(manifest, recipe, stage, device):
    planner_config = PlannerConfig(**manifest["native_contract"]["config"])
    source = manifest["config"]["planner"]
    if sha(source["path"]) != source["sha256"]:
        raise ValueError("Initial planner checkpoint changed")
    runtime, processor = load_native(planner_config, device)
    del processor
    planner = OnlinePlanner(runtime.thinker, planner_config, "lora")
    planner.connectors.to(device)
    planner.decoders.to(device)
    planner.mask_token.data = planner.mask_token.data.to(device)
    planner.modality_embedding.data = planner.modality_embedding.data.to(device)
    restore(planner, torch.load(source["path"], map_location="cpu", weights_only=True))
    planner.thinker.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    # Input embeddings are already cached; frozen perception towers/lm_head are unused.
    for name in ("visual", "audio_tower", "lm_head"):
        module = getattr(planner.thinker, name, None)
        if module is not None:
            module.cpu()
    renderer_config = RendererConfig(**recipe.get("renderer", {}))
    if stage == "P":
        renderer, strategy, sampler = torch.nn.Identity(), None, None
    else:
        renderer, strategy, sampler = load_renderer(renderer_config, device=device)
        configure_grouped_checkpointing(renderer, recipe.get("checkpoint_group_size", 1))
    model = OnlineTraining(planner, renderer, strategy, stage,
        OnlineConfig(**recipe["online"]), None if stage == "P" else GenerationConnector().to(device),
        LoRAConfig(rank=recipe["lora_rank"], alpha=recipe["lora_alpha"]))
    return model, strategy, sampler, renderer_config


def update_learning_rates(optimizer, recipe, step, joint, stage="A"):
    mapping = dict(connector="connector_lr", dit_text_lora="dit_lr",
                   omni_lora="planner_lr", feature_heads="heads_lr")
    for group in optimizer.param_groups:
        lr = recipe[mapping[group["name"]]]
        if group["name"] == "dit_text_lora":
            lr *= min(1., step / 50)
        elif group["name"] in ("omni_lora", "feature_heads"):
            lr *= min(1., step / 50) if stage == "P" else (min(1., (step - recipe["online"]["planner_warmup_steps"]) / 50) if joint else 0.)
        group["lr"] = lr
