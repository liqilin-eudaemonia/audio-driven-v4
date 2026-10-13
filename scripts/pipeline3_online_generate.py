"""Generate using the online checkpoint's OWN planner, never old Pred25 hidden."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--job", type=Path, required=True)
    p.add_argument("--gpu", type=int, required=True)
    a = p.parse_args()
    from pipeline3_staged_worker import probe
    if any(g["pids"] or g["memory_mib"] >= 1000 or g["util"] > 5 for g in probe([a.gpu])["gpus"]):
        raise RuntimeError("Selected GPU is occupied")
    os.environ.update(CUDA_VISIBLE_DEVICES=str(a.gpu), OMP_NUM_THREADS="2", TOKENIZERS_PARALLELISM="false")
    import torch
    from PIL import Image
    from omni_av.data import Observation, TextPlan
    from omni_av.ltx_runtime import with_text
    from omni_av_p2.checkpoint import atomic_save
    from omni_av_p2.data import unload
    from omni_av_p2.ltx import append_conditions, unpack_latents
    from omni_av_p2.inference import reference_latents
    from omni_av_p3.distributed import sha
    from omni_av_p3_staged.integrity import load_record, verified_load
    from omni_av_p3_staged.sampling import guided_sample
    from omni_av_p3_staged.media import decode_and_save
    from omni_av_p3_staged.orchestration import read, write
    from omni_av_p3_online import VERSION
    from omni_av_p3_online.checkpoint import checked, restore_model
    from omni_av_p3_online.data import read_native
    from omni_av_p3_online.model import select_contexts
    from omni_av_p3_online.runtime import build

    torch.set_num_threads(2)
    job = read(a.job)
    if job["version"] != VERSION or job["mode"] not in ("pred25", "oracle"):
        raise ValueError("Unknown online generation contract")
    for path_key, digest_key in (("manifest", "manifest_sha256"), ("checkpoint", "checkpoint_sha256"),
                                ("negative_text", "negative_sha256")):
        if sha(job[path_key]) != job[digest_key]:
            raise ValueError("Generation source changed: " + path_key)
    if sha(Path(job["integrity_root"]) / "complete.json") != job["input_integrity_sha256"]:
        raise ValueError("Generation integrity seal changed")
    out = Path(job["output"])
    out.mkdir(parents=True, exist_ok=True)
    if (out / "completed.json").exists():
        done = read(out / "completed.json")
        if done["job"] != job or done["video_sha256"] != sha(out / "generated.mp4"):
            raise ValueError("Completed generation identity mismatch")
        return
    manifest = read(job["manifest"])
    cp = checked(job["checkpoint"])
    if cp.get("version") != VERSION or cp["identity"]["manifest_sha256"] != job["manifest_sha256"]:
        raise ValueError("Use an online checkpoint with its own manifest")
    if cp["identity"]["planner_sha256"] != manifest["config"]["planner"]["sha256"]:
        raise ValueError("Initial planner contract changed")
    if any(not (ROOT / name).is_file() or sha(ROOT / name) != value
           for name, value in cp["identity"]["code"].items()):
        raise ValueError("Generation must use the checkpoint's bound training source")
    if cp["stage"] == "P":
        raise ValueError("P is feature-only; generate with an adapted C0/A checkpoint")
    renderer_settings = cp["identity"]["renderer_config"]
    entry = manifest["dev"][job["index"]]
    meta = entry["metadata"]
    geometry = dict(frames=meta["target_frames"], height=meta["target_height"], width=meta["target_width"],
        fps=meta["fps"], duration_s=meta["target_frames"] / meta["fps"])
    with Image.open(entry["example"]["observation"]["listener_image"]) as im:
        if im.size != (geometry["width"], geometry["height"]):
            raise ValueError("Original aspect/geometry must be preserved")
    latent_path = out / "sampled_latents.pt"
    if not latent_path.exists():
        record = load_record(job["integrity_root"], entry, job["manifest_sha256"], manifest["config"]["data_root"])
        text = verified_load(entry["text"], record["files"], map_location="cuda:0")
        if text["prompt"] != TextPlan(**entry["example"]["text_plan"]).render_prompt():
            raise ValueError("Plan text changed")
        negative = torch.load(job["negative_text"], map_location="cuda:0", weights_only=True)
        raw, strategy, sampler, cfg = build(manifest, cp["identity"]["recipe"], cp["stage"], "cuda:0")
        restore_model(cp, raw)
        raw.requires_grad_(False).eval()
        del cp
        u = None
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            if job["mode"] == "oracle":
                layout, target = read_native(entry, "cuda:0")
                unmasked = torch.zeros_like(layout.valid)
                prefix, hidden = raw.planner.readout(layout, unmasked, target)
                u = raw.connector(select_contexts(prefix, hidden, layout, unmasked))
                del target, prefix, hidden
            else:
                layout = read_native(entry, "cuda:0", include_targets=False)
                u = raw.predicted_conditions(layout, job["seed"], planning_steps=25)
            del layout
        # The online readout is complete. Free Planner GPU weights before LTX sampling.
        raw.planner.cpu()
        unload()
        av = reference_latents(cfg, Observation(**entry["example"]["observation"]), geometry, device="cuda:0")
        torch.manual_seed(job["seed"])
        prepared = strategy.prepare_training_inputs(with_text(av, text), sampler)
        if u is not None:
            prepared = append_conditions(prepared, u)
        torch.manual_seed(job["seed"])
        sampled = guided_sample(raw.renderer, prepared, negative, steps=30, video_cfg=3., audio_cfg=7.)
        video, audio = unpack_latents(sampled, av)
        atomic_save(latent_path, dict(job=job, video=video.cpu(), audio=audio.cpu()))
        del raw, av, text, negative, u, prepared, sampled, video, audio
        unload()
    saved = torch.load(latent_path, map_location="cpu", weights_only=True)
    if saved["job"] != job:
        raise ValueError("Saved latent identity mismatch")
    # cfg is also needed when resuming after completed sampling.
    from omni_av_p2.config import Config
    # The actual renderer config is bound in the checkpoint identity, not the job.
    cfg = Config(**renderer_settings)
    pending = out / "generated.pending.mp4"
    pending.unlink(missing_ok=True)
    decoded = decode_and_save(cfg, saved["video"], saved["audio"], geometry["fps"], pending,
                              raw_audio_path=out / "raw_audio.pt")
    if any(decoded[k] != geometry[k] for k in ("frames", "height", "width")):
        raise ValueError("Decoded geometry changed")
    pending.replace(out / "generated.mp4")
    wave = torch.load(out / "raw_audio.pt", weights_only=True)["wave"].float()
    write(out / "completed.json", dict(job=job, sample_id=entry["sample_id"], decoded=decoded,
        video_sha256=sha(out / "generated.mp4"), target_features_used=job["mode"] == "oracle",
        target_vae_used=False, fixed_gt_reply_timing=True, sampler="nativeEuler30 CFGv3/a7",
        planner_checkpoint_sha256=job["checkpoint_sha256"],
        waveform=dict(finite=bool(torch.isfinite(wave).all()), pre_clamp_peak=float(wave.abs().max()),
                      pre_clamp_over_one_fraction=float((wave.abs() > 1).float().mean()))))


if __name__ == "__main__":
    main()
