"""torchrun online P/C0/A; explicit stage launch, independent output and strict resume."""
import argparse
from contextlib import nullcontext
from dataclasses import asdict, replace
from datetime import timedelta
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True, help="Existing P3 staged manifest")
    p.add_argument("--integrity-root", type=Path, required=True, help="Existing readouts/inputs directory")
    p.add_argument("--config", type=Path, default=ROOT / "configs/pipeline3_bernini_online.json")
    p.add_argument("--stage", choices=("P", "C0", "A"), required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--control-root", type=Path, required=True, help="NEW online run root, not the old queue root")
    init = p.add_mutually_exclusive_group(required=True)
    init.add_argument("--initialize", type=Path)
    init.add_argument("--resume", type=Path)
    p.add_argument("--initialization-sha256", required=True, help="Original warm-start checkpoint hash, also on resume")
    p.add_argument("--engineering-indices", type=Path, help="Explicit train subset; cannot initialize production A")
    p.add_argument("--engineering-steps", type=int, default=3)
    p.add_argument("--stop-after", type=int, help="Save and stop at this update; does not change resume identity")
    p.add_argument("--engineering-mask", choices=("beta", "all_mask"), default="beta")
    a = p.parse_args()
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.tensorboard import SummaryWriter
    from omni_av.ltx_runtime import with_text
    from omni_av_p2.checkpoint import capture_rng, restore_rng, atomic_json_new
    from omni_av_p2.distributed import parameter_digest
    from omni_av_p3.distributed import GlobalOrder, BatchAssignments, master_action, sha
    from omni_av_p3_staged.orchestration import read, write
    from omni_av_p3_online import VERSION
    from omni_av_p3_online.model import sample_mask, OnlineConfig
    from omni_av_p3_online.data import load_sample, read_native, validate_manifest
    from omni_av_p3_online.checkpoint import checked, initialize, initialize_connector, resume, save
    from omni_av_p3_online.runtime import build, update_learning_rates

    recipe, manifest = read(a.config), read(a.manifest)
    if recipe["version"] != VERSION:
        raise ValueError("Online recipe version mismatch")
    OnlineConfig(**recipe["online"])
    engineering = a.engineering_indices is not None
    world, rank, local = (int(os.environ[k]) for k in ("WORLD_SIZE", "RANK", "LOCAL_RANK"))
    accumulation = recipe["accumulation"]
    if not engineering and a.engineering_mask != "beta":
        raise ValueError("Mask override is engineering-only")
    if not engineering and (world != recipe["world_size"] or world * accumulation != recipe["global_batch"]):
        raise ValueError("World/global batch mismatch")
    if (a.control_root / "STOP").exists() or (a.output / "STOP").exists():
        raise RuntimeError("Persistent STOP; not starting")
    digest = sha(a.manifest)
    seal = a.integrity_root / "complete.json"
    if not engineering:
        validate_manifest(manifest, read(seal), digest)
    entries = manifest["train"]
    if engineering:
        indices = read(a.engineering_indices)
        if not indices or len(set(indices)) != len(indices) or any(type(i) != int or not 0 <= i < len(entries) for i in indices):
            raise ValueError("Invalid explicit engineering indices")
        entries = [entries[i] for i in indices]
    steps = a.engineering_steps if engineering else recipe["steps"][a.stage]
    if steps < 1 or accumulation < 1 or recipe["save_every"] < 1 or recipe["validate_every"] < 0:
        raise ValueError("Invalid training intervals")
    if a.initialize and a.stage in ("C0", "A"):
        source = checked(a.initialize)
        if not engineering and source.get("identity", {}).get("engineering"):
            raise ValueError("Engineering checkpoint cannot initialize production C0/A")
    torch.set_num_threads(2)
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    dist.init_process_group("nccl", timeout=timedelta(minutes=15), device_id=device)
    writer = None
    try:
        master_action(lambda: a.output.mkdir(parents=True, exist_ok=False), rank)
        torch.manual_seed(recipe["seed"])
        raw, strategy, sampler, renderer_config = build(manifest, recipe, a.stage, device)
        optimizer = torch.optim.AdamW(raw.optimizer_groups(recipe["connector_lr"], recipe["dit_lr"],
            recipe["planner_lr"], recipe["heads_lr"]), fused=True, weight_decay=0.)
        identity = dict(version=VERSION, manifest_sha256=digest, config_sha256=sha(a.config),
            recipe=recipe, world_size=world, accumulation=accumulation, stage=a.stage, steps=steps,
            planner_sha256=manifest["config"]["planner"]["sha256"],
            planner_config=manifest["native_contract"]["config"], renderer_config=asdict(renderer_config),
            initialization_sha256=a.initialization_sha256, input_seal_sha256=sha(seal) if seal.exists() else None,
            engineering=engineering, engineering_mask=a.engineering_mask, subset_sha256=sha(a.engineering_indices) if engineering else None,
            planner_names=raw.planner_names, targets=raw.targets,
            code={str(f.relative_to(ROOT)): sha(f) for f in sorted([
                *ROOT.glob("src/**/*.py"), ROOT / "scripts/pipeline3_online_train.py"])})
        identities = [None] * world
        dist.all_gather_object(identities, identity)
        if any(value != identity for value in identities):
            raise ValueError("Ranks disagree on source/data/recipe identity")
        start, saved_rng = 0, None
        if a.resume:
            data = resume(a.resume, raw, optimizer, identity)
            start, saved_rng = data["step"], data["rank_rng"][rank]
        else:
            initialize(a.initialize, raw, digest, a.initialization_sha256, identity["planner_sha256"])
            if a.stage == "C0":
                c = recipe["connector_initialization"]
                initialize_connector(c["path"], raw, c["sha256"])
        if start > steps:
            raise ValueError("Checkpoint exceeds target")
        # Planner parameters are deliberately unused during A warmup. Masks can
        # also leave a modality without visible targets; DDP must handle both.
        model = DDP(raw, device_ids=[local], broadcast_buffers=False, gradient_as_bucket_view=True,
                    find_unused_parameters=True)
        if saved_rng is not None:
            restore_rng(saved_rng)
        raw.train()
        frozen = [(p, p._version) for m in (raw.planner, raw.renderer) for p in m.parameters() if not p.requires_grad]
        order = GlobalOrder(len(entries), recipe["seed"])
        costs = [(((e["metadata"]["target_frames"] - 1) // 8 + 1)
            * (e["metadata"]["target_height"] // 32) * (e["metadata"]["target_width"] // 32)) ** 2 for e in entries]
        assignments = BatchAssignments(order, costs, world, accumulation, True)
        if rank == 0:
            writer = SummaryWriter(str(a.output / "tensorboard"))
            atomic_json_new(a.output / "run.json", dict(identity=identity, start=start,
                initialize=str(a.initialize), resume=str(a.resume)))

        def checkpoint(step):
            rngs, hashes = [None] * world, [None] * world
            dist.all_gather_object(rngs, capture_rng())
            dist.all_gather_object(hashes, parameter_digest(raw))
            if len(set(hashes)) != 1:
                raise RuntimeError("Rank parameters diverged")
            master_action(lambda: save(a.output / f"step_{step:06d}.pt", raw, optimizer, step, identity, rngs), rank)

        def sample(entry):
            if a.stage == "P":
                layout, target = read_native(entry, device)
                return layout, target, None, None
            return load_sample(entry, a.integrity_root, digest, manifest["config"]["data_root"], device, engineering)

        def prepare(av, text):
            return None if a.stage == "P" else strategy.prepare_training_inputs(with_text(av, text), sampler)

        def evaluate(step):
            rng = capture_rng()
            raw.eval()
            rows = []
            try:
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    for i in range(rank, len(manifest["dev"]), world):
                        e = manifest["dev"][i]
                        layout, target, text, av = sample(e)
                        for mode in ("beta", "all_mask"):
                            seed = recipe["seed"] + 100000 + i
                            torch.manual_seed(seed)
                            mask = layout.valid.clone() if mode == "all_mask" else sample_mask(
                                layout, raw.config, seed)
                            values = raw(layout, target, prepare(av, text),
                                seed, step, mask)
                            rows.append(dict(sample_id=e["sample_id"], mode=mode,
                                values={k: float(v) for k, v in values.items()}))
                gathered = [None] * world
                dist.all_gather_object(gathered, rows)
                def publish():
                    allrows = [row for group in gathered for row in group]
                    atomic_json_new(a.output / f"dev_{step:06d}.json", dict(step=step, rows=allrows,
                        note="Beta/all-mask objective diagnostics, not Pred25 generation quality"))
                    for mode in ("beta", "all_mask"):
                        values = [r["values"] for r in allrows if r["mode"] == mode]
                        for k in values[0]:
                            writer.add_scalar(f"dev/{mode}/{k}", sum(v[k] for v in values) / len(values), step)
                master_action(publish, rank)
            finally:
                restore_rng(rng)
                raw.train()

        if not engineering and a.stage == "P" and start == 0 and not a.resume:
            checkpoint(0)
            evaluate(0)

        # A checkpoint is published before Dev evaluation. Recompute the due
        # evaluation on resume, including start == steps, before completion.
        if not engineering and (start > 0 or (a.stage == "P" and a.resume)) and recipe["validate_every"] and start % recipe["validate_every"] == 0:
            evaluate(start)

        last, saved = start, None
        with (a.output / f"rank{rank}.jsonl").open("x") as log:
            for step in range(start + 1, steps + 1):
                stop = torch.tensor(int((a.control_root / "STOP").exists() or (a.output / "STOP").exists()
                    or (a.stop_after is not None and step > a.stop_after)), device=device)
                dist.all_reduce(stop, op=dist.ReduceOp.MAX)
                if stop.item():
                    break
                begin = time.monotonic()
                optimizer.zero_grad(set_to_none=True)
                update_learning_rates(optimizer, recipe, step, raw.joint(step), stage=a.stage)
                sums = {}
                for micro in range(accumulation):
                    ordinal = assignments[step][micro][rank]
                    entry = entries[order[ordinal]]
                    seed = recipe["seed"] + ordinal
                    write(a.output / f"progress_rank{rank}.json", dict(step=step, micro=micro,
                        phase="load", sample_id=entry["sample_id"], time=time.time()))
                    layout, target, text, av = sample(entry)
                    torch.manual_seed(seed)
                    with model.no_sync() if micro < accumulation - 1 else nullcontext():
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            values = model(layout, target, prepare(av, text), seed, step,
                                layout.valid.clone() if engineering and a.engineering_mask == "all_mask" else None)
                        if not all(torch.isfinite(v).all() for v in values.values()):
                            raise FloatingPointError("Nonfinite online objective")
                        (values["loss"] / accumulation).backward()
                    for k, v in values.items():
                        sums[k] = sums.get(k, 0.) + v.detach().float() / accumulation
                    del layout, target, text, av, values
                    write(a.output / f"progress_rank{rank}.json", dict(step=step, micro=micro,
                        phase="backward", time=time.time()))
                grads = {g["name"]: float(torch.stack([p.grad.detach().float().square().sum()
                    for p in g["params"] if p.grad is not None]).sum().sqrt())
                    if any(p.grad is not None for p in g["params"]) else 0. for g in optimizer.param_groups}
                norm = torch.nn.utils.clip_grad_norm_(raw.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                raw.assert_ownership()
                if any(p.grad is not None or p._version != version for p, version in frozen):
                    raise RuntimeError("Frozen base changed")
                packed = torch.stack(list(sums.values()))
                dist.all_reduce(packed)
                torch.cuda.synchronize()
                row = dict(step=step, stage=a.stage, joint=raw.joint(step), seconds=time.monotonic() - begin,
                    values=dict(zip(sums, (packed / world).tolist())), grad_norm=float(norm), group_grad_norms=grads)
                log.write(json.dumps(row) + "\n")
                log.flush()
                last = step
                if rank == 0:
                    print(json.dumps(row), flush=True)
                    for k, v in row["values"].items():
                        writer.add_scalar("train/" + k, v, step)
                    for k, v in grads.items():
                        writer.add_scalar("grad/" + k, v, step)
                    writer.add_scalar("speed/seconds", row["seconds"], step)
                if step % recipe["save_every"] == 0 or step == steps or step == a.stop_after:
                    checkpoint(step)
                    saved = step
                if not engineering and recipe["validate_every"] and step % recipe["validate_every"] == 0:
                    evaluate(step)
            if saved != last:
                checkpoint(last)
        master_action(lambda: atomic_json_new(a.output / "completed.json", dict(stage=a.stage, step=last,
            stopped=last < steps, engineering=engineering, next_stage_launched=False)), rank)
    finally:
        if writer:
            writer.close()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
