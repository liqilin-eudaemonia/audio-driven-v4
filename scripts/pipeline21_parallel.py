#!/usr/bin/env python3
"""Launch P2.1 DDP or matched single-GPU accumulation on explicitly selected idle GPUs."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    p.add_argument("--cache", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--checkpoint-dir", type=Path, help="Dedicated directory for this run weights/state")
    p.add_argument("--config", type=Path)
    p.add_argument("--resume", type=Path, help="Resume complete P2.1 training state")
    p.add_argument("--init-p2", type=Path, help="Warm-start Extractor from full P2 checkpoint; fresh LoRA/optimizer")
    p.add_argument("--recipe", choices=["av", "joint"], default="joint")
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--balance-workload", action="store_true", help="Balance token-squared cost within the same accumulated global batch")
    p.add_argument("--accumulation", type=int, default=1)
    p.add_argument("--save-every", type=int, default=100)
    p.add_argument("--startup-save-step", type=int, default=0,
                   help="Optional early safety checkpoint; normal save interval remains unchanged")
    p.add_argument("--seed", type=int, default=45)
    p.add_argument("--tiny", action="store_true")
    p.add_argument("--allow-terminal-trim", action="store_true")
    p.add_argument("--checkpoint-every", type=int, default=1,
                   help="Renderer recomputation: 1 all blocks, 2 alternating, 0 none")
    p.add_argument("--decoder-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--fused-adam", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--cache-samples", type=int, default=2)
    p.add_argument("--data-root", type=Path, default=Path("/private/liqi_workspace/data/seamless/subset_300h_pure"))
    p.add_argument("--validation-cache", type=Path)
    p.add_argument("--validate-every", type=int, default=100)
    p.add_argument("--validation-seed", type=int, default=1045)
    p.add_argument("--tensorboard", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--nnodes", type=int, default=1)
    p.add_argument("--node-rank", type=int, default=0)
    p.add_argument("--master-addr")
    p.add_argument("--master-port", type=int, default=29521)
    p.add_argument("--run-id", default="p21")
    p.add_argument("--network-config", type=Path, help="JSON NCCL/GLOO environment profile")
    p.add_argument("--preflight-only", action="store_true")
    p.set_defaults(probe=None)  # Shared launch_environment; P2 probe does not exercise LoRA.
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = p.parse_args()
    if min(args.steps, args.accumulation, args.save_every) < 1 or min(args.checkpoint_every, args.cache_samples) < 0:
        p.error("Invalid step, accumulation, cache or checkpoint setting")
    if args.nnodes < 1 or not 0 <= args.node_rank < args.nnodes or args.validate_every < 1:
        p.error("Invalid topology or validation interval")
    if args.startup_save_step < 0:
        p.error("Startup checkpoint step must be nonnegative")
    if args.nnodes > 1 and not args.master_addr:
        p.error("Multi-node requires --master-addr")
    if not args.cache:
        p.error("Training requires --cache")
    if args.resume and args.init_p2:
        p.error("--resume and --init-p2 are mutually exclusive")
    if args.worker:
        from omni_av_p21.distributed import run
        run(args)
        return
    gpus = [int(g) for g in args.gpus.split(",")]
    if not gpus or len(set(gpus)) != len(gpus):
        p.error("GPU IDs must be nonempty and unique")
    from omni_av_p2.launch import launch_command, launch_environment, idle_gpus
    idle_gpus(gpus)
    for path in (args.cache, args.validation_cache, args.config, args.resume, args.init_p2):
        if path is not None and not path.is_file():
            raise FileNotFoundError(path)
    env = launch_environment(args, gpus)
    command = launch_command(args, len(gpus), Path(__file__).resolve(), sys.argv[1:])
    if args.preflight_only:
        print(json.dumps(dict(command=command, network={k: v for k, v in env.items()
              if k.startswith(("NCCL_", "GLOO_", "TORCH_NCCL_"))})))
        return
    raise SystemExit(subprocess.call(command, env=env))


if __name__ == "__main__":
    main()
