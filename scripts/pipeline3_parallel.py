#!/usr/bin/env python3
"""One node's torchrun launcher or P3 distributed worker."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--arm", choices=("frozen", "lora"), default="lora")
    group = p.add_mutually_exclusive_group()
    group.add_argument("--resume", type=Path)
    group.add_argument("--initialize-from", type=Path)
    p.add_argument("--total-steps", type=int, default=2000)
    p.add_argument("--accumulation", type=int, default=1)
    p.add_argument("--microbatch", type=int, default=1)
    p.add_argument("--save-every", type=int, default=100)
    p.add_argument("--validate-every", type=int, default=100)
    p.add_argument("--seed", type=int, default=20261008)
    p.add_argument("--cache-samples", type=int, default=4)
    p.add_argument("--balance-workload", action="store_true")
    p.add_argument("--prefetch", action="store_true")
    p.add_argument("--checkpointing", action="store_true")
    p.add_argument("--fused-adam", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--timeout", type=int, default=600)
    p.add_argument("--gpus", default="0")
    p.add_argument("--nnodes", type=int, default=1)
    p.add_argument("--node-rank", type=int, default=0)
    p.add_argument("--master-addr")
    p.add_argument("--master-port", type=int, default=29873)
    p.add_argument("--run-id", default="p3")
    p.add_argument("--network-config", type=Path)
    p.add_argument("--tiny", action="store_true", help="Synthetic real tiny Omni architecture, not pretrained")
    p.add_argument("--cpu", action="store_true", help="Gloo tiny-model checks only")
    p.add_argument("--preflight-only", action="store_true")
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    a = p.parse_args()
    if min(a.total_steps, a.accumulation, a.microbatch, a.save_every, a.validate_every, a.threads, a.timeout) < 1 or a.cache_samples < 0:
        p.error("Invalid steps/cache settings")
    if (a.cpu and not a.tiny) or (not a.tiny and not a.cache):
        p.error("CPU requires --tiny; pretrained run requires --cache")
    if a.nnodes < 1 or not 0 <= a.node_rank < a.nnodes or (a.nnodes > 1 and not a.master_addr):
        p.error("Invalid node topology")
    if a.worker:
        from omni_av_p3.distributed import run
        run(a)
        return
    from omni_av_p3.launch import environment, preflight
    from omni_av_p2.launch import launch_command
    result = preflight(a)
    command = launch_command(a, len(result["gpus"]), Path(__file__).resolve(), sys.argv[1:])
    env = environment(a, result["gpus"])
    if a.preflight_only:
        print(json.dumps(dict(**result, command=command, network={k: v for k, v in env.items()
            if k.startswith(("NCCL_", "GLOO_", "TORCH_NCCL_"))})))
        return
    raise SystemExit(subprocess.call(command, env=env))


if __name__ == "__main__":
    main()
