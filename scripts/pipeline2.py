#!/usr/bin/env python3
"""Independent Pipeline 2 entry point; all outputs must be new paths."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["test", "prepare", "gate", "train", "infer", "export-condition", "export-checkpoint"])
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--observation", type=Path)
    parser.add_argument("--text-plan", type=Path, help="Optional user-specified TextPlan JSON; otherwise native Omni reply")
    parser.add_argument("--text-only", action="store_true", help="E0 baseline: native Gemma text + first frame, no U/checkpoint")
    parser.add_argument("--zero-condition", action="store_true", help="Diagnostic: preserve U token count but set values to zero")
    parser.add_argument("--recipe", choices=["av", "joint"], default="joint")
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--budget", type=float, default=8)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--sampling-steps", type=int, default=5)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=45)
    parser.add_argument("--tiny", action="store_true", help="Engineering-only VAE crop")
    parser.add_argument("--engineering-small", action="store_true", help="Small heads, unchanged pretrained Omni/Gemma/LTX")
    parser.add_argument("--allow-terminal-trim", action="store_true", help="Acknowledge recorded inherited VAE terminal trim for this run")
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--fps", type=float, default=30)
    parser.add_argument("--data-root", type=Path, default=Path("/private/liqi_workspace/data/seamless/subset_300h_pure"))
    args = parser.parse_args()
    if args.shards < 1 or not 0 <= args.shard_index < args.shards:
        parser.error("Invalid preparation shard")
    if args.command == "export-checkpoint":
        if args.checkpoint is None or args.output is None:
            parser.error("export-checkpoint needs --checkpoint and new --output directory")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        from omni_av_p2.checkpoint import export_connector
        args.output.mkdir(parents=True, exist_ok=False)
        export_connector(args.checkpoint, args.output / "connector.pt")
        print(str(args.output / "connector.pt"))
        return
    if args.command == "test":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        import torch
        import pytest
        torch.set_num_threads(4)
        raise SystemExit(pytest.main([str(ROOT / "tests"), "-q"]))
    if args.gpu is None or args.output is None:
        parser.error("--gpu and a new --output are required")
    result = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits"], text=True)
    rows = {int(parts[0]): (int(parts[1]), int(parts[2])) for line in result.splitlines() if (parts := line.split(","))}
    if args.gpu not in rows or rows[args.gpu][0] > 1000 or rows[args.gpu][1] > 5:
        raise RuntimeError(f"GPU is not idle: {args.gpu}: {rows.get(args.gpu)}")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args.output.mkdir(parents=True, exist_ok=False)
    from omni_av_p2.runner import run
    run(args)


if __name__ == "__main__":
    main()
