#!/usr/bin/env python3
"""Local v4 CLI. GPU operations require an explicit idle physical device."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["test", "omni-gate", "prepare", "train-planner", "ltx-gate",
                                             "train-renderer", "train-joint", "infer-plan"])
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--planner-checkpoint", type=Path)
    parser.add_argument("--renderer-checkpoint", type=Path)
    parser.add_argument("--observation", type=Path, help="Causal Observation JSON only, no target fields")
    parser.add_argument("--manifest", type=Path, help="Explicit training examples JSON, see DATA.md")
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--budget", type=float, default=8)
    parser.add_argument("--tiny", action="store_true", help="Explicit AV latent crop for engineering only")
    parser.add_argument("--data-root", type=Path, default=Path("/private/liqi_workspace/data/seamless/subset_300h_pure"))
    args = parser.parse_args()
    if args.command == "test":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        import torch
        torch.set_num_threads(4)
        import pytest
        raise SystemExit(pytest.main([str(ROOT / "tests"), "-q"]))
    if args.gpu is None or args.output is None:
        parser.error("GPU commands require --gpu and a new --output directory")
    output = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits"], text=True)
    rows = {int(v[0]): (int(v[1]), int(v[2])) for line in output.splitlines() if (v := line.split(","))}
    if args.gpu not in rows or rows[args.gpu][0] > 1000 or rows[args.gpu][1] > 5:
        raise RuntimeError(f"Requested GPU is not idle: {args.gpu}: {rows.get(args.gpu)}")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args.output.mkdir(parents=True, exist_ok=False)
    from omni_av.runner import run
    run(args)


if __name__ == "__main__":
    main()
