#!/usr/bin/env python3
"""P3 paired speed benchmark and controlled convergence experiment."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from omni_av_p3.extended import benchmark, run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("benchmark", "train"))
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--arm", choices=("frozen", "lora"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--benchmark-steps", type=int, default=30)
    parser.add_argument("--total-steps", type=int, default=2000)
    parser.add_argument("--validate-every", type=int, default=500)
    parser.add_argument("--train-limit", type=int)
    args = parser.parse_args()
    if args.validate_every < 1 or args.benchmark_steps < 1:
        parser.error("Step intervals must be positive")
    (benchmark if args.command == "benchmark" else run)(args)


if __name__ == "__main__":
    main()
