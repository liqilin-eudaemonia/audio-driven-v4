#!/usr/bin/env python3
"""Pipeline 3 controlled planning experiment, independent of P1/P2 outputs."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from omni_av_p3.model import Config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    train = sub.add_parser("train")
    for command in (prep, train):
        command.add_argument("--config", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--device", default="cuda:0")
    prep.add_argument("--train-manifest", type=Path, required=True)
    prep.add_argument("--dev-manifest", type=Path, required=True)
    train.add_argument("--cache", type=Path, required=True)
    train.add_argument("--arm", choices=("frozen", "lora"), required=True)
    comparison = sub.add_parser("compare")
    comparison.add_argument("--frozen", type=Path, required=True)
    comparison.add_argument("--lora", type=Path, required=True)
    comparison.add_argument("--output", type=Path, required=True)
    diagnosis = sub.add_parser("diagnose")
    diagnosis.add_argument("--run", type=Path, required=True)
    diagnosis.add_argument("--cache", type=Path, required=True)
    diagnosis.add_argument("--output", type=Path, required=True)
    diagnosis.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.command == "compare":
        from omni_av_p3.experiment import compare
        compare(args.frozen, args.lora, args.output)
    elif args.command == "diagnose":
        from omni_av_p3.diagnostics import diagnose
        diagnose(args.run, args.cache, args.output, args.device)
    else:
        cfg = Config(**json.loads(args.config.read_text()))
        if args.command == "prepare":
            from omni_av_p3.data import prepare
            prepare(cfg, args.train_manifest, args.dev_manifest, args.output, args.device)
        else:
            from omni_av_p3.experiment import train_arm
            train_arm(cfg, args.cache, args.output, args.arm, args.device)


if __name__ == "__main__":
    main()
