#!/usr/bin/env python3
"""Arm an existing trainer's cooperative STOP at an exact completed update."""
import argparse
import json
from pathlib import Path
import time


def last_row(path):
    if not path.exists():
        return None
    with path.open('rb') as stream:
        stream.seek(max(0, path.stat().st_size - 65536))
        lines = stream.read().splitlines()
    for line in reversed(lines):
        try:
            return json.loads(line)
        except (ValueError, UnicodeDecodeError):
            continue
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--step', type=int, required=True)
    p.add_argument('--record', type=Path, required=True)
    args = p.parse_args()
    if args.step < 1 or not (args.run / 'training').is_dir():
        p.error('Need a live run and positive target step')
    log = args.run / 'training/train_rank0.jsonl'
    while True:
        row = last_row(log)
        if row and row['step'] >= args.step:
            # The trainer writes this row BEFORE periodic checkpoint publication,
            # then checks STOP on the next iteration. Arm now, let save finish.
            stop = args.run / 'training/STOP'
            with stop.open('a') as stream:
                stream.write(f'User requested stop after step {args.step}\n')
            value = dict(target_step=args.step, observed_step=row['step'], stop_file=str(stop), time=time.time())
            args.record.parent.mkdir(parents=True, exist_ok=True)
            with args.record.open('x') as stream:
                json.dump(value, stream, indent=2)
            print(json.dumps(value), flush=True)
            break
        if (args.run / 'training/completed.json').exists() or list((args.run / 'training').glob('failed_rank*.json')):
            raise RuntimeError('Training ended before requested stop step')
        time.sleep(.05 if row and row['step'] >= args.step - 1 else 1)


if __name__ == '__main__':
    main()
