#!/usr/bin/env python3
"""Prepare Dev pairs on CPU; explicitly launch generation only on idle GPUs."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
DATA = Path('/private/liqi_workspace/data/seamless/subset_300h_pure')


def launch(args):
    cluster = json.loads(args.cluster.read_text())
    nodes, python = cluster['nodes'], cluster['python']
    if len({n['host'] for n in nodes}) != len(nodes):
        raise ValueError('Duplicate cluster nodes')
    key = cluster.get('ssh_key', f'/private/ssh_keys/root_{socket.gethostname()}_ed25519')
    def transport(node, command):
        if node.get('local'):
            return command
        return ['ssh', '-i', key, '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
                '-o', 'StrictHostKeyChecking=yes', f"root@{node['host']}", shlex.join(command)]
    def preflight(node):
        if len(set(node['gpus'])) != len(node['gpus']):
            raise ValueError('Duplicate GPU IDs')
        p = subprocess.run(transport(node, ['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu',
            '--format=csv,noheader,nounits']), capture_output=True, text=True, check=True, timeout=20)
        rows = {int(x[0]): (int(x[1]), int(x[2])) for line in p.stdout.splitlines() if (x := line.split(','))}
        if any(g not in rows or rows[g][0] > 1000 or rows[g][1] > 5 for g in node['gpus']):
            raise RuntimeError(f"Validation requires idle GPUs; will not interrupt training: {node['name']}: {rows}")
    with ThreadPoolExecutor(max_workers=len(nodes)) as pool:
        list(pool.map(preflight, nodes))
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    sys.path.insert(0, str(ROOT / 'src'))
    from omni_av_p2.config import Config
    from omni_av_p2.paired_validation import checkpoint_state, validate_suite, write_comparison_report
    suite = json.loads(args.suite.read_text())
    validate_suite(suite, Config(**suite['config']))
    state, step = checkpoint_state(args.checkpoint, suite)
    del state
    slots = [(node, gpu) for node in nodes for gpu in node['gpus']]
    if not slots or len(slots) > len(suite['cases']):
        raise ValueError('Need one or more cases per worker')
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'logs').mkdir()
    processes, logs = [], []
    try:
        for i, (node, gpu) in enumerate(slots):
            command = [python, str(Path(__file__).resolve()), 'run', '--suite', str(args.suite.resolve()),
                '--checkpoint', str(args.checkpoint.resolve()), '--output', str(args.output.resolve()),
                '--gpu', str(gpu), '--shards', str(len(slots)), '--shard-index', str(i)]
            command = [python, str(ROOT / 'scripts/pipeline2_cluster.py'), '--guard', *command]
            stream = (args.output / 'logs' / f'worker_{i:03d}.log').open('x')
            logs.append(stream)
            processes.append(subprocess.Popen(transport(node, command), stdin=subprocess.PIPE,
                stdout=stream, stderr=subprocess.STDOUT))
        while any(p.poll() is None for p in processes):
            time.sleep(1)
        codes = [p.returncode for p in processes]
        summaries = []
        for i in range(len(slots)):
            p = args.output / f'worker_{i:03d}.completed.json'
            if p.exists():
                summaries.append(json.loads(p.read_text()))
        cases = [case for summary in summaries for case in summary['cases']]
        passed = (all(code == 0 for code in codes) and len(cases) == len(suite['cases'])
                  and len({c['sample_id'] for c in cases}) == len(cases)
                  and all(c['status'] == 'complete' for c in cases))
        from omni_av_p2.checkpoint import atomic_json_new
        atomic_json_new(args.output / 'completed.json', dict(passed=passed, step=step, returncodes=codes,
            cases=cases, expected_cases=len(suite['cases']), modes=['gt', 'omni'], quality_pass=None))
        write_comparison_report(args.output, suite)
        return passed
    finally:
        for p in processes:
            p.stdin.close()
        for p in processes:
            try:
                p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                p.terminate()
        for stream in logs:
            stream.close()


def main():
    if sys.argv[1:2] == ['metrics']:
        from pipeline2_metrics import main as metrics_main
        metrics_main(sys.argv[2:])
        return
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    prepare = sub.add_parser('prepare', help='CPU-only Dev48 task preparation; no inference')
    prepare.add_argument('--manifest', type=Path, default=ROOT / 'outputs/p2_full91009_dev48_20261008/dev.json')
    prepare.add_argument('--metadata', type=Path, default=DATA / 'ltx23_precomputed_720p_area_qualified_20260920/metadata_dev.jsonl')
    prepare.add_argument('--config', type=Path)
    prepare.add_argument('--seed', type=int, default=1045)
    prepare.add_argument('--sampling-steps', type=int, default=30)
    prepare.add_argument('--max-plan-tokens', type=int, default=512)
    prepare.add_argument('--output', type=Path, required=True)
    for name in ('run', 'launch'):
        command = sub.add_parser(name)
        command.add_argument('--suite', type=Path, required=True)
        command.add_argument('--checkpoint', type=Path, required=True)
        command.add_argument('--output', type=Path, required=True)
        if name == 'run':
            command.add_argument('--gpu', type=int, required=True)
            command.add_argument('--shards', type=int, default=1)
            command.add_argument('--shard-index', type=int, default=0)
        else:
            command.add_argument('--cluster', type=Path, default=ROOT / 'configs/p2_three_nodes.json')
    args = p.parse_args()
    if args.command == 'launch':
        raise SystemExit(0 if launch(args) else 1)
    if args.command == 'prepare':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    else:
        rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu',
                                       '--format=csv,noheader,nounits'], text=True)
        values = {int(x[0]): (int(x[1]), int(x[2])) for line in rows.splitlines() if (x := line.split(','))}
        if args.gpu not in values or values[args.gpu][0] > 1000 or values[args.gpu][1] > 5:
            raise RuntimeError('Selected GPU is occupied; validation will not interrupt training')
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    sys.path.insert(0, str(ROOT / 'src'))
    import torch
    torch.set_num_threads(4)
    from omni_av_p2.config import Config
    from omni_av_p2.paired_validation import prepare_suite, run_shard
    if args.command == 'prepare':
        cfg = Config(**json.loads(args.config.read_text())) if args.config else Config()
        suite = prepare_suite(args.manifest, args.metadata, args.output, cfg, args.seed,
                              args.sampling_steps, args.max_plan_tokens)
        print(json.dumps(dict(cases=len(suite['cases']), generations=2 * len(suite['cases']),
                              suite=str(args.output / 'suite.json'), generated=False)))
    else:
        result = run_shard(args.suite, args.checkpoint, args.output, args.shard_index, args.shards)
        raise SystemExit(0 if result['passed'] else 1)


if __name__ == '__main__':
    main()
