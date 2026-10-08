#!/usr/bin/env python3
"""Prepare immutable P2 shards on explicitly reserved GPUs; publish only full coverage."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import queue
import shlex
import socket
import subprocess
import sys
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]


def snapshot(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def worker(args):
    # The controller checks ALL GPUs before reserving its multiple workers per GPU.
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['OMP_NUM_THREADS'] = '4'
    sys.path.insert(0, str(ROOT / 'src'))
    import fcntl
    from types import SimpleNamespace
    from omni_av_p2.config import Config
    from omni_av_p2.cache_prepare import prepare
    # Multiple queues may help on still-unclaimed shards. Exactly one writer per shard.
    with args.output.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        index_path = args.output / 'cache_index.json'
        if index_path.exists():
            from omni_av_p2.data import contract
            index = json.loads(index_path.read_text())
            examples = json.loads(args.manifest.read_text())
            if (index['contract'] != contract(Config()) or index['split'] != 'train'
                    or [e['sample_id'] for e in index['entries']]
                    != [e['observation']['sample_id'] for e in examples]
                    or any(not Path(e[k]).is_file() for e in index['entries'] for k in ('features','text'))):
                raise ValueError('Existing shard does not match requested manifest/contract')
            print(json.dumps(dict(event='reused_complete_shard', samples=len(examples))), flush=True)
            return
        args.output.mkdir(parents=True, exist_ok=True)
        prepare(SimpleNamespace(output=args.output, manifest=args.manifest, gpu=args.gpu,
                                data_root=Path('/private/liqi_workspace/data/seamless/subset_300h_pure')),
                Config(), batch_size=4)


def controller(args):
    config = json.loads(args.cluster.read_text())
    nodes, python = config['nodes'], config['python']
    key = config.get('ssh_key', f'/private/ssh_keys/root_{socket.gethostname()}_ed25519')

    def transport(node, command):
        if node.get('local'):
            return command
        return ['ssh', '-i', key, '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
                '-o', 'StrictHostKeyChecking=yes', f"root@{node['host']}", shlex.join(command)]

    if len({n['host'] for n in nodes}) != len(nodes):
        raise ValueError('Duplicate nodes')
    for node in nodes:
        if len(node['gpus']) != len(set(node['gpus'])):
            raise ValueError('Duplicate GPU IDs')
        result = subprocess.run(transport(node, ['nvidia-smi',
            '--query-gpu=index,memory.used,utilization.gpu', '--format=csv,noheader,nounits']),
            capture_output=True, text=True, check=True, timeout=30)
        rows = {int(v[0]): (int(v[1]), int(v[2])) for line in result.stdout.splitlines()
                if (v := line.split(','))}
        if any(g not in rows or rows[g][0] > 1000 or rows[g][1] > 5 for g in node['gpus']):
            raise RuntimeError(f"GPUs not idle on {node['name']}: {rows}")
    examples = json.loads(args.manifest.read_text())
    ids = [e['observation']['sample_id'] for e in examples]
    if len(examples) != args.expected or len(set(ids)) != len(ids) or any(e['split'] != 'train' for e in examples):
        raise ValueError('Full Train manifest count/identity mismatch')
    manifest_hash = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    if args.resume:
        previous = json.loads((args.output/'run.json').read_text())
        if previous['manifest_sha256'] != manifest_hash or previous['shards'] != args.shards:
            raise ValueError('Resume manifest or shard count changed')
        if (args.output/'STOP').exists():
            raise ValueError('STOP exists; explicitly clear it before restarting this run')
    args.output.mkdir(parents=True, exist_ok=args.resume)
    for name in ('manifests', 'train', 'logs'):
        (args.output / name).mkdir(exist_ok=args.resume)
    generation = str(time.time_ns())
    tasks = []
    jobs = queue.Queue()
    for number in range(args.shards):
        selected = examples[number::args.shards]
        if not selected:
            raise ValueError('Empty shard')
        manifest = args.output / 'manifests' / f'shard{number:03d}.json'
        snapshot(manifest, selected)
        tasks.append(dict(shard=number, samples=len(selected), status='pending',
                          features=0, text=0, manifest=str(manifest)))
        jobs.put(number)
    provenance = dict(manifest=str(args.manifest), samples=len(examples),
        manifest_sha256=manifest_hash,
        base_commit=subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip(),
        nodes=nodes, workers_per_gpu=args.workers_per_gpu, shards=args.shards,
        started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        output=str(args.output), train_cache=str(args.output / 'train/cache_index.json'),
        training_started=False, preparation_only=True, text_batch_size=4,
        max_gemma_processes_per_gpu=2, generation=generation)
    if (args.output/'run.json').exists():
        (args.output/'run.json').rename(args.output/f'run_before_{generation}.json')
    snapshot(args.output / 'run.json', provenance)
    stop = threading.Event()
    launch_lock = threading.Lock()

    def slot(node, gpu, lane):
        while not stop.is_set():
            try:
                number = jobs.get_nowait()
            except queue.Empty:
                return
            task = tasks[number]
            destination = args.output / 'train' / f'shard{number:03d}'
            log_path = args.output / 'logs' / f'shard{number:03d}_{generation}.log'
            task.update(status='running', host=node['host'], node=node['name'], gpu=gpu,
                        lane=lane, output=str(destination), log=str(log_path), started=time.time())
            command = [python, str(Path(__file__).resolve()), '--worker', '--gpu', str(gpu),
                       '--manifest', task['manifest'], '--output', str(destination)]
            guarded = [python, str(ROOT / 'scripts/pipeline2_cluster.py'), '--guard', *command]
            try:
                with log_path.open('x') as log:
                    # Stagger SSH handshakes/model loads; the GPU reservation is controller-owned.
                    with launch_lock:
                        process = subprocess.Popen(transport(node, guarded), stdin=subprocess.PIPE,
                                                   stdout=log, stderr=subprocess.STDOUT)
                        time.sleep(.3)
                    task['agent_pid'] = process.pid
                    while process.poll() is None:
                        if stop.wait(1):
                            process.stdin.close()
                            break
                    try:
                        code = process.wait(timeout=45)
                    except subprocess.TimeoutExpired:
                        process.terminate()
                        code = process.wait(timeout=15)
                    if not process.stdin.closed:
                        process.stdin.close()
                task.update(status='complete' if code == 0 else 'failed', returncode=code,
                            ended=time.time())
                if code == 0:
                    index = json.loads((destination / 'cache_index.json').read_text())
                    if len(index['entries']) != task['samples']:
                        raise ValueError('Shard completion count mismatch')
                    task['features'] = task['text'] = task['samples']
            except Exception:
                task.update(status='failed', error=traceback.format_exc())
            finally:
                jobs.task_done()

    positions = {}
    def progress():
        for task in tasks:
            if task['status'] != 'running' or not Path(task['log']).exists():
                continue
            path = task['log']
            with open(path) as stream:
                stream.seek(positions.get(path, 0))
                for line in stream:
                    if line.startswith('{'):
                        try:
                            event = json.loads(line).get('event')
                        except json.JSONDecodeError:
                            continue
                        if event in ('p2_features', 'p2_text', 'reused_features', 'reused_text'):
                            task['features' if event in ('p2_features','reused_features') else 'text'] += 1
                positions[path] = stream.tell()
        summary = dict(updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            total_samples=len(examples), features=sum(t['features'] for t in tasks),
            text=sum(t['text'] for t in tasks), completed_shards=sum(t['status']=='complete' for t in tasks),
            running_shards=sum(t['status']=='running' for t in tasks),
            failed_shards=sum(t['status']=='failed' for t in tasks), tasks=tasks)
        snapshot(args.output / 'progress.json', summary)
        snapshot(args.output / 'overall_progress.json', dict(time=time.time(), total_samples=len(examples),
            feature_files=summary['features'], text_files=summary['text'],
            completed_shards=summary['completed_shards'],running_shards=summary['running_shards'],
            failed_shards=summary['failed_shards'],global_index_published=(args.output/'train/cache_index.json').exists(),
            training_started=False, note='Counts include only read-verified reused files and new complete writes in this generation'))
        print(json.dumps({k:v for k,v in summary.items() if k != 'tasks'}), flush=True)

    def telemetry(node):
        try:
            result = subprocess.run(transport(node, ['nvidia-smi',
                '--query-gpu=index,memory.used,utilization.gpu,power.draw,temperature.gpu',
                '--format=csv,noheader,nounits']), capture_output=True, text=True, timeout=15)
            with (args.output / f"gpu_{node['name']}.jsonl").open('a') as log:
                log.write(json.dumps(dict(time=time.time(), host=node['host'],
                                          csv=result.stdout, returncode=result.returncode)) + '\n')
        except Exception as exc:
            print(f"Telemetry {node['name']}: {exc}", flush=True)

    slots = [(n, g, lane) for n in nodes for g in n['gpus'] for lane in range(args.workers_per_gpu)]
    with ThreadPoolExecutor(max_workers=len(slots)) as pool, ThreadPoolExecutor(max_workers=len(nodes)) as monitors:
        futures = [pool.submit(slot, *item) for item in slots]
        try:
            while not all(f.done() for f in futures):
                if (args.output / 'STOP').exists():
                    stop.set()
                progress()
                list(monitors.map(telemetry, nodes))
                time.sleep(20)
            for future in futures:
                future.result()
        finally:
            stop.set()
    progress()
    if any(t['status'] != 'complete' for t in tasks):
        snapshot(args.output / 'failed.json', dict(reason='Not all shards completed; global index not published'))
        raise RuntimeError('Preparation incomplete; inspect progress.json and shard logs')
    sys.path.insert(0, str(ROOT / 'scripts'))
    from pipeline2_merge_cache import merge
    index = merge([Path(t['output']) / 'cache_index.json' for t in tasks], args.manifest)
    # Publish only after exact ID coverage, split/contract and file existence checks.
    from omni_av_p2.checkpoint import atomic_json_new
    atomic_json_new(args.output / 'train/cache_index.json', index)
    snapshot(args.output / 'completed.json', dict(samples=len(index['entries']), shards=len(tasks),
        exact_coverage=True, index=str(args.output / 'train/cache_index.json'),
        training_started=False, payload_readback='not exhaustive; native extraction validation and merge checks'))
    progress()  # Refresh publication status after merge, not only before it.


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cluster', type=Path, default=ROOT/'configs/p2_three_nodes.json')
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--shards', type=int, default=192)
    p.add_argument('--workers-per-gpu', type=int, default=2)
    p.add_argument('--expected', type=int, default=91009)
    p.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--gpu', type=int)
    args = p.parse_args()
    if min(args.shards, args.workers_per_gpu, args.expected) < 1:
        p.error('Counts must be positive')
    if args.worker:
        if args.gpu is None:
            p.error('Worker requires explicit GPU')
        worker(args)
    else:
        controller(args)


if __name__ == '__main__':
    main()
