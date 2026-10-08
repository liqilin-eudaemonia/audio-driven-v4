#!/usr/bin/env python3
"""Refresh only Omni conditions on the three reserved nodes, with verified retirement."""
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
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def worker(args):
    os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu), OMP_NUM_THREADS='4',
                      TOKENIZERS_PARALLELISM='false')
    sys.path.insert(0, str(ROOT / 'src'))
    import fcntl
    from omni_av_p2.cache_refresh import refresh_job
    from omni_av_p2.config import Config
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.with_suffix('.lock').open('a') as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        refresh_job(json.loads(args.job.read_text()), args.output, args.source, Config(), args.delete_old)


def controller(args):
    config = json.loads(args.cluster.read_text())
    nodes, python = config['nodes'], config['python']
    key = config.get('ssh_key', f'/private/ssh_keys/root_{socket.gethostname()}_ed25519')
    def transport(node, cmd):
        if node.get('local'):
            return cmd
        return ['ssh', '-i', key, '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
                '-o', 'StrictHostKeyChecking=yes', f"root@{node['host']}", shlex.join(cmd)]
    if args.source.resolve() == args.output.resolve():
        raise ValueError('Use a distinct output directory')
    if len({n['host'] for n in nodes}) != len(nodes):
        raise ValueError('Duplicate nodes')
    for node in nodes:
        r = subprocess.run(transport(node, ['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu',
            '--format=csv,noheader,nounits']), capture_output=True, text=True, check=True, timeout=30)
        rows = {int(v[0]): (int(v[1]), int(v[2])) for line in r.stdout.splitlines() if (v := line.split(','))}
        if len(set(node['gpus'])) != len(node['gpus']) or any(
                g not in rows or rows[g][0] > 1000 or rows[g][1] > 5 for g in node['gpus']):
            raise RuntimeError(f"GPUs not idle on {node['name']}: {rows}")
    sys.path.insert(0, str(ROOT / 'src'))
    from omni_av_p2.config import Config
    from omni_av_p2.data import contract
    from omni_av_p2.cache_refresh import legacy_contract
    expected = contract(Config())
    manifest_paths = dict(train=args.train_manifest, dev=args.dev_manifest)
    manifests = {s: json.loads(p.read_text()) for s, p in manifest_paths.items()}
    source_indexes = {}
    for split, count in [('train', 91009), ('dev', 48)]:
        p = args.source / split / 'cache_index.json'
        if not p.exists():
            p = args.source / split / 'cache_index.superseded_no_time.json'
        index = json.loads(p.read_text())
        wanted = [e['observation']['sample_id'] for e in manifests[split]]
        if (len(wanted) != count or len(set(wanted)) != count
                or any(e['split'] != split for e in manifests[split])
                or index['contract'] != legacy_contract(expected) or index['split'] != split
                or [e['sample_id'] for e in index['entries']] != wanted):
            raise ValueError(f'{split} source/manifest mismatch')
        source_indexes[split] = index
    if set(e['sample_id'] for e in source_indexes['train']['entries']) & set(
            e['sample_id'] for e in source_indexes['dev']['entries']):
        raise ValueError('Train/dev overlap')
    identity = dict(source=str(args.source.resolve()), contract=expected, delete_old=args.delete_old,
        manifests={s: hashlib.sha256(p.read_bytes()).hexdigest() for s, p in manifest_paths.items()},
        train_shards=args.shards, dev_shards=8)
    args.output.mkdir(parents=True, exist_ok=args.resume)
    if args.resume:
        if json.loads((args.output / 'identity.json').read_text()) != identity:
            raise ValueError('Refresh identity changed')
        if (args.output / 'STOP').exists():
            raise ValueError('Explicitly resolve STOP before resuming')
    else:
        snapshot(args.output / 'identity.json', identity)
    for name in ('jobs', 'logs', 'train', 'dev'):
        (args.output / name).mkdir(exist_ok=True)
    generation = str(time.time_ns())
    if args.resume and (args.output / 'failed.json').exists():
        (args.output / 'failed.json').rename(args.output / f'failed_before_{generation}.json')
    snapshot(args.output / f'run_{generation}.json', dict(identity=identity, nodes=nodes,
        workers_per_gpu=args.workers_per_gpu, started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())))
    snapshot(args.output / f'source_sha256_{generation}.json', {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for folder in ('src', 'scripts') for p in (ROOT / folder).rglob('*.py')})
    # Disable discovery of the old global indexes before any old payload is retired.
    if args.delete_old:
        for split in ('train', 'dev'):
            p = args.source / split / 'cache_index.json'
            if p.exists():
                retired = p.with_name('cache_index.superseded_no_time.json')
                if retired.exists():
                    raise ValueError('Source already has a retired index')
                p.rename(retired)
        snapshot(args.source / 'SUPERSEDED.json', dict(replacement=str(args.output),
            reason='User authorized removal of time-bearing Omni conditions after verified replacement',
            status='refreshing; use replacement only after completed.json'))
    tasks, pending = [], queue.Queue()
    for split, count in [('dev', 8), ('train', args.shards)]:
        for shard in range(count):
            items = [dict(entry=e, example=x) for e, x in zip(
                source_indexes[split]['entries'][shard::count], manifests[split][shard::count])]
            if not items:
                raise ValueError('Empty refresh shard')
            job = args.output / 'jobs' / f'{split}_{shard:03d}.json'
            snapshot(job, dict(split=split, source_contract=source_indexes[split]['contract'], items=items))
            task = dict(split=split, shard=shard, samples=len(items), status='pending', done=0,
                        job=str(job), output=str(args.output / split / f'shard{shard:03d}'))
            tasks.append(task)
            pending.put(task)
    stop, launch_lock = threading.Event(), threading.Lock()
    def slot(node, gpu, lane):
        while not stop.is_set():
            try:
                task = pending.get_nowait()
            except queue.Empty:
                return
            log = args.output / 'logs' / f"{task['split']}_{task['shard']:03d}_{generation}.log"
            task.update(status='running', host=node['host'], gpu=gpu, lane=lane, log=str(log))
            cmd = [python, str(Path(__file__).resolve()), '--worker', '--gpu', str(gpu),
                   '--job', task['job'], '--source', str(args.source), '--output', task['output']]
            if args.delete_old:
                cmd.append('--delete-old')
            guarded = [python, str(ROOT / 'scripts/pipeline2_cluster.py'), '--guard', *cmd]
            process = None
            try:
                with log.open('x') as stream:
                    with launch_lock:
                        process = subprocess.Popen(transport(node, guarded), stdin=subprocess.PIPE,
                                                   stdout=stream, stderr=subprocess.STDOUT)
                        time.sleep(.3)
                    task['agent_pid'] = process.pid
                    while process.poll() is None:
                        if stop.wait(1):
                            process.stdin.close()
                            break
                    code = process.wait(timeout=45)
                    if code:
                        raise RuntimeError(f'Worker exited {code}')
                    index = json.loads((Path(task['output']) / 'cache_index.json').read_text())
                    if len(index['entries']) != task['samples'] or index['contract'] != expected:
                        raise ValueError('Completed shard mismatch')
                    task.update(status='complete', done=task['samples'])
            except Exception:
                task.update(status='failed', error=traceback.format_exc())
                stop.set()
            finally:
                if process is not None:
                    if not process.stdin.closed:
                        process.stdin.close()
                    try:
                        process.wait(timeout=45)
                    except subprocess.TimeoutExpired:
                        process.terminate()
                pending.task_done()
    positions = {}
    def progress():
        for task in tasks:
            if task['status'] != 'running' or not Path(task['log']).exists():
                continue
            with open(task['log']) as stream:
                stream.seek(positions.get(task['log'], 0))
                for line in stream:
                    if line.startswith('{'):
                        try:
                            event = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if event.get('event') in ('refreshed_condition', 'reused_refreshed_condition'):
                            task['done'] = max(task['done'], event['index'] + 1)
                positions[task['log']] = stream.tell()
        summary = dict(time=time.time(), total=91057, refreshed=sum(t['done'] for t in tasks),
            completed_shards=sum(t['status'] == 'complete' for t in tasks),
            running_shards=sum(t['status'] == 'running' for t in tasks),
            failed_shards=sum(t['status'] == 'failed' for t in tasks), training_started=False)
        snapshot(args.output / 'progress.json', dict(summary, tasks=tasks))
        print(json.dumps(summary), flush=True)
    def telemetry(node):
        try:
            p = subprocess.run(transport(node, ['nvidia-smi',
                '--query-gpu=index,memory.used,utilization.gpu,power.draw,temperature.gpu',
                '--format=csv,noheader,nounits']), capture_output=True, text=True, timeout=15)
            with (args.output / f"gpu_{node['name']}.jsonl").open('a') as f:
                f.write(json.dumps(dict(time=time.time(), code=p.returncode, csv=p.stdout)) + '\n')
        except Exception as exc:
            print(f'Telemetry: {exc}', flush=True)
    slots = [(n, g, lane) for lane in range(args.workers_per_gpu) for n in nodes for g in n['gpus']]
    with ThreadPoolExecutor(max_workers=len(slots)) as pool, ThreadPoolExecutor(max_workers=3) as monitor:
        futures = [pool.submit(slot, *s) for s in slots]
        try:
            while not all(f.done() for f in futures):
                if (args.output / 'STOP').exists():
                    stop.set()
                progress()
                list(monitor.map(telemetry, nodes))
                time.sleep(20)
            for f in futures:
                f.result()
        finally:
            stop.set()
    progress()
    if any(t['status'] != 'complete' for t in tasks):
        snapshot(args.output / 'failed.json', dict(reason='Incomplete refresh; sources deleted only after verified replacement'))
        raise RuntimeError('Refresh incomplete')
    sys.path.insert(0, str(ROOT / 'scripts'))
    from pipeline2_merge_cache import merge
    from omni_av_p2.checkpoint import atomic_json_new
    for split in ('train', 'dev'):
        index = merge([Path(t['output']) / 'cache_index.json' for t in tasks if t['split'] == split],
                      manifest_paths[split])
        path = args.output / split / 'cache_index.json'
        if path.exists():
            if json.loads(path.read_text()) != index:
                raise ValueError('Existing global refresh index mismatch')
        else:
            atomic_json_new(path, index)
    remaining = [e['features'] for idx in source_indexes.values() for e in idx['entries'] if Path(e['features']).exists()]
    if args.delete_old and remaining:
        raise ValueError(f'{len(remaining)} old feature files remain')
    snapshot(args.output / 'completed.json', dict(train=91009, dev=48, exact_coverage=True,
        old_feature_files_remaining=len(remaining), old_deleted=args.delete_old,
        all_new_payloads_readback_verified=True, all_teacher_tensors_equal_verified=True,
        gemma_text_reused_by_hardlink=True, vae_untouched=True, training_started=False,
        completed_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())))
    if args.delete_old:
        snapshot(args.source / 'SUPERSEDED.json', dict(replacement=str(args.output),
            status='complete', old_feature_files_deleted=91057, unaffected_text_preserved=True))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--cluster', type=Path, default=ROOT / 'configs/p2_three_nodes.json')
    p.add_argument('--train-manifest', type=Path, default=ROOT / 'outputs/p2_full91009_dev48_20261008/train.json')
    p.add_argument('--dev-manifest', type=Path, default=ROOT / 'outputs/p2_full91009_dev48_20261008/dev.json')
    p.add_argument('--shards', type=int, default=192)
    p.add_argument('--workers-per-gpu', type=int, default=6)
    p.add_argument('--delete-old', action='store_true')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--worker', action='store_true')
    p.add_argument('--gpu', type=int)
    p.add_argument('--job', type=Path)
    a = p.parse_args()
    if a.workers_per_gpu < 1 or not 1 <= a.shards <= 91009:
        p.error('Invalid worker/shard count')
    if a.worker:
        if a.job is None or a.gpu is None:
            p.error('Worker requires job and GPU')
        worker(a)
    else:
        controller(a)


if __name__ == '__main__':
    main()
