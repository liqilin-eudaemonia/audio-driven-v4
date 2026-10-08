#!/usr/bin/env python3
"""Add two owned workers per GPU to an existing cache run; shard locks avoid duplicate work."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import queue
import shlex
import subprocess
import threading
import time

from pipeline2_cache_cluster import ROOT, snapshot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--first-shard', type=int, default=96)
    args = parser.parse_args()
    root = args.output
    config = json.loads((ROOT/'configs/p2_three_nodes.json').read_text())
    python = config['python']
    run = json.loads((root/'run.json').read_text())
    nodes = run['nodes']
    key = '/private/ssh_keys/root_dedicated-developjob-js-public-huvlx_ed25519'

    def transport(node, command):
        if node.get('local'):
            return command
        return ['ssh','-i',key,'-o','BatchMode=yes','-o','ConnectTimeout=15',
                '-o','StrictHostKeyChecking=yes',f"root@{node['host']}",shlex.join(command)]

    # Never share a GPU with an unrelated job. This expansion is scoped to this run only.
    check = '''import subprocess
from pathlib import Path
pids = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).split()
assert len(set(pids)) == 16, pids
for pid in set(pids):
 cmd=Path('/proc/'+pid+'/cmdline').read_bytes().replace(b'\\0',b' ')
 assert b'pipeline2_cache_cluster.py' in cmd and b'--worker' in cmd and OUTPUT.encode() in cmd, cmd
print('All existing GPU processes belong to this cache run')
'''.replace('OUTPUT',repr(str(root)))
    for node in nodes:
        result = subprocess.run(transport(node,[python,'-c',check]),capture_output=True,text=True,check=True)
        print(node['name'],result.stdout.strip(),flush=True)
    jobs = queue.Queue()
    for shard in range(args.first_shard,run['shards']):
        jobs.put(shard)
    tasks = {}
    stop = threading.Event()
    launch_lock = threading.Lock()

    def slot(node,gpu,lane):
        while not stop.is_set():
            try:
                shard=jobs.get_nowait()
            except queue.Empty:
                return
            task=dict(shard=shard,node=node['name'],gpu=gpu,lane=lane,status='running')
            tasks[shard]=task
            command=[python,str(ROOT/'scripts/pipeline2_cache_cluster.py'),'--worker','--gpu',str(gpu),
                     '--manifest',str(root/'manifests'/f'shard{shard:03d}.json'),
                     '--output',str(root/'train'/f'shard{shard:03d}')]
            guarded=[python,str(ROOT/'scripts/pipeline2_cluster.py'),'--guard',*command]
            try:
                with (root/'logs'/f'assist_shard{shard:03d}.log').open('x') as log:
                    with launch_lock:
                        process=subprocess.Popen(transport(node,guarded),stdin=subprocess.PIPE,stdout=log,stderr=subprocess.STDOUT)
                        time.sleep(.3)
                    task['agent_pid']=process.pid
                    while process.poll() is None:
                        if stop.wait(1):
                            process.stdin.close()
                            break
                    code=process.wait(timeout=45)
                    if not process.stdin.closed:
                        process.stdin.close()
                task.update(status='complete' if code==0 else 'failed',returncode=code)
            except Exception as exc:
                task.update(status='failed',error=str(exc))
            finally:
                jobs.task_done()

    slots=[(node,gpu,lane) for node in nodes for gpu in node['gpus'] for lane in (2,3)]
    snapshot(root/'assist_run.json',dict(pid=os.getpid(),first_shard=args.first_shard,
        additional_workers=len(slots),target_total_workers_per_gpu=4,started=time.time()))
    with ThreadPoolExecutor(max_workers=len(slots)) as pool:
        futures=[pool.submit(slot,*s) for s in slots]
        try:
            while True:
                if (root/'STOP').exists() or (root/'failed.json').exists():
                    stop.set()
                counts=dict(feature_files=0,text_files=0,completed_shards=0)
                for directory in (root/'train').glob('shard*'):
                    if not directory.is_dir():
                        continue
                    names=[entry.name for entry in os.scandir(directory)]
                    counts['feature_files']+=sum(n.startswith('features_') and n.endswith('.pt') for n in names)
                    counts['text_files']+=sum(n.startswith('text_') and n.endswith('.pt') for n in names)
                    counts['completed_shards']+=int('completed.json' in names)
                snapshot(root/'overall_progress.json',dict(time=time.time(),total_samples=run['samples'],
                    **counts,global_index_published=(root/'train/cache_index.json').exists(),
                    assist_tasks=list(tasks.values()),training_started=False))
                print(json.dumps(counts),flush=True)
                if all(f.done() for f in futures) and ((root/'completed.json').exists() or stop.is_set()):
                    break
                time.sleep(20)
            for f in futures:
                f.result()
        finally:
            stop.set()


if __name__=='__main__':
    main()
