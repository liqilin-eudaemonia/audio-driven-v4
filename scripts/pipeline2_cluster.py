#!/usr/bin/env python3
"""Preflight every node, then launch one guarded torchrun agent per node."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shlex
import signal
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def guard(command):
    import select
    process = subprocess.Popen(command, start_new_session=True)
    try:
        while process.poll() is None:
            ready, _, _ = select.select([sys.stdin.fileno()], [], [], .5)
            if ready and not os.read(sys.stdin.fileno(), 4096):
                break
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
        return process.wait()
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == '--guard':
        raise SystemExit(guard(sys.argv[2:]))
    p = argparse.ArgumentParser()
    p.add_argument('--cluster', type=Path, required=True)
    p.add_argument('--log-dir', type=Path, required=True)
    p.add_argument('arguments', nargs=argparse.REMAINDER)
    a = p.parse_args()
    config = json.loads(a.cluster.read_text())
    nodes, python = config['nodes'], config['python']
    if len({n['host'] for n in nodes}) != len(nodes) or len({len(n['gpus']) for n in nodes}) != 1:
        p.error('Distinct nodes and equal GPU counts required')
    args = a.arguments[1:] if a.arguments[:1] == ['--'] else a.arguments
    key = config.get('ssh_key', f'/private/ssh_keys/root_{socket.gethostname()}_ed25519')
    def transport(node, command):
        if node.get('local', False):
            return command
        return ['ssh', '-i', key, '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
                '-o', 'StrictHostKeyChecking=accept-new', f"root@{node['host']}", shlex.join(command)]
    commands = []
    for rank, node in enumerate(nodes):
        command = [python, str(ROOT/'scripts/pipeline2_parallel.py'), *args,
            '--nnodes', str(len(nodes)), '--node-rank', str(rank), '--master-addr', nodes[0]['host'],
            '--master-port', str(config['master_port']), '--gpus', ','.join(map(str,node['gpus']))]
        commands.append(command)
    a.log_dir.mkdir(parents=True, exist_ok=False)
    def check(item):
        node, command = item
        result = subprocess.run(transport(node, command+['--preflight-only']), capture_output=True, text=True, timeout=60)
        (a.log_dir/f"preflight_{node['name']}.log").write_text(result.stdout+result.stderr)
        if result.returncode:
            raise RuntimeError(f"Preflight failed: {node['name']}: {result.stderr}")
    with ThreadPoolExecutor(max_workers=len(nodes)) as pool:
        list(pool.map(check, zip(nodes,commands)))
    processes, logs = [], []
    try:
        for node, command in zip(nodes,commands):
            log=(a.log_dir/f"{node['name']}.log").open('x')
            logs.append(log)
            guarded=[python,str(Path(__file__).resolve()),'--guard',*command]
            processes.append(subprocess.Popen(transport(node,guarded), stdin=subprocess.PIPE,stdout=log,stderr=subprocess.STDOUT))
        while any(x.poll() is None for x in processes):
            if any(x.poll() not in (None,0) for x in processes):
                raise RuntimeError('A node failed; closing only this run agents')
            time.sleep(1)
        result=dict(returncodes=[x.returncode for x in processes],nodes=nodes)
        (a.log_dir/'completed.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result),flush=True)
    finally:
        for process in processes:
            process.stdin.close()
        for process in processes:
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.terminate()
        for log in logs:
            log.close()


if __name__ == '__main__':
    main()
