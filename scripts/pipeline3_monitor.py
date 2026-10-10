#!/usr/bin/env python3
"""Append-only GPU, cache progress and P3 training telemetry; no process mutation."""
import argparse
import json
from pathlib import Path
import statistics
import time
import subprocess
from concurrent.futures import ThreadPoolExecutor

from pipeline21_supervise import tail
from pipeline3_full_cache import transport


def gpu_snapshot(cfg):
    def query(node):
        fields = ['index', 'used_mib', 'total_mib', 'util', 'watts', 'temp_c']
        command = ['nvidia-smi', '--query-gpu=index,memory.used,memory.total,utilization.gpu,power.draw,temperature.gpu',
                   '--format=csv,noheader,nounits']
        try:
            result = subprocess.run(transport(cfg, node, command), capture_output=True, text=True, timeout=20, check=True)
            gpus = [dict(zip(fields, map(float, line.split(',')))) for line in result.stdout.splitlines()]
            return dict(node=node['name'], host=node['host'], ok=True, gpus=gpus)
        except Exception as e:
            return dict(node=node['name'], host=node['host'], ok=False, error=str(e))
    with ThreadPoolExecutor(max_workers=len(cfg['nodes'])) as pool:
        return dict(time=time.time(), nodes=list(pool.map(query, cfg['nodes'])))


def latest_json(path):
    for line in reversed(tail(path).splitlines()):
        try:
            value=json.loads(line)
            if isinstance(value,dict): return value
        except (ValueError,TypeError): pass
    return {}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--cluster',type=Path,required=True)
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--run-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--interval',type=float,default=15)
    a=p.parse_args()
    cfg=json.loads(a.cluster.read_text())
    world=sum(len(n['gpus']) for n in cfg['nodes'])
    a.output.mkdir(parents=True,exist_ok=False)
    with (a.output/'telemetry.jsonl').open('x') as stream:
        while not (a.output/'STOP').exists():
            snapshot=gpu_snapshot(cfg)
            progress={}
            for log in sorted((a.cache/'logs').glob('*.log')):
                row=latest_json(log)
                if 'shard' in row:
                    progress[str(row['shard'])]=row.get('count',row.get('index',-1)+1)
            runs=[]
            for path in a.run_root.glob('*/train_rank0.jsonl'):
                row=latest_json(path)
                if row:
                    rows=[latest_json(path.parent/f'train_rank{rank}.jsonl') for rank in range(world)]
                    runs.append(dict(path=str(path.parent),last=row,rank_steps=[r.get('step') for r in rows],
                        last_modified=path.stat().st_mtime,complete=(path.parent/'completed.json').exists(),
                        failures=[str(p) for p in path.parent.glob('failed_rank*.json')]))
            report=dict(time=time.time(),gpu=snapshot,cache_count=sum(progress.values()),
                        cache_shards_reporting=len(progress),cache_complete=(a.cache/'completed.json').exists(),runs=runs)
            report['issues']=[dict(kind='gpu_query_failed',node=n['node']) for n in snapshot['nodes'] if not n['ok']]
            for r in runs:
                if r['failures']:
                    report['issues'].append(dict(kind='worker_failed',run=r['path'],files=r['failures']))
                if not r['complete'] and report['time']-r['last_modified']>900:
                    report['issues'].append(dict(kind='no_training_progress_15min',run=r['path']))
            stream.write(json.dumps(report)+'\n');stream.flush()
            print(json.dumps(dict(time=report['time'],cache=report['cache_count'],
                gpu_mean_util=statistics.mean(g['util'] for n in snapshot['nodes'] if n['ok'] for g in n['gpus'])
                    if any(n['ok'] for n in snapshot['nodes']) else None,
                runs=[dict(path=r['path'],step=r['last']['step']) for r in runs])),flush=True)
            if (a.run_root/'training/completed.json').exists() or (a.run_root/'FAILED.json').exists():
                break
            time.sleep(a.interval)


if __name__=='__main__': main()
