#!/usr/bin/env python3
"""Wait for exact full cache, benchmark microbatches, then own one 1000-step P3 job."""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from omni_av.data import write_json_new


def launch(a,name,steps,optimized,microbatch=1):
    command=[sys.executable,str(ROOT/'scripts/pipeline3_cluster.py'),'--cluster',str(a.cluster),
        '--log-dir',str(a.output/(name+'_logs')),'--','--cache',str(a.cache/'index.json'),
        '--output',str(a.output/name),'--arm','lora','--total-steps',str(steps),
        '--accumulation',str(4//microbatch),'--microbatch',str(microbatch),
        '--save-every',str(100 if name=='training' else steps),
        '--validate-every',str(100 if name=='training' else steps),'--threads','2',
        '--cache-samples','8','--timeout','900',
        '--network-config',str(ROOT/'configs/p3_nccl_roce_example.json')]
    if optimized: command+=['--prefetch','--balance-workload']
    write_json_new(a.output/(name+'_command.json'),dict(command=command,start=time.time()))
    with (a.output/(name+'_controller.log')).open('x') as log:
        process=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        write_json_new(a.output/(name+'_pid.json'),dict(pid=process.pid))
        code=process.wait()
    if code: raise RuntimeError(f'{name} exited {code}; inspect node logs')
    result=json.loads((a.output/name/'completed.json').read_text())
    if result['end_step']!=steps or not result['ranks_identical'] or result['stopped']:
        raise RuntimeError('Run did not complete requested steps consistently')
    rows=[json.loads(x) for x in (a.output/name/'train_rank0.jsonl').read_text().splitlines()]
    steady=rows[5:]
    return dict(result=result,mean_seconds=statistics.mean(r['seconds'] for r in steady),
                median_seconds=statistics.median(r['seconds'] for r in steady),
                samples_per_second=96/statistics.mean(r['seconds'] for r in steady))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cluster',type=Path,required=True)
    p.add_argument('--cache-pid',type=int,required=True)
    p.add_argument('--record-name',default='supervisor.json')
    a=p.parse_args()
    try:
        if Path(a.record_name).name != a.record_name:
            raise ValueError('Supervisor record must be a basename')
        write_json_new(a.output/a.record_name,dict(pid=os.getpid(),started=time.time(),
            target_steps=1000,train=91009,dev=48,arm='lora',global_batch=96,
            initialization='fresh heads/LoRA and pretrained Omni; no pilot weights',
            cache=str(a.cache),cluster=str(a.cluster)))
        while not (a.cache/'completed.json').exists():
            try: os.kill(a.cache_pid,0)
            except ProcessLookupError: raise RuntimeError('Cache controller exited without completion')
            if Path(f'/proc/{a.cache_pid}/stat').read_text().split()[2]=='Z':
                raise RuntimeError('Cache controller exited without completion (zombie process)')
            if (a.output/'STOP').exists(): raise RuntimeError('Stopped before training')
            time.sleep(15)
        cache=json.loads((a.cache/'index.json').read_text())
        if len(cache['train'])!=91009 or len(cache['dev'])!=48:
            raise RuntimeError('Full cache coverage mismatch')
        # Give extraction process guards time to finish reaping; no foreign tasks are killed.
        time.sleep(10)
        baseline=launch(a,'benchmark_reference',20,False)
        candidates=[dict(name='reference',optimized=False,microbatch=1,metrics=baseline)]
        failures=[]
        for microbatch in (1,2,4):
            name=f'benchmark_optimized_b{microbatch}'
            try:
                measured=launch(a,name,20,True,microbatch)
                candidates.append(dict(name=name,optimized=True,microbatch=microbatch,metrics=measured))
            except Exception:
                failures.append(dict(name=name,error=traceback.format_exc()))
                time.sleep(5)
        best=min(candidates,key=lambda c:c['metrics']['mean_seconds'])
        write_json_new(a.output/'benchmark.json',dict(candidates=candidates,failures=failures,selected=best,
            speedup=baseline['mean_seconds']/best['metrics']['mean_seconds'],
            warmup_steps=5,measured_steps=15,global_batch=96,
            note='Same initialization, global sample/noise set and update count; not bitwise trajectory equality'))
        result=launch(a,'training',1000,best['optimized'],best['microbatch'])
        write_json_new(a.output/'COMPLETED.json',result)
    except Exception:
        write_json_new(a.output/'FAILED.json',dict(time=time.time(),traceback=traceback.format_exc()))
        raise


if __name__=='__main__': main()
