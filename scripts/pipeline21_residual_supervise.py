#!/usr/bin/env python3
"""Own one authorized three-stage run; the independent watch delivers alerts.

ROOT/STOP requests a checkpointed stop. Each stage is a separate guarded cluster
transaction. Only finite, complete same-stage checkpoints may resume, and only
two retries for recognizable transport/IO failures are permitted.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import os
from pathlib import Path
import shlex
import signal
import statistics
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def write(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)


def tail(path, size=65536):
    if not path.exists(): return ''
    with path.open('rb') as f:
        f.seek(max(0, path.stat().st_size-size))
        return f.read().decode(errors='replace')


def metrics(folder):
    rows = []
    for line in tail(folder/'train_rank0.jsonl').splitlines():
        try: rows.append(json.loads(line))
        except ValueError: pass
    return dict(latest=rows[-1] if rows else None,
                recent_step_seconds=statistics.mean(r['seconds'] for r in rows[-20:]) if rows else None)


def gpu_status(cluster):
    def one(node):
        query = ['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu,power.draw,temperature.gpu',
                 '--format=csv,noheader,nounits']
        command = query
        if not node.get('local'):
            proxy = ['-o','ProxyCommand='+node['proxy_command']] if node.get('proxy_command') else []
            control=['-S',node['control_path'],'-o','ControlMaster=auto','-o','ControlPersist=14400'] if node.get('control_path') else []
            command = ['ssh',*control,*proxy,'-i',cluster['ssh_key'],'-o','BatchMode=yes','-o','ConnectTimeout=10',
                       '-o','StrictHostKeyChecking=yes','root@'+node['host'],shlex.join(query)]
        try:
            p = subprocess.run(command, capture_output=True, text=True, timeout=25, check=True)
            rows = [dict(zip(('gpu','memory_mib','util','watts','temperature_c'),map(float,l.split(','))))
                    for l in p.stdout.splitlines()]
            rows = [r for r in rows if int(r['gpu']) in node['gpus']]
            if len(rows) != len(node['gpus']): raise ValueError('Missing GPUs')
            return dict(node=node['name'],host=node['host'],ok=True,gpus=rows)
        except Exception as e:
            return dict(node=node['name'],host=node['host'],ok=False,error=str(e))
    with ThreadPoolExecutor(max_workers=len(cluster['nodes'])) as pool:
        return list(pool.map(one,cluster['nodes']))


def latest_checkpoint(folder, stage):
    records = []
    for p in folder.glob(f'stage{stage}_step_*.complete.json'):
        r = json.loads(p.read_text())
        model = p.with_name(p.name.replace('.complete.json','.pt'))
        shards = r['optimizer_shards']
        if (r['stage']==stage and model.is_file() and len(shards)==r['world']
                and all(p.with_name(s['file']).is_file() for s in shards)):
            records.append((r['step'],model))
    # The trainer independently verifies model/shard SHA and the full identity.
    return max(records,default=(0,None))


def transient(text):
    text = text.lower()
    if any(s in text for s in ('out of memory','nonfinite','checksum mismatch','changed','invalid checkpoint')):
        return False
    return any(s in text for s in ('ncclsystemerror','ncclremoteerror','connection reset','connection closed',
                                   'timed out','collective operation timeout','input/output error'))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    a = p.parse_args(); root = a.root.resolve()
    plan = json.loads((root/'plan.json').read_text())
    cluster_path = root/'cluster.json'; cluster = json.loads(cluster_path.read_text())
    lock = (root/'supervisor.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    signal.signal(signal.SIGTERM,lambda *_:(root/'STOP').touch())
    signal.signal(signal.SIGINT,lambda *_:(root/'STOP').touch())
    state = dict(pid=os.getpid(),status='starting',time=time.time(),plan=plan)
    write(root/'state.json',state)
    previous = None
    for stage,target in enumerate(plan['boundaries'],1):
        start = 0 if stage==1 else plan['boundaries'][stage-2]
        resume = None
        for attempt in range(3):
            if (root/'STOP').exists():
                state.update(status='stopped',time=time.time());write(root/'state.json',state);return
            folder = root/f'stage{stage}_attempt{attempt}'
            agents = root/f'stage{stage}_attempt{attempt}_agents'
            # A restarted controller must never replay completed stages or
            # overwrite an interrupted attempt. The trainer checks resume SHA.
            done = folder/'completed.json'
            if done.is_file():
                result=json.loads(done.read_text())
                if result['stopped']:
                    state.update(status='stopped',result=result,time=time.time());write(root/'state.json',state);return
                if result['end_step']==target and result['ranks_identical']:
                    previous=Path(result['checkpoint']);break
            if (root/f'stage{stage}_attempt{attempt}_launch.log').exists():
                new_step,new_resume=latest_checkpoint(folder,stage)
                if new_resume: start,resume=new_step,new_resume
                if attempt==2:
                    state.update(status='failed',error='Attempt budget exhausted',time=time.time())
                    write(root/'state.json',state);return
                continue
            command = [cluster['python'],str(ROOT/'scripts/pipeline21_residual_cluster.py'),
                       '--cluster',str(cluster_path),'--log-dir',str(agents),'--',
                       '--stage',str(stage),'--steps',str(target-start),'--accumulation',str(plan['accumulation']),
                       '--save-every','100','--cache',plan['cache'],'--output',str(folder),
                       '--config',plan['config'],'--cache-samples','0','--allow-terminal-trim',
                       '--balance-workload','--network-config',plan['network_config'],
                       '--run-id',f'residual-stage{stage}-attempt{attempt}']
            if stage==1: command+=['--no-decoder-checkpointing']
            else: command+=['--gradient-communication','bf16']
            if resume: command+=['--resume',str(resume)]
            elif previous: command+=['--init-previous',str(previous)]
            state.update(status='running',stage=stage,target=target,attempt=attempt,folder=str(folder),
                         command=command,time=time.time())
            write(root/'state.json',state)
            started = time.time(); last_progress=started; last_step=start; history=[]
            with (root/f'stage{stage}_attempt{attempt}_launch.log').open('x') as log:
                process = subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                try:
                    while process.poll() is None:
                        if (root/'STOP').exists() and folder.is_dir(): (folder/'STOP').touch()
                        snapshot = dict(time=time.time(),stage=stage,attempt=attempt,
                                        nodes=gpu_status(cluster),**metrics(folder))
                        latest=snapshot['latest']
                        if latest and latest['step']>last_step:
                            last_step=latest['step'];last_progress=time.time()
                        issues=[]
                        if any(not n['ok'] for n in snapshot['nodes']): issues.append('node_unreachable')
                        if list(folder.glob('failed_rank*.json')): issues.append('worker_failure')
                        if time.time()-last_progress>900: issues.append('no_progress_15min')
                        history.append(snapshot);history=history[-30:]
                        if len(history)==30:
                            utils=[g['util'] for s in history for n in s['nodes'] if n['ok'] for g in n['gpus']]
                            if utils and statistics.mean(utils)<70: issues.append('low_utilization_15min')
                        snapshot['issues']=issues
                        write(root/'health.json',snapshot)
                        with (root/'gpu_history.jsonl').open('a') as f:f.write(json.dumps(snapshot)+'\n')
                        state.update(time=time.time(),latest=latest,recent_step_seconds=snapshot['recent_step_seconds'])
                        write(root/'state.json',state)
                        if issues:
                            with (root/'alerts.jsonl').open('a') as f:f.write(json.dumps(snapshot)+'\n')
                        if 'worker_failure' in issues:
                            # A rank has published a fatal exception. Its NCCL
                            # teardown can wait15min; close this run's guarded
                            # cluster now rather than spin on the surviving GPUs.
                            process.terminate()
                            try:process.wait(timeout=60)
                            except subprocess.TimeoutExpired:process.kill();process.wait()
                            break
                        # Metric or probe warnings alone never kill a working job.
                        for _ in range(30):
                            if process.poll() is not None:break
                            if (root/'STOP').exists() and folder.is_dir(): (folder/'STOP').touch()
                            time.sleep(1)
                finally:
                    if process.poll() is None:
                        process.terminate() # guarded launcher reaps only its own agents
                        try:process.wait(timeout=60)
                        except subprocess.TimeoutExpired:process.kill();process.wait()
            result_path=folder/'completed.json'
            if process.returncode==0 and result_path.is_file():
                result=json.loads(result_path.read_text())
                if result['stopped'] or (root/'STOP').exists():
                    state.update(status='stopped',result=result,time=time.time());write(root/'state.json',state);return
                if result['end_step']!=target or not result['ranks_identical']:
                    raise RuntimeError('Stage incomplete or ranks differ')
                previous=Path(result['checkpoint']);break
            errors='\n'.join(tail(x) for x in [*agents.glob('*.log'),*folder.glob('failed_rank*.json')])
            new_step,new_resume=latest_checkpoint(folder,stage)
            if new_resume: start,resume=new_step,new_resume
            if attempt==2 or not transient(errors) or start>=target:
                state.update(status='failed',returncode=process.returncode,error_tail=errors[-8000:],time=time.time())
                write(root/'state.json',state);return
            state.update(status='retrying',resume=str(resume),time=time.time());write(root/'state.json',state)
            # All guarded children have exited; preflight will reject occupied GPUs.
            time.sleep(10)
    state.update(status='completed',checkpoint=str(previous),time=time.time())
    write(root/'state.json',state);write(root/'completed.json',state)


if __name__=='__main__':main()
