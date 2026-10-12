"""Dedicated eight-GPU diagnostic queue; does not mutate the training run."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def retryable_connection_failure(returncode,log_text,attempts):
    return returncode==255 and attempts<3 and any(s in log_text for s in (
        'kex_exchange_identification: Connection closed','Connection timed out',
        'Connection refused','mux_client_request_session: session request failed'))


def diagnostic_pool(root,cluster,jobs,logs,update,timeout=10800):
    from omni_av_p3_staged.orchestration import stopped,probe,free_slots,transport,finish_owned
    pending=[dict(j,attempts=0) for j in jobs];active=[];logs=Path(logs);logs.mkdir(parents=True,exist_ok=True)
    last_probe=0;available=[]
    try:
        while pending or active:
            if stopped(root):raise InterruptedError('Diagnostic STOP')
            for task in active[:]:
                rc=task['process'].poll()
                if rc is None:continue
                task['process'].stdin.close();task['log'].close();active.remove(task)
                if rc:
                    text=(logs/(task['name']+'.log')).read_text(errors='replace')[task['log_offset']:]
                    job=task['job']
                    if retryable_connection_failure(rc,text,job['attempts']):
                        job['attempts']+=1;pending.append(job)
                        with (root/'transport_retries.jsonl').open('a') as f:
                            f.write(json.dumps(dict(time=time.time(),job=job['name'],retry=job['attempts'],error=text[-1000:]))+'\n')
                    else:raise RuntimeError(f"Pool job failed: {task['name']} rc={rc}")
            if time.time()-last_probe>15:
                busy={(t['node']['name'],t['gpu']) for t in active}
                available=[(n,g) for n,g in free_slots(cluster,probe(cluster)) if (n['name'],g) not in busy];last_probe=time.time()
            while pending and available:
                node,gpu=available.pop();job=pending.pop(0)
                command=[str(gpu) if x=='{GPU}' else x for x in job['command']]
                guard=[cluster['python'],cluster['source']+'/scripts/pipeline3_staged_worker.py','guard','--',*command]
                path=logs/(job['name']+'.log');offset=len(path.read_text(errors='replace')) if path.exists() else 0
                f=path.open('a');process=subprocess.Popen(transport(cluster,node,guard),stdin=subprocess.PIPE,stdout=f,stderr=subprocess.STDOUT)
                active.append(dict(name=job['name'],node=node,gpu=gpu,process=process,log=f,start=time.time(),job=job,log_offset=offset))
            if any(time.time()-t['start']>timeout for t in active):raise TimeoutError('Diagnostic job timeout')
            update(pending=len(pending),active=[dict(name=t['name'],node=t['node']['name'],gpu=t['gpu']) for t in active])
            time.sleep(3)
    finally:finish_owned(active)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    root=parser.parse_args().root.resolve()
    plan=json.loads((root/'plan.json').read_text());source=Path(plan['source'])
    sys.path.insert(0,str(source/'src'))
    from omni_av_p3_staged.orchestration import read,write,probe,notify,sha
    lock=(root/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    cluster=read(root/'cluster.json');state=dict(pid=os.getpid(),status='starting',time=time.time())
    last_health=0
    def update(**changes):
        nonlocal last_health
        state.update(changes,time=time.time());write(root/'state.json',state)
        if time.time()-last_health>=60:
            health=dict(time=time.time(),nodes=probe(cluster),phase=state.get('phase'))
            write(root/'health.json',health)
            with (root/'health_history.jsonl').open('a') as f:f.write(json.dumps(health)+'\n')
            last_health=time.time()
    try:
        for phase in plan['phases']:
            if (root/(phase+'_completed.json')).exists():continue
            jobs=read(root/(phase+'_jobs.json'))
            reused=0
            if phase=='generation':
                unfinished=[]
                for job in jobs:
                    command=job['command'];spec=read(command[command.index('--job')+1]);out=Path(spec['output'])
                    if (out/'completed.json').exists():
                        done=read(out/'completed.json')
                        if done['job']!=spec or done['video_sha256']!=sha(out/'generated.mp4'):raise ValueError('Completed generation identity changed')
                        reused+=1
                    else:unfinished.append(job)
                jobs=unfinished
            update(status='running',phase=phase,total=len(jobs))
            diagnostic_pool(root,cluster,jobs,root/(phase+'_logs'),update,timeout=10800)
            write(root/(phase+'_completed.json'),dict(time=time.time(),jobs=len(jobs),reused=reused))
        update(status='summarizing')
        subprocess.run([cluster['python'],str(root/'code/report.py'),'--root',str(root)],check=True)
        update(status='completed');write(root/'completed.json',state)
        notify(root,plan['thread'],'diagnostics_completed',dict(report=str(root/'report.json'),checkpoint=plan['checkpoint']))
    except BaseException as e:
        update(status='stopped' if isinstance(e,InterruptedError) else 'failed',error=repr(e))
        notify(root,plan['thread'],'diagnostics_failed',dict(error=repr(e),phase=state.get('phase')))
        raise


if __name__=='__main__':main()
