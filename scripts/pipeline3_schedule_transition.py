"""Schedule an explicitly authorized C0 checkpoint -> A, with parallel evaluation.

The running C0 source/recipe remains immutable. Its existing output STOP is
observed at the next optimizer boundary, after the requested checkpoint/Dev save.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from omni_av_p3_staged.orchestration import read,write,sha,notify,stopped


def process_matches(pid, script, root):
    try:
        args=Path(f'/proc/{int(pid)}/cmdline').read_bytes().split(b'\0')
        return str(root).encode() in args and any(x.endswith(('/'+script).encode()) for x in args)
    except (OSError,TypeError,ValueError):return False


def last_step(folder):
    path=folder/'rank0.jsonl'
    if not path.exists():return 0
    with path.open('rb') as f:
        f.seek(max(0,path.stat().st_size-16384))
        lines=f.read().splitlines()
    for line in reversed(lines):
        try:return int(json.loads(line)['step'])
        except (ValueError,KeyError):continue
    return 0


def launch(python,script,root,log_name):
    with (root/log_name).open('ab') as log:
        child=subprocess.Popen([python,str(script),'--root',str(root)],stdin=subprocess.DEVNULL,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    return child.pid


def stop_watch(root):
    watch=read(root/'watch.json');pid=watch['pid']
    if process_matches(pid,'pipeline3_staged_watch.py',root):
        os.kill(pid,signal.SIGTERM)
        for _ in range(100):
            if not process_matches(pid,'pipeline3_staged_watch.py',root):return
            time.sleep(.1)
        raise RuntimeError('Owned watch did not exit')


def start_diagnostics(request, checkpoint, digest):
    template=Path(request['diagnostic_template']);dest=Path(request['diagnostic_root'])
    previous=read(template/'plan.json')
    if not (dest/'plan.json').exists():
        dest.mkdir(parents=True,exist_ok=True)
        files=[template/n for n in ('plan.json','cluster.json','generation_jobs.json','metric_jobs.json','metric_plan.json')]
        for directory in ('jobs','metric_jobs','metrics'):
            files+=list((template/directory).glob('*.json'))+list((template/directory).glob('*.jsonl'))
        for p in files:
            value=p.read_text().replace(str(template),str(dest)).replace(previous['checkpoint'],str(checkpoint))
            value=value.replace(previous['checkpoint_sha256'],digest)
            out=dest/p.relative_to(template);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(value)
        plan=read(dest/'plan.json');plan.update(checkpoint_step=request['from_step'],previous=str(template),
            source=str(ROOT),description=f"C0 step{request['from_step']} matched Dev8 Pred25; parallel with A1000",
            scheduling='User authorized node194 GPUs4-7; wait for idle cards, never stop other jobs')
        write(dest/'plan.json',plan)
        cluster=read(dest/'cluster.json');cluster['source']=str(ROOT);cluster['nodes'][0]['gpus']=[4,5,6,7];write(dest/'cluster.json',cluster)
        mapping={'supervise.py':'pipeline3_diagnostic_queue.py','report.py':'pipeline3_checkpoint_report.py',
                 'generate.py':'pipeline3_diagnostic_generate.py','cpu_identity.py':'pipeline3_diagnostic_cpu_identity.py'}
        (dest/'code').mkdir(exist_ok=True)
        for name,source in mapping.items():(dest/'code'/name).write_bytes((ROOT/'scripts'/source).read_bytes())
        write(dest/'code_manifest.json',{name:sha(dest/'code'/name) for name in mapping})
    plan=read(dest/'plan.json')
    if plan['checkpoint_sha256']!=digest:raise ValueError('Diagnostic checkpoint changed')
    if stopped(dest):return dict(status='stopped',root=str(dest))
    python=read(dest/'cluster.json')['python']
    for stem,script,state_file,done in [('cpu_identity','cpu_identity.py','cpu_identity_state.json','cpu_identity_completed.json'),
                                      ('queue','supervise.py','state.json','completed.json')]:
        state=read(dest/state_file) if (dest/state_file).exists() else {}
        if (dest/done).exists() or process_matches(state.get('pid'),script,dest):continue
        pid=launch(python,dest/'code'/script,dest,stem+'.log')
        write(dest/(stem+'_launch.json'),dict(pid=pid,time=time.time(),script=str(dest/'code'/script)))
    return dict(status='launched',root=str(dest))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True,type=Path)
    root=parser.parse_args().root.resolve();request=read(root/'scheduled_transition.json')
    lock=(root/'transition_queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state=dict(pid=os.getpid(),status='waiting_checkpoint',time=time.time(),from_step=request['from_step'])
    def update(**kw):state.update(kw,time=time.time());write(root/'transition_state.json',state)
    update()
    def check_stop():
        if stopped(root):raise InterruptedError('Persistent user STOP')
    def signal_stop(*_):(root/'STOP').touch()
    signal.signal(signal.SIGTERM,signal_stop);signal.signal(signal.SIGINT,signal_stop)
    folder=Path(request['c0_folder']);target=request['from_step'];checkpoint=folder/f'step_{target:06d}.pt'
    monitor_stopped=False;handed_over=False
    try:
        if request['user_authorized'] is not True or request['a_steps']!=1000:raise ValueError('Invalid request')
        while not (root/'early_transition_authorization.json').exists():
            check_stop();step=last_step(folder);update(observed_step=step)
            if step>=target:
                if step!=target:raise ValueError(f'C0 exceeded requested boundary: {step}')
                stop_watch(root);monitor_stopped=True
                control=folder/'STOP'
                if not control.exists():
                    with control.open('x') as f:f.write('User requested transition after C0 step700; retain this marker.\n')
                update(status='waiting_checkpoint_and_exit')
                break
            current=read(root/'state.json')
            if (current.get('stage')!='C0' or current.get('status')!='training'
                    or current.get('folder')!=str(folder)):
                raise RuntimeError('C0 state changed before requested boundary: '+repr(current))
            if (folder/'STOP').exists():raise InterruptedError('Existing C0 STOP before requested boundary')
            time.sleep(1 if step>=target-1 else 10)
        if not (root/'early_transition_authorization.json').exists():
            deadline=time.monotonic()+1800;diagnostic=None
            while True:
                check_stop()
                if checkpoint.with_suffix('.complete.json').exists() and diagnostic is None:
                    marker=read(checkpoint.with_suffix('.complete.json'))
                    if marker['step']!=target or sha(checkpoint)!=marker['sha256']:raise ValueError('C0 checkpoint integrity mismatch')
                    diagnostic=start_diagnostics(request,checkpoint,marker['sha256']);update(diagnostic=diagnostic)
                if (folder/'completed.json').exists():
                    done=read(folder/'completed.json')
                    if done['step']!=target or done['stage']!='C0' or done['engineering_gate']:
                        raise ValueError('C0 did not end at requested production checkpoint')
                    current=read(root/'state.json')
                    if not process_matches(current.get('pid'),'pipeline3_staged_queue.py',root):break
                if time.monotonic()>deadline:raise TimeoutError('C0 checkpoint/exit timeout')
                time.sleep(1)
            if diagnostic is None:raise ValueError('C0 completion lacks checked checkpoint')
            if not (folder/f'dev_{target:06d}.json').exists():raise ValueError('C0 final Dev evaluation missing')
            authorization=dict(request,version='p3.user-stage-transition.v1',checkpoint=str(checkpoint),
                checkpoint_sha256=sha(checkpoint),manifest_sha256=read(root/'plan.json')['manifest_sha256'],time=time.time())
            write(root/'early_transition_authorization.json',authorization)
        check_stop()
        # The old controller must have released its lock before deployment.
        queue_lock=(root/'queue.lock').open('a');fcntl.flock(queue_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        authorization=read(root/'early_transition_authorization.json')
        plan=read(root/'plan.json');cluster=read(root/'cluster.json')
        for name,value in [('plan_before_early_A.json',plan),('cluster_before_early_A.json',cluster)]:
            if not (root/name).exists():write(root/name,value)
        plan.update(source_snapshot=str(ROOT),stage_steps={'C0':target,'A':1000},
            early_transition_authorization=str(root/'early_transition_authorization.json'),thread=request['thread'],
            quality_policy='Explicit user transition at C0 step700; paired inference runs in parallel and does not gate A.')
        cluster['source']=str(ROOT);write(root/'cluster.json',cluster);write(root/'plan.json',plan)
        write(root/'C0_completed.json',dict(checkpoint=str(checkpoint),step=target,sha256=authorization['checkpoint_sha256'],
            termination='user_requested_early_transition',original_recipe_steps=2000,quality_gate_passed=None))
        write(root/'state.json',dict(pid=os.getpid(),status='starting_A',stage='A',time=time.time(),source=str(ROOT)))
        fcntl.flock(queue_lock,fcntl.LOCK_UN);queue_lock.close()
        python=cluster['python'];pid=launch(python,ROOT/'scripts/pipeline3_staged_queue.py',root,'queue_early_A.log')
        update(status='A_queue_launched',queue_pid=pid,checkpoint=str(checkpoint))
        # Give the new queue time to acquire the lock and publish its PID.
        for _ in range(60):
            check_stop()
            if read(root/'state.json').get('pid')==pid:break
            if not process_matches(pid,'pipeline3_staged_queue.py',root):raise RuntimeError('A queue exited during launch')
            time.sleep(1)
        else:raise TimeoutError('A queue state publication')
        watch_pid=launch(python,ROOT/'scripts/pipeline3_staged_watch.py',root,'watch_early_A.log')
        handed_over=True;update(status='completed',watch_pid=watch_pid)
        notify(root,request['thread'],'C0_700_to_A_started',dict(checkpoint=str(checkpoint),A_steps=1000,
            evaluation='parallel_nonblocking',diagnostic_root=request['diagnostic_root']))
    except BaseException as e:
        update(status='stopped' if isinstance(e,InterruptedError) else 'failed',error=repr(e))
        if monitor_stopped and not handed_over:
            current=read(root/'state.json')
            if not process_matches(current.get('pid'),'pipeline3_staged_queue.py',root):
                write(root/'state.json',dict(pid=os.getpid(),status='needs_attention',stage='C0',time=time.time(),reason=repr(e)))
        notify(root,request['thread'],'scheduled_transition_stopped' if isinstance(e,InterruptedError) else 'scheduled_transition_failed',dict(error=repr(e)))
        raise


if __name__=='__main__':main()
