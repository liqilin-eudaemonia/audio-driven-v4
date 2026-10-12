"""Authorized dependency queue: GPU gates -> full cache -> C0 -> measured gate -> A1000."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from omni_av_p3_staged.orchestration import read,write,sha,notify,prerequisite,wait_idle,run_commands,run_pool,stopped,engineering_complete,update_phase
from omni_av_p3_staged.quality import QUALITY_VERSION


def render_gate(root,plan,cluster,update):
    """Real native generation/decode using engineering weights, not a quality claim."""
    base=root/'render_gate'
    if (base/'completed.json').exists():return
    from pipeline3_staged_quality import selected
    m=read(plan['manifest']);indices=selected(m);digest=sha(plan['manifest'])
    ids=base/'ids.json';write(ids,[m['dev'][i]['sample_id'] for i in indices])
    wait_idle(root,cluster,update);update(status='testing_generation',stage=None,folder=str(base))
    commands=[]
    for gpu in range(8):
        command=['env',f'CUDA_VISIBLE_DEVICES={gpu}','OMP_NUM_THREADS=2',cluster['python'],cluster['source']+'/scripts/pipeline3_staged_cache.py',
            '--manifest',plan['manifest'],'--output',str(root/'readouts'),'--ids',str(ids),'--shard',str(gpu),'--shards','8','--control-root',str(root)]
        commands.append((f'dev_cache_{gpu}',cluster['nodes'][0],command))
    run_commands(root,cluster,commands,base/'cache_logs',timeout=7200)
    jobs=[]
    for variant in ('text','C0','A'):
        cp=root/f'gate24_{"A" if variant=="A" else "C0"}_resume/step_000003.pt'
        for i in indices:
            job=dict(engineering_gate=True,manifest=plan['manifest'],manifest_sha256=digest,cache=str(root/'readouts'),index=i,wrong_index=i,
                mode='text' if variant=='text' else 'pred25',checkpoint=str(cp),checkpoint_sha256=sha(cp),
                negative_text=plan['negative_text'],negative_sha256=plan['negative_sha256'],seed=1045+i,output=str(base/'generation'/f'{variant}_{i:03d}'))
            path=base/'jobs'/f'{variant}_{i:03d}.json';write(path,job)
            jobs.append(dict(name=f'{variant}_{i:03d}',command=[cluster['python'],cluster['source']+'/scripts/pipeline3_staged_generate.py','--job',str(path),'--gpu','{GPU}']))
    run_pool(root,cluster,jobs,base/'logs',update)
    write(base/'completed.json',dict(passed=True,cases=24,variants=['text','C0','A'],steps=30,note='Engineering weights; finite output and geometry gate only'))


def cluster_training(root,plan,cluster,stage,output,initialize=None,resume=None,gate=False,stop_after=None):
    args=['--manifest',plan['manifest'],'--cache',str(root/'readouts'),'--stage',stage,'--output',str(output),'--control-root',str(root)]
    args+=['--resume',str(resume)] if resume else ['--initialize',str(initialize)]
    if gate:
        args+=['--engineering-gate','--gate-steps','3','--gate-indices',str(root/'gate_indices.json')]
        if stop_after:args+=['--gate-stop-after',str(stop_after)]
    elif stage=='A' and not resume:
        if plan.get('early_transition_authorization'):
            args+=['--transition-authorization',plan['early_transition_authorization']]
        else:args+=['--quality-gate',str(root/'C0_quality_gate.json')]
    network=read(plan['network_config']);commands=[]
    for rank,node in enumerate(cluster['nodes']):
        env=['env',f"CUDA_VISIBLE_DEVICES={','.join(map(str,node['gpus']))}",'OMP_NUM_THREADS=2','PYTHONUNBUFFERED=1',
             'TOKENIZERS_PARALLELISM=false','PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True','TORCH_NCCL_ASYNC_ERROR_HANDLING=1']
        env += [f'{k}={v}' for k,v in network.items()]
        command=[*env,cluster['python'],'-m','torch.distributed.run','--nproc_per_node=8','--nnodes=3','--node_rank',str(rank),
                 '--master_addr',cluster['nodes'][0]['host'],'--master_port',str(cluster['master_port']),'--max_restarts=0',
                 cluster['source']+'/scripts/pipeline3_staged_train.py',*args]
        commands.append((node['name'],node,command))
    run_commands(root,cluster,commands,output.with_name(output.name+'_agents'),heartbeat=lambda:True,timeout=plan['training_phase_timeout_seconds'])


def latest_checkpoint(root,stage):
    candidates=[]
    for folder in root.glob(stage+'_attempt*'):
        if not folder.is_dir() or folder.name.endswith('_agents'):continue
        for marker in folder.glob('step_*.complete.json'):
            d=read(marker);path=marker.with_name(marker.name.removesuffix('.complete.json')+'.pt')
            if path.is_file():candidates.append((d['step'],path))
    for step,path in sorted(candidates,reverse=True):
        if sha(path)==read(path.with_suffix('.complete.json'))['sha256']:return path
    return None


def train_stage(root,plan,cluster,stage,initial,update):
    done=root/(stage+'_completed.json')
    if done.exists():return Path(read(done)['checkpoint'])
    target=plan['stage_steps'][stage]
    for attempt in range(plan['max_training_attempts']):
        out=root/f'{stage}_attempt{attempt}'
        if (out/'completed.json').exists():
            result=read(out/'completed.json')
            if result['step']==target and not result['stopped']:
                cp=out/f'step_{target:06d}.pt';write(done,dict(checkpoint=str(cp),step=target,sha256=sha(cp)));return cp
        if out.exists():continue
        wait_idle(root,cluster,lambda **kw:update(stage=stage,**kw))
        cp=latest_checkpoint(root,stage)
        update(status='training',stage=stage,attempt=attempt,folder=str(out),resume=str(cp) if cp else None,target=target)
        try:cluster_training(root,plan,cluster,stage,out,initialize=initial,resume=cp)
        except InterruptedError:raise
        except Exception as e:
            write(root/f'{stage}_attempt{attempt}_failure.json',dict(error=repr(e),time=time.time()))
            notify(root,plan['thread'],'training_attempt_failed',dict(stage=stage,attempt=attempt,error=repr(e)))
            # Numerical errors, OOM and contract failures require diagnosis, not blind retries.
            logs='\n'.join(p.read_text(errors='replace')[-12000:] for p in out.with_name(out.name+'_agents').glob('*.log'))
            if any(x in logs for x in ['OutOfMemoryError','Nonfinite','mismatch','Frozen DiT changed','Ranks diverged']):raise
            continue
        result=read(out/'completed.json')
        if result['stopped'] or stopped(root):raise InterruptedError('Checkpointed user stop')
        cp=out/f'step_{target:06d}.pt';write(done,dict(checkpoint=str(cp),step=target,sha256=sha(cp)));return cp
    raise RuntimeError(f'{stage} attempts exhausted; complete checkpoints preserved')


def full_cache(root,plan,cluster,update):
    cache=root/'readouts'
    if (cache/'complete.json').exists():
        if read(cache/'complete.json')['manifest_sha256']!=sha(plan['manifest']):raise ValueError('Cache seal changed')
        return
    for attempt,workers in enumerate([4,2,1]):
        wait_idle(root,cluster,lambda **kw:update(phase='full_cache',**kw))
        commands=[];slots=[(n,g) for n in cluster['nodes'] for g in n['gpus']]
        for shard,(node,gpu) in enumerate(slots):
            for worker in range(workers):
                command=['env',f'CUDA_VISIBLE_DEVICES={gpu}','OMP_NUM_THREADS=2','PYTHONUNBUFFERED=1',cluster['python'],cluster['source']+'/scripts/pipeline3_staged_cache.py',
                    '--manifest',plan['manifest'],'--output',str(cache),'--shard',str(shard),'--shards','24',
                    '--worker-count',str(workers),'--worker-index',str(worker),'--control-root',str(root),
                    '--progress-dir',str(root/f'cache_attempt{attempt}'/'progress')]
                commands.append((f'shard{shard:02d}_worker{worker}',node,command))
        update(status='extracting_full_cache',workers_per_gpu=workers,folder=str(root/f'cache_attempt{attempt}'))
        try:run_commands(root,cluster,commands,root/f'cache_attempt{attempt}',timeout=plan['cache_timeout_seconds']);break
        except InterruptedError:raise
        except Exception as e:
            notify(root,plan['thread'],'cache_attempt_failed',dict(attempt=attempt,error=repr(e),next_workers=2 if workers==4 else 1))
            if workers==1:raise
    update(status='sealing_cache',folder=str(cache))
    command=[cluster['python'],cluster['source']+'/scripts/pipeline3_staged_seal.py','--manifest',plan['manifest'],
             '--output',str(cache),'--workers','8','--control-root',str(root)]
    run_commands(root,cluster,[('seal',cluster['nodes'][0],command)],root/'cache_seal_logs',timeout=plan['cache_timeout_seconds'])


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root.resolve()
    lock=(root/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    def stop(*_):(root/'STOP').touch()
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    plan=read(root/'plan.json');cluster=read(root/'cluster.json');m=read(plan['manifest']);manifest_sha=sha(plan['manifest'])
    state=dict(pid=os.getpid(),status='waiting_prerequisite',time=time.time(),source=str(ROOT),stage=None)
    def update(**kw):update_phase(state,**kw);write(root/'state.json',state)
    update()
    try:
        if plan.get('early_transition_authorization'):
            authorization=read(plan['early_transition_authorization'])
            c0=Path(authorization['checkpoint'])
            from omni_av_p3_staged.checkpoint import checked,authorized_transition
            if not prerequisite(plan['prerequisite']):raise ValueError('Predecessor validation incomplete')
            if not authorized_transition(authorization,c0,checked(c0),manifest_sha):
                raise ValueError('Invalid explicit C0 to A transition')
            if plan['stage_steps']!={'C0':authorization['from_step'],'A':1000}:
                raise ValueError('Authorized stage lengths mismatch')
            cp=train_stage(root,plan,cluster,'A',c0,update)
            update(status='completed',stage='A',step=1000,checkpoint=str(cp));write(root/'completed.json',state)
            notify(root,plan['thread'],'training_completed',dict(checkpoint=str(cp),C0_steps=authorization['from_step'],A_steps=1000))
            return
        alerted=0
        while not prerequisite(plan['prerequisite']):
            if stopped(root):raise InterruptedError('User STOP')
            prerequisite_state=read(Path(plan['prerequisite'])/'state.json')
            update(status='waiting_prerequisite',prerequisite_status=prerequisite_state.get('status'))
            if prerequisite_state.get('status')=='failed' and time.time()-alerted>1200:
                notify(root,plan['thread'],'prerequisite_failed',dict(path=plan['prerequisite'],error=prerequisite_state.get('error')));alerted=time.time()
            time.sleep(15)
        # Local eight-card testing was separately authorized during predecessor evaluation.
        while not (root/'local_gate/completed.json').exists():
            if stopped(root):raise InterruptedError('User STOP')
            local=read(root/'local_gate/state.json')
            update(status='waiting_local_gate',local_gate=local)
            if local.get('phase')=='failed':raise RuntimeError('Local GPU gate failed; inspect logs before continuing')
            time.sleep(15)
        if not read(root/'local_gate/completed.json')['passed']:raise RuntimeError('Local gate not passed')
        indices=[]
        for p in sorted((root/'local_gate').glob('indices_*.json')):indices+=read(p)
        write(root/'gate_indices.json',sorted(set(indices)))
        rebuilt=False
        for stage in ['C0','A']:
            for phase in ['initial','resume']:
                out=root/f'gate24_{stage}_{phase}'
                if stopped(root):raise InterruptedError('Persistent STOP')
                if not rebuilt and engineering_complete(out,stage,phase,manifest_sha,24):continue
                rebuilt=True
                aggregate=root/'gpu_gate_passed.json'
                if aggregate.exists():aggregate.rename(root/f'gpu_gate_passed.superseded_{time.time_ns()}.json')
                if (root/'render_gate').exists():
                    (root/'render_gate').rename(root/f'render_gate_superseded_{time.time_ns()}')
                if out.exists():out.rename(out.with_name(out.name+f'_failed_{time.time_ns()}'))
                wait_idle(root,cluster,update);update(status='testing_24gpu',stage=stage,phase=phase,folder=str(out))
                initial=m['config']['initial_connector']['path'] if stage=='C0' else str(root/'gate24_C0_resume/step_000003.pt')
                resume_path=root/f'gate24_{stage}_initial/step_000002.pt' if phase=='resume' else None
                cluster_training(root,plan,cluster,stage,out,initialize=initial,resume=resume_path,gate=True,stop_after=2 if phase=='initial' else None)
                if not engineering_complete(out,stage,phase,manifest_sha,24):raise RuntimeError('GPU gate incomplete or checkpoint mismatch')
        write(root/'gpu_gate_passed.json',dict(passed=True,time=time.time(),world=24,stages=['C0','A'],resume_tested=True))
        render_gate(root,plan,cluster,update)
        full_cache(root,plan,cluster,update)
        update(status='preparing_input_integrity',folder=str(root/'readouts/inputs'))
        command=[cluster['python'],cluster['source']+'/scripts/pipeline3_staged_integrity.py',
                 '--manifest',plan['manifest'],'--output',str(root/'readouts/inputs'),'--control-root',str(root)]
        run_commands(root,cluster,[('input_integrity',cluster['nodes'][0],command)],root/'input_integrity_logs',timeout=plan['cache_timeout_seconds'])
        c0=train_stage(root,plan,cluster,'C0',m['config']['initial_connector']['path'],update)
        gate=root/'C0_quality_gate.json'
        if not gate.exists() or read(gate).get('quality_version')!=QUALITY_VERSION:
            wait_idle(root,cluster,update);update(status='C0_quality_evaluation',stage='C0')
            from pipeline3_staged_quality import build
            build(root,plan,cluster,c0,update)
        if not read(gate)['passed']:
            update(status='needs_attention',reason='C0 automatic non-regression gate failed',gate=str(gate))
            notify(root,plan['thread'],'C0_quality_gate_failed',dict(gate=str(gate),checks=read(gate)['checks']));return
        cp=train_stage(root,plan,cluster,'A',c0,update)
        update(status='completed',stage='A',step=1000,checkpoint=str(cp));write(root/'completed.json',state)
        notify(root,plan['thread'],'training_completed',dict(checkpoint=str(cp),C0_steps=2000,A_steps=1000))
    except InterruptedError as e:update(status='stopped',reason=str(e))
    except BaseException as e:
        update(status='failed',error=repr(e));notify(root,plan['thread'],'queue_failed',dict(error=repr(e),folder=state.get('folder')));raise
if __name__=='__main__':main()
