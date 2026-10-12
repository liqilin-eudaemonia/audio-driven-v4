"""Use only authorized local GPUs for readout and C0/A engineering tests."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from omni_av_p3_staged.orchestration import read,write,run_commands,probe,free_slots,notify,sha,engineering_complete


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--thread',required=True);a=p.parse_args()
    root=a.root;gate=root/'local_gate';gate.mkdir(exist_ok=True);lock=(gate/'lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    cluster=dict(python=sys.executable,source=str(ROOT),nodes=[dict(name='node12',host='172.29.252.24',local=True,gpus=list(range(8)))])
    node=cluster['nodes'][0];manifest=root/'manifest.json';m=read(manifest);cache=root/'readouts'
    token=lambda e:((e['metadata']['target_frames']-1)//8+1)*(e['metadata']['target_height']//32)*(e['metadata']['target_width']//32)
    ordered=sorted(range(len(m['train'])),key=lambda i:token(m['train'][i]))
    shorts=ordered[:8];longs=ordered[-8:];ids=[m['train'][i]['sample_id'] for i in shorts+longs]
    write(gate/'ids.json',ids)
    for i in range(8):write(gate/f'indices_{i}.json',[shorts[i],longs[i]])
    def update(phase,**kw):write(gate/'state.json',dict(pid=os.getpid(),phase=phase,time=time.time(),**kw))
    try:
        if len(free_slots(cluster,probe(cluster)))!=8:raise RuntimeError('Local GPU no longer idle')
        if not (gate/'cache_completed.json').exists():
            update('extracting_conditions')
            commands=[]
            for gpu in range(8):
                command=['env',f'CUDA_VISIBLE_DEVICES={gpu}','OMP_NUM_THREADS=2','PYTHONUNBUFFERED=1',sys.executable,str(ROOT/'scripts/pipeline3_staged_cache.py'),
                    '--manifest',str(manifest),'--output',str(cache),'--ids',str(gate/'ids.json'),'--shard',str(gpu),'--shards','8','--control-root',str(root)]
                commands.append((f'cache_gpu{gpu}',node,command))
            run_commands(root,cluster,commands,gate/'logs',timeout=7200)
            write(gate/'cache_completed.json',dict(time=time.time(),samples=len(ids),manifest_sha256=sha(manifest)))
        rebuilt=set();manifest_sha=sha(manifest)
        for stage in ['C0','A']:
            for phase in ['initial','resume']:
                outputs=[gate/f'{stage}_{phase}_gpu{gpu}' for gpu in range(8)]
                update(stage+'_'+phase)
                commands=[]
                for gpu,out in enumerate(outputs):
                    if gpu not in rebuilt and engineering_complete(out,stage,phase,manifest_sha,1):continue
                    rebuilt.add(gpu)
                    if out.exists():out.rename(out.with_name(out.name+f'_failed_{time.time_ns()}'))
                    command=['env',f'CUDA_VISIBLE_DEVICES={gpu}','PYTHONUNBUFFERED=1','OMP_NUM_THREADS=2','PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True',
                        'NCCL_SOCKET_IFNAME=eth0','GLOO_SOCKET_IFNAME=eth0',sys.executable,'-m','torch.distributed.run','--nproc_per_node=1',
                        '--nnodes=1','--master_addr=127.0.0.1','--master_port',str(29940+gpu),'--max_restarts=0',str(ROOT/'scripts/pipeline3_staged_train.py'),
                        '--manifest',str(manifest),'--cache',str(cache),'--stage',stage,'--output',str(out),'--control-root',str(root),
                        '--engineering-gate','--gate-steps','3','--gate-indices',str(gate/f'indices_{gpu}.json')]
                    if phase=='resume':command+=['--resume',str(gate/f'{stage}_initial_gpu{gpu}'/'step_000002.pt')]
                    else:
                        init=m['config']['initial_connector']['path'] if stage=='C0' else str(gate/f'C0_resume_gpu{gpu}'/'step_000003.pt')
                        command+=['--initialize',init,'--gate-stop-after','2']
                    commands.append((f'{stage}_{phase}_gpu{gpu}',node,command))
                if commands:run_commands(root,cluster,commands,gate/'logs',heartbeat=lambda:True,timeout=10800)
                if not all(engineering_complete(out,stage,phase,manifest_sha,1) for out in outputs):
                    raise RuntimeError('Local GPU gate incomplete or checkpoint mismatch')
        write(gate/'completed.json',dict(passed=True,time=time.time(),gpus=8,stages=['C0','A'],
            coverage='8 independent short+long original-area samples; each stage updates2 then strict resumes1',manifest_sha256=sha(manifest)))
        update('completed');notify(root,a.thread,'local_gpu_gate_completed',dict(report=str(gate/'completed.json')))
    except BaseException as e:
        update('stopped' if (root/'STOP').exists() else 'failed',error=repr(e))
        if not (root/'STOP').exists():notify(root,a.thread,'local_gpu_gate_failed',dict(error=repr(e),logs=str(gate/'logs')))
        raise
if __name__=='__main__':main()
