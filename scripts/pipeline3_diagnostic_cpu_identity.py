"""Run CPU identity scoring as each mode completes, alongside GPU generation."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);root=parser.parse_args().root
    read=lambda p:json.loads(p.read_text())
    plan=read(root/'plan.json');sys.path.insert(0,str(Path(plan['source'])/'src'))
    from omni_av_p3_staged.orchestration import write,sha
    cfg=read(Path(read(root/'metric_plan.json')['metrics_config']))
    lock=(root/'cpu_identity.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    jobs=[read(p) for p in sorted((root/'metric_jobs').glob('*_identity.json'))]
    mode_count=len(jobs)
    write(root/'cpu_identity_state.json',dict(pid=os.getpid(),status='waiting',time=time.time()))
    try:
        while jobs:
            if (root/'STOP').exists():raise InterruptedError('Diagnostic STOP')
            ready=[]
            for job in jobs:
                records=[r for r in plan['records'] if r['mode']==job['mode']]
                if all((Path(r['output'])/'completed.json').exists() for r in records):ready.append(job)
            for job in ready:
                out=Path(job['output']);out.mkdir(parents=True,exist_ok=True)
                identity=dict(job=job,input_sha256=sha(job['manifest']))
                ident=out/'identity_source.json'
                if ident.exists() and read(ident)!=identity:raise ValueError('Identity metric source changed')
                write(ident,identity)
                if not (out/'completed.json').exists():
                    write(root/'cpu_identity_state.json',dict(pid=os.getpid(),status='scoring',mode=job['mode'],time=time.time()))
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',
                             PYTHONPATH=cfg['benchmark_root'],TORCH_HOME=cfg['torch_home'])
                    command=[cfg['id_python'],str(Path(cfg['legacy_trainer'])/'scripts/compute_arcface_id.py'),
                             '--manifest',job['manifest'],'--output',str(out/'identity.json'),'--model-root',cfg['id_model_root'],
                             '--model-name',cfg['id_model_name'],'--sample-frames',str(cfg['id_frames']),'--cpu']
                    subprocess.run(command,env=env,cwd=cfg['benchmark_root'],check=True,timeout=6500)
                    write(out/'visual.json',{});write(out/'completed.json',dict(identity,visual={},identity=read(out/'identity.json')))
                jobs.remove(job)
            if jobs:time.sleep(10)
        write(root/'cpu_identity_completed.json',dict(time=time.time(),modes=mode_count))
        write(root/'cpu_identity_state.json',dict(pid=os.getpid(),status='completed',time=time.time()))
    except BaseException as e:
        write(root/'cpu_identity_state.json',dict(pid=os.getpid(),status='failed',error=repr(e),time=time.time()));raise


if __name__=='__main__':main()
