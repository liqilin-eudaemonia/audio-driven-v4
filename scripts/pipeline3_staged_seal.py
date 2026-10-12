"""Parallel full readout audit, preserving the existing cache seal contract."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))

def check_batch(args):
    root,digest,entries=args
    from omni_av_p3_staged.data import load_readouts
    for entry in entries:load_readouts(root,entry,digest)
    return len(entries)

def seal(manifest,root,workers=8,control_root=None):
    import torch
    from omni_av_p3_staged.data import read
    from omni_av_p3.distributed import sha
    from omni_av_p2.checkpoint import atomic_json_new
    from omni_av_p3_staged.orchestration import write
    torch.set_num_threads(1)
    m=read(manifest);digest=sha(manifest)
    entries=[dict(sample_id=e['sample_id'],split=e['split']) for e in m['train']+m['dev']]
    if len(m['train'])!=91009 or len(m['dev'])!=48 or len({(e['split'],e['sample_id']) for e in entries})!=91057:
        raise ValueError('Full manifest coverage mismatch')
    root=Path(root);start=time.time();done=0
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('fork')) as pool:
        futures=[pool.submit(check_batch,(root,digest,entries[i:i+256])) for i in range(0,len(entries),256)]
        for f in as_completed(futures):
            done+=f.result()
            if control_root and (Path(control_root)/'STOP').exists():raise InterruptedError('User STOP')
            value=dict(completed=done,validated=done,total=len(entries),workers=workers,elapsed_seconds=time.time()-start,time=time.time())
            write(root/'seal_progress.json',value);print(value,flush=True)
    if done!=len(entries):raise RuntimeError('Incomplete audit')
    atomic_json_new(root/'complete.json',dict(manifest_sha256=digest,count=done,train=len(m['train']),dev=len(m['dev']),modes=['oracle','pred25']))
    print('Full cache validated; parallel audit passed',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=8);p.add_argument('--control-root',type=Path)
    a=p.parse_args()
    if not 1<=a.workers<=32:raise ValueError('workers out of range')
    seal(a.manifest,a.output,a.workers,a.control_root)
