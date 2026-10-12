"""Frozen step17000 Oracle/Pred25 readouts; independent resumable shards."""
import argparse
import fcntl
import json
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=24)
    p.add_argument('--ids',type=Path,help='Engineering subset IDs; does not publish full seal')
    p.add_argument('--worker-count',type=int,default=1)
    p.add_argument('--worker-index',type=int,default=0)
    p.add_argument('--control-root',type=Path)
    p.add_argument('--progress-dir',type=Path)
    p.add_argument('--device',default='cuda:0');p.add_argument('--seal',action='store_true')
    a=p.parse_args()
    from omni_av_p3_staged.data import read,cache_path,cache_identity,load_readouts,condition_seed
    from omni_av_p3.distributed import sha
    from omni_av_p2.checkpoint import atomic_save,atomic_json_new
    from omni_av_p3_staged.orchestration import write,stopped
    manifest=read(a.manifest);digest=sha(a.manifest);entries=manifest['train']+manifest['dev']
    if not 0<=a.shard<a.shards:raise ValueError('Invalid shard')
    if a.ids:
        ids=set(read(a.ids));entries=[e for e in entries if e['sample_id'] in ids]
        if len(entries)!=len(ids):raise ValueError('Engineering IDs absent')
    if a.seal and a.ids:raise ValueError('Subset cannot seal full cache')
    if not 0<=a.worker_index<a.worker_count:raise ValueError('Worker index out of range')
    a.output.mkdir(parents=True,exist_ok=True)
    progress_dir=a.progress_dir or a.output/'progress'
    progress_file=progress_dir/('seal.json' if a.seal else f'shard{a.shard}_worker{a.worker_index}.json')
    def progress(count,total):write(progress_file,dict(completed=count,total=total,time=time.time()))
    # A single owner per shard, released on process death. Immutable manifest key.
    with (a.output/f'{digest}.shard{a.shard}.worker{a.worker_index}.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if a.seal:
            progress(0,len(entries))
            for count,e in enumerate(entries,1):
                if a.control_root and stopped(a.control_root):raise InterruptedError('User STOP')
                load_readouts(a.output,e,digest)
                if count%100==0 or count==len(entries):progress(count,len(entries))
            atomic_json_new(a.output/'complete.json',dict(manifest_sha256=digest,count=len(entries),
                train=len(manifest['train']),dev=len(manifest['dev']),modes=['oracle','pred25']))
            print('Full cache validated; no training launched');return
        import torch
        from omni_av_p3.model import Config,Layout
        from omni_av_p3.data import load_native
        from omni_av_p3.accelerated import FastPlanner,execution_mode
        from omni_av_p3.diagnostics import restore
        cfg=Config(**manifest['native_contract']['config']);cp=manifest['config']['planner']
        if sha(cp['path'])!=cp['sha256']:raise ValueError('Planner checkpoint changed')
        torch.set_num_threads(2)
        assigned=entries[a.shard::a.shards][a.worker_index::a.worker_count]
        pending=[e for e in assigned if not cache_path(a.output,e).with_suffix('.json').exists()]
        progress(len(assigned)-len(pending),len(assigned))
        if not pending:return
        runtime,_=load_native(cfg,a.device);model=FastPlanner(runtime.thinker,cfg,'lora')
        model.connectors.to(a.device);model.decoders.to(a.device)
        model.mask_token.data=model.mask_token.data.to(a.device)
        model.modality_embedding.data=model.modality_embedding.data.to(a.device)
        saved=torch.load(cp['path'],map_location='cpu',weights_only=True);restore(model,saved);del saved
        model.requires_grad_(False);execution_mode(model,True);model.eval()
        completed=len(assigned)-len(pending)
        for e in pending:
            if a.control_root and (a.control_root/'STOP').exists():return
            path=cache_path(a.output,e);path.parent.mkdir(parents=True,exist_ok=True)
            if path.with_suffix('.json').exists():
                load_readouts(a.output,e,digest);continue
            # Recover a complete payload whose marker publication was interrupted.
            identity=cache_identity(digest,e)
            if path.exists():
                payload=torch.load(path,map_location='cpu',weights_only=True)
                if payload['identity']!=identity or set(payload['readouts'])!={'oracle','pred25'}:
                    raise ValueError('Unpublished cache identity mismatch')
            else:
                n=e['native']
                if sha(n['path'])!=n['sha256']:raise ValueError('Native cache changed')
                row=torch.load(n['path'],map_location='cpu',weights_only=True)
                if row['sample_id']!=e['sample_id'] or row['split']!=e['split']:raise ValueError('Native identity mismatch')
                layout=Layout(**row['layout']).to(a.device);values={}
                with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                    for mode in ('oracle','pred25'):
                        if mode=='oracle':features=row['targets'].to(a.device)
                        else:
                            features,remaining=model.sample(layout,condition_seed(manifest['config']['seed'],e['sample_id']),steps=25)
                            if remaining[-1]!=0:raise ValueError('Incomplete planner sampling')
                        hidden=model.hidden(layout,torch.zeros_like(layout.valid),features)
                        values[mode]={name:hidden[:,layout.modalities[0]==i].detach().cpu()
                                      for i,name in enumerate(('video','audio'))}
                        if any(not torch.isfinite(v).all() for v in values[mode].values()):raise ValueError('Nonfinite hidden')
                payload=dict(identity=identity,readouts=values)
                atomic_save(path,payload);del row,features,hidden,layout,values
                model._geometry.clear()
            atomic_json_new(path.with_suffix('.json'),dict(identity=identity,sha256=sha(path)))
            completed+=1;progress(completed,len(assigned))
            print(json.dumps(dict(sample_id=e['sample_id'],split=e['split'],status='complete')),flush=True)
if __name__=='__main__':main()
