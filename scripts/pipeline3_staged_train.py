"""torchrun entry for C0 or A; no automatic transition and no stage B."""
import argparse
import json
import os
import sys
import time
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--stage',choices=['C0','A'],required=True);p.add_argument('--output',type=Path,required=True)
    init=p.add_mutually_exclusive_group(required=True)
    init.add_argument('--initialize',type=Path);init.add_argument('--resume',type=Path)
    p.add_argument('--quality-gate',type=Path)
    p.add_argument('--transition-authorization',type=Path)
    p.add_argument('--engineering-gate',action='store_true')
    p.add_argument('--gate-steps',type=int,default=2)
    p.add_argument('--gate-indices',type=Path)
    p.add_argument('--gate-stop-after',type=int)
    p.add_argument('--control-root',type=Path,required=True)
    a=p.parse_args()
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.tensorboard import SummaryWriter
    from omni_av.ltx_runtime import with_text
    from omni_av_p2.config import Config
    from omni_av_p2.ltx import load_renderer
    from omni_av_p2.checkpoint import capture_rng,restore_rng,atomic_json_new
    from omni_av_p3.distributed import GlobalOrder,BatchAssignments,sha
    from omni_av_p2.distributed import parameter_digest
    from omni_av_p21.config import LoRAConfig
    from omni_av_p3_staged.model import RenderTraining,GenerationConnector
    from omni_av_p3_staged.data import read,condition_mode,load_training_sample
    from omni_av_p3_staged.checkpoint import initialize,resume,save
    from omni_av_p21.residual_performance import configure_grouped_checkpointing
    from pipeline21_residual_supervise import write
    m=read(a.manifest);digest=sha(a.manifest);cfg=m['config'];recipe=cfg['stages'][a.stage]
    world=int(os.environ['WORLD_SIZE']);rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK'])
    recipe=dict(recipe)
    accumulation=cfg['accumulation'];train_entries=m['train']
    if a.engineering_gate:
        if a.gate_steps<1 or a.gate_indices is None:raise ValueError('Gate needs positive target and explicit indices')
        indices=read(a.gate_indices)
        if not indices or any(type(i)!=int or not 0<=i<len(m['train']) for i in indices):raise ValueError('Invalid gate indices')
        train_entries=[m['train'][i] for i in indices]
        recipe['steps']=a.gate_steps;accumulation=3 if world==24 else 1
    # Only global rank zero may reject/create the shared output directory.
    # A slower rank must not mistake rank zero's new directory for an old run.
    if rank==0 and a.output.exists():raise FileExistsError(a.output)
    if (a.control_root/'STOP').exists():raise RuntimeError('Persistent user STOP; not starting')
    if not a.engineering_gate and (world!=cfg['world_size'] or world*accumulation!=cfg['global_batch']):
        raise ValueError('World/global batch mismatch; conversion not implemented')
    seal=read(a.cache/'complete.json') if not a.engineering_gate else None
    if not a.engineering_gate and (seal['manifest_sha256']!=digest or seal['train']!=91009 or seal['dev']!=48
            or seal['count']!=91057 or set(seal['modes'])!={'oracle','pred25'}):
        raise ValueError('Training requires complete full-data Oracle/Pred25 cache')
    input_seal=a.cache/'inputs/complete.json'
    if not a.engineering_gate:
        inputs=read(input_seal)
        from omni_av_p3_staged.integrity import VERSION as INPUT_VERSION
        expected={e['split']+':'+e['sample_id'] for e in m['train']+m['dev']}
        if inputs.get('version')!=INPUT_VERSION or inputs['manifest_sha256']!=digest or set(inputs['entries'])!=expected:
            raise ValueError('Full input-integrity seal mismatch')
    if a.initialize and a.stage=='C0' and sha(a.initialize)!=cfg['initial_connector']['sha256']:
        raise ValueError('C0 warm start differs from the selected Connector100')
    torch.set_num_threads(2);torch.cuda.set_device(local);device=torch.device('cuda',local)
    dist.init_process_group('nccl',timeout=timedelta(minutes=15),device_id=device)
    writer=None
    try:
        if rank==0:a.output.mkdir(parents=True,exist_ok=False)
        dist.barrier();torch.manual_seed(cfg['seed'])
        renderer,strategy,sampler=load_renderer(Config(),device=device)
        configure_grouped_checkpointing(renderer,1)
        raw=RenderTraining(renderer,strategy,a.stage,GenerationConnector().to(device),
            LoRAConfig(rank=cfg['lora_rank'],alpha=cfg['lora_alpha'],learning_rate=recipe['dit_lr']))
        optimizer=torch.optim.AdamW(raw.optimizer_groups(recipe['connector_lr'],recipe['dit_lr']),fused=True)
        identity=dict(manifest_sha256=digest,world_size=world,global_batch=world*accumulation,
            input_integrity_sha256=sha(input_seal) if not a.engineering_gate else None,
            accumulation=accumulation,stage=a.stage,recipe=recipe,targets=raw.targets,
            engineering_gate=a.engineering_gate,gate_indices_sha256=sha(a.gate_indices) if a.engineering_gate else None,
            code={str(p.relative_to(ROOT)):sha(p) for p in sorted([
                *ROOT.glob('src/omni_av_p3_staged/*.py'),ROOT/'scripts/pipeline3_staged_train.py',
                ROOT/'src/omni_av_p2/ltx.py',ROOT/'src/omni_av/ltx_runtime.py',
                ROOT/'src/omni_av_p21/lora.py',ROOT/'src/omni_av_p2/connector.py',
                ROOT/'src/omni_av_p21/residual_performance.py',ROOT/'src/omni_av_p3/distributed.py'])})
        start=0;resume_data=None
        if a.resume:
            transition=a.control_root/'checkpoint_transition.json'
            resume_data=resume(a.resume,raw,optimizer,identity,transition if transition.exists() else None);start=resume_data['step']
        else:initialize(a.initialize,raw,digest,a.quality_gate,engineering_gate=a.engineering_gate,
                        transition_authorization=a.transition_authorization)
        if start>recipe['steps']:raise ValueError('Checkpoint exceeds stage target')
        model=DDP(raw,device_ids=[local],broadcast_buffers=False,gradient_as_bucket_view=True)
        if cfg.get('performance',{}).get('gradient_communication')=='bf16':
            from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import bf16_compress_hook
            model.register_comm_hook(None,bf16_compress_hook)
        if resume_data is not None:restore_rng(resume_data['rank_rng'][rank]);del resume_data
        raw.train();order=GlobalOrder(len(train_entries),cfg['seed'])
        costs=[(((e['metadata']['target_frames']-1)//8+1)*(e['metadata']['target_height']//32)*(e['metadata']['target_width']//32))**2 for e in train_entries]
        assignments=BatchAssignments(order,costs,world,accumulation,True)
        frozen=[(p,p._version) for p in renderer.parameters() if not p.requires_grad]
        if rank==0:
            writer=SummaryWriter(str(a.output/'tensorboard'))
            atomic_json_new(a.output/'run.json',dict(identity=identity,start=start,initialize=str(a.initialize),resume=str(a.resume),
                transition_authorization_sha256=sha(a.transition_authorization) if a.transition_authorization else None,
                resume_checkpoint_sha256=sha(a.resume) if a.resume else None,
                transition_sha256=sha(a.control_root/'checkpoint_transition.json') if a.resume and (a.control_root/'checkpoint_transition.json').exists() else None))
        def checkpoint(step):
            states=[None]*world;dist.all_gather_object(states,capture_rng())
            hashes=[None]*world;dist.all_gather_object(hashes,parameter_digest(raw))
            if len(set(hashes))!=1:raise RuntimeError('Ranks diverged')
            if rank==0:save(a.output/f'step_{step:06d}.pt',raw,optimizer,step,identity,states)
            dist.barrier()
        def evaluate(step):
            raw.eval();rows=[]
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                for i in range(rank,len(m['dev']),world):
                    for mode in ('oracle','pred25'):
                        h,t,av=load_training_sample(m['dev'][i],a.cache,digest,cfg['data_root'],mode,device,allow_unsealed=a.engineering_gate)
                        torch.manual_seed(cfg['seed']+100000+i)
                        values=raw(h,strategy.prepare_training_inputs(with_text(av,t),sampler))
                        rows.append(dict(sample_id=m['dev'][i]['sample_id'],mode=mode,loss=[float(v) for v in values]))
            collected=[None]*world;dist.all_gather_object(collected,rows)
            if rank==0:
                allrows=[row for group in collected for row in group]
                atomic_json_new(a.output/f'dev_{step:06d}.json',dict(step=step,rows=allrows))
                for mode in ('oracle','pred25'):
                    vals=[row['loss'] for row in allrows if row['mode']==mode]
                    for k,name in enumerate(('av','video','audio')):writer.add_scalar(f'dev/{mode}/{name}',sum(v[k] for v in vals)/len(vals),step)
            raw.train()
        last=start;saved_step=None
        with (a.output/f'rank{rank}.jsonl').open('x') as log:
            for step in range(start+1,recipe['steps']+1):
                stop=torch.tensor(int((a.output/'STOP').exists() or (a.control_root/'STOP').exists()),device=device);dist.all_reduce(stop,op=dist.ReduceOp.MAX)
                if stop.item():break
                begin=time.monotonic();optimizer.zero_grad(set_to_none=True)
                # AV losses per condition mode: sum + count, synchronized across all ranks.
                sums=torch.zeros(2,4,device=device)
                for group in optimizer.param_groups:
                    if group['name']=='dit_text_lora':group['lr']=recipe['dit_lr']*min(1.,step/50)
                for micro in range(accumulation):
                    ordinal=assignments[step][micro][rank]
                    entry=train_entries[order[ordinal]];mode=condition_mode(cfg['seed'],ordinal)
                    write(a.output/f'progress_rank{rank}.json',dict(step=step,micro=micro,sample_id=entry['sample_id'],phase='load',time=time.time()))
                    h,t,av=load_training_sample(entry,a.cache,digest,cfg['data_root'],mode,device,allow_unsealed=a.engineering_gate)
                    torch.manual_seed(cfg['seed']+ordinal)
                    sync=nullcontext() if micro==accumulation-1 else model.no_sync()
                    with sync,torch.autocast('cuda',dtype=torch.bfloat16):
                        vals=model(h,strategy.prepare_training_inputs(with_text(av,t),sampler))
                        if not torch.isfinite(vals[0]):raise ValueError('Nonfinite AV loss')
                        (vals[0]/accumulation).backward()
                    k=int(mode=='pred25');sums[k,:3]+=torch.stack([v.detach().float() for v in vals]);sums[k,3]+=1
                    del vals,h,t,av
                    write(a.output/f'progress_rank{rank}.json',dict(step=step,micro=micro,phase='backward_complete',time=time.time()))
                norm=torch.nn.utils.clip_grad_norm_(raw.parameters(),1.,error_if_nonfinite=True)
                optimizer.step();raw.assert_ownership()
                if any(p.grad is not None or p._version!=v for p,v in frozen):raise RuntimeError('Frozen DiT changed')
                dist.all_reduce(sums);torch.cuda.synchronize()
                row=dict(step=step,stage=a.stage,seconds=time.monotonic()-begin,grad_norm=float(norm),
                    modes={mode:dict(count=int(sums[k,3]),loss=(sums[k,:3]/sums[k,3].clamp_min(1)).tolist())
                           for k,mode in enumerate(('oracle','pred25'))})
                log.write(json.dumps(row)+'\n');log.flush();last=step
                if rank==0:
                    print(json.dumps(row),flush=True)
                    for mode,v in row['modes'].items():
                        for k,name in enumerate(('av','video','audio')):writer.add_scalar(f'train/{mode}/{name}',v['loss'][k],step)
                    writer.add_scalar('train/grad_norm',float(norm),step);writer.add_scalar('speed/seconds',row['seconds'],step)
                if step%cfg['save_every']==0 or step==recipe['steps'] or (a.engineering_gate and a.gate_stop_after==step):
                    checkpoint(step);saved_step=step
                    if not a.engineering_gate:evaluate(step)
                if a.engineering_gate and a.gate_stop_after==step:break
            if last!=saved_step:checkpoint(last)
            # Recover a failure after the final save but before Dev scoring or
            # completion publication, without repeating any optimizer update.
            if start==recipe['steps'] and not a.engineering_gate:evaluate(last)
        if rank==0:atomic_json_new(a.output/'completed.json',dict(stage=a.stage,step=last,stopped=last<recipe['steps'],engineering_gate=a.engineering_gate,next_stage_launched=False))
    finally:
        if writer:writer.close()
        dist.destroy_process_group()
if __name__=='__main__':main()
