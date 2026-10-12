"""Paired C0 optimizer-step benchmark on an exclusive GPU, without DDP."""
import argparse
from contextlib import nullcontext
from dataclasses import replace
from functools import partial
import json
import os
from pathlib import Path
import statistics
import sys
import time
from types import MethodType


def checkpoint_blocks(self,video,audio,perturbations):
    import torch
    from torch.utils.checkpoint import checkpoint
    from omni_av_p21.residual_performance import tensor_block
    vt=replace(video,x=None) if video is not None else None
    at=replace(audio,x=None) if audio is not None else None
    vx=video.x if video is not None else None;ax=audio.x if audio is not None else None
    cfg=self._benchmark_config;size=cfg.get('group',1)
    def group(vx,ax,*,blocks):
        for block in blocks:
            vx,ax=tensor_block(vx,ax,block=block,video_template=vt,audio_template=at,perturbations=perturbations)
        return vx,ax
    for index,start in enumerate(range(0,len(self.transformer_blocks),size)):
        call=partial(group,blocks=tuple(self.transformer_blocks[start:start+size]))
        active=self.training and torch.is_grad_enabled() and index%cfg.get('checkpoint_every',1)==0
        with torch.autograd.graph.save_on_cpu(pin_memory=True) if active and cfg.get('offload',False) else nullcontext():
            vx,ax=checkpoint(call,vx,ax,use_reentrant=False) if active else call(vx,ax)
    return replace(video,x=vx) if video is not None else None,replace(audio,x=ax) if audio is not None else None


def configure(renderer,cfg):
    from omni_av_p21.residual_performance import configure_grouped_checkpointing
    from ltx_core.model.transformer.attention import Attention,PytorchAttention
    from torch.nn.attention import SDPBackend,sdpa_kernel
    configure_grouped_checkpointing(renderer,1)
    if cfg.get('custom_checkpoint'):
        renderer._benchmark_config=cfg;renderer._process_transformer_blocks=MethodType(checkpoint_blocks,renderer)
    backend=cfg.get('attention','auto')
    class SelectedAttention:
        def __init__(self):self.native=PytorchAttention()
        def __call__(self,q,k,v,heads,mask=None):
            selected={'flash':SDPBackend.FLASH_ATTENTION,'cudnn':SDPBackend.CUDNN_ATTENTION}
            with sdpa_kernel(selected[backend]) if backend!='auto' and mask is None else nullcontext():
                return self.native(q,k,v,heads,mask)
    for m in renderer.modules():
        if isinstance(m,Attention):m.attention_function=SelectedAttention()


def relative_error(actual,reference):
    numerator=sum(float((a.float()-b.float()).square().sum()) for a,b in zip(actual,reference))
    denominator=sum(float(b.float().square().sum()) for b in reference)
    return (numerator/max(denominator,1e-30))**.5


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--job',type=Path,required=True);p.add_argument('--gpu',type=int,required=True)
    a=p.parse_args();plan=json.loads((a.root/'plan.json').read_text());source=Path(plan['source']);sys.path[:0]=[str(source/'src'),str(source/'scripts')]
    from pipeline3_staged_worker import probe
    if any(g['pids'] or g['memory_mib']>=1000 or g['util']>5 for g in probe([a.gpu])['gpus']):raise RuntimeError('Exclusive benchmark requires an idle GPU')
    os.environ.update(CUDA_VISIBLE_DEVICES=str(a.gpu),OMP_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    import torch
    from omni_av.ltx_runtime import with_text
    from omni_av_p2.config import Config
    from omni_av_p2.ltx import load_renderer
    from omni_av_p3_staged.model import RenderTraining,GenerationConnector
    from omni_av_p3_staged.checkpoint import checked
    from omni_av_p3_staged.data import load_training_sample
    from omni_av_p3_staged.orchestration import read,write,sha
    torch.set_num_threads(2);job=read(a.job);out=Path(job['output']);out.mkdir(parents=True,exist_ok=True)
    if (out/'completed.json').exists():
        if read(out/'completed.json')['job']!=job:raise ValueError('Benchmark job changed')
        return
    if sha(plan['checkpoint'])!=plan['checkpoint_sha256'] or sha(plan['manifest'])!=plan['manifest_sha256']:raise ValueError('Benchmark source changed')
    manifest=read(plan['manifest']);cp=checked(plan['checkpoint']);loaded=time.perf_counter()
    batches=[load_training_sample(manifest[s['split']][s['index']],plan['cache'],plan['manifest_sha256'],manifest['config']['data_root'],'pred25','cuda:0') for s in job['samples']]
    load_seconds=time.perf_counter()-loaded
    renderer,strategy,sampler=load_renderer(Config());model=RenderTraining(renderer,strategy,'C0',GenerationConnector().cuda())
    frozen=[(p,p._version) for p in renderer.parameters()];reference=None;results=[]
    for label,cfg in [('baseline',{}),('candidate',job['config'])]:
        configure(renderer,cfg);model.load_state_dict(cp['model'],strict=True);model.train()
        optimizer=torch.optim.AdamW(model.optimizer_groups(3e-5,0.),fused=True);optimizer.load_state_dict(cp['optimizer'])
        rows=[];gradients=None;updated=None
        try:
            for iteration in range(3):
                if (a.root/'STOP').exists():raise InterruptedError('Diagnostic STOP')
                optimizer.zero_grad(set_to_none=True);torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter();losses=[]
                for micro,(hidden,text,av) in enumerate(batches):
                    write(out/'progress.json',dict(configuration=label,iteration=iteration,micro=micro,time=time.time()))
                    torch.manual_seed(job['seed']+iteration*len(batches)+micro)
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        vals=model(hidden,strategy.prepare_training_inputs(with_text(av,text),sampler))
                        (vals[0]/len(batches)).backward()
                    losses.append([float(v.detach()) for v in vals]);del vals
                torch.cuda.synchronize();backward_end=time.perf_counter()
                if iteration==0:gradients=[p.grad.detach().cpu().clone() for p in model.parameters()]
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step();torch.cuda.synchronize()
                # Exclude the one-off correctness copy from timing; only iterations 1/2 are reported as steady-state.
                row=dict(iteration=iteration,seconds=time.perf_counter()-start,forward_backward_seconds=backward_end-start,
                         peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,losses=losses,grad_norm=float(norm))
                if iteration==0:updated=[p.detach().cpu().clone() for p in model.parameters()]
                rows.append(row);write(out/'partial.json',dict(configuration=label,rows=rows));print(json.dumps(row),flush=True)
            result=dict(configuration=label,status='ok',rows=rows,mean_seconds=statistics.mean(r['seconds'] for r in rows[1:]),
                        max_peak_allocated_gib=max(r['peak_allocated_gib'] for r in rows))
            if label=='baseline':reference=dict(gradients=gradients,updated=updated,rows=rows)
            else:
                loss_ref=[v[0] for v in reference['rows'][0]['losses']];loss_new=[v[0] for v in rows[0]['losses']]
                result.update(gradient_relative_l2=relative_error(gradients,reference['gradients']),
                    updated_weights_relative_l2=relative_error(updated,reference['updated']),
                    max_loss_relative_error=max(abs(x-y)/max(abs(y),1e-12) for x,y in zip(loss_new,loss_ref)))
                result['screening_passed']=result['gradient_relative_l2']<.02 and result['max_loss_relative_error']<.002
            results.append(result)
            if job.get('profile'):
                optimizer.zero_grad(set_to_none=True);hidden,text,av=batches[1];torch.manual_seed(job['seed'])
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as profiler:
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        values=model(hidden,strategy.prepare_training_inputs(with_text(av,text),sampler));values[0].backward()
                    torch.cuda.synchronize()
                events=profiler.key_averages()
                result['attention_operators']=[dict(name=e.key,calls=e.count,device_time_us=e.device_time_total) for e in events if 'attention' in e.key.lower()]
                result['top_device_operators']=[dict(name=e.key,calls=e.count,self_device_time_us=e.self_device_time_total) for e in sorted(events,key=lambda e:e.self_device_time_total,reverse=True)[:20]]
                del values,profiler
        except (RuntimeError,NotImplementedError) as e:
            results.append(dict(configuration=label,status='failed',error=str(e)[:2000],rows=rows))
            if label=='baseline':break
        finally:
            model.zero_grad(set_to_none=True);del optimizer;torch.cuda.empty_cache()
    if any(p.grad is not None or p._version!=version for p,version in frozen):raise RuntimeError('Frozen renderer changed')
    summary=dict(job=job,results=results,data_load_seconds=load_seconds,checkpoint_sha256=plan['checkpoint_sha256'],
                 accumulation=len(batches),ddp=False,parameters_updated_only_in_memory=True,
                 note='Paired single-GPU C0 updates, preloaded inputs, one warmup plus two timed updates. Speed excludes DDP and input loading. Requires 24-GPU verification before production adoption.')
    if len(results)==2 and all(r['status']=='ok' for r in results):summary['speedup']=results[0]['mean_seconds']/results[1]['mean_seconds']
    write(out/'completed.json',summary)


if __name__=='__main__':main()
