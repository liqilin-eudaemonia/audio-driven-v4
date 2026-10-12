"""Paired fixed-noise loss and per-loss Connector gradient diagnostics."""
import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
import time


def gradient_summary(named_parameters, video_grads, audio_grads):
    import torch
    result={}
    for branch in ('video','audio'):
        vg=[];ag=[]
        for (name,p),v,a in zip(named_parameters,video_grads,audio_grads):
            if name.startswith(branch+'.'):
                vg.append(torch.zeros_like(p,device='cpu') if v is None else v.detach().float().cpu())
                ag.append(torch.zeros_like(p,device='cpu') if a is None else a.detach().float().cpu())
        vv=sum(float(v.square().sum()) for v in vg);aa=sum(float(a.square().sum()) for a in ag)
        dot=sum(float((v*a).sum()) for v,a in zip(vg,ag));count=sum(v.numel() for v in vg)
        result[branch]=dict(parameters=count,video_loss_grad_norm=vv**.5,audio_loss_grad_norm=aa**.5,
            video_loss_grad_rms=(vv/max(count,1))**.5,audio_loss_grad_rms=(aa/max(count,1))**.5,
            cosine=dot/(vv*aa)**.5 if vv and aa else None,
            combined_grad_norm=max(0,vv+aa+2*dot)**.5,
            video_weight2_grad_norm=max(0,4*vv+aa+4*dot)**.5)
    return result


def match_tokens(hidden,reference):
    """Length-match wrong-sample controls to separate content from token count."""
    import torch.nn.functional as F
    return {k:(v if v.shape[1]==reference[k].shape[1] else
               F.interpolate(v.transpose(1,2).float(),size=reference[k].shape[1],mode='linear',align_corners=False).transpose(1,2).to(v.dtype))
            for k,v in hidden.items()}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--job',type=Path,required=True);parser.add_argument('--gpu',type=int,required=True)
    args=parser.parse_args();plan=json.loads((args.root/'plan.json').read_text());source=Path(plan['source'])
    sys.path[:0]=[str(source/'src'),str(source/'scripts')]
    from pipeline3_staged_worker import probe
    if any(g['pids'] or g['memory_mib']>=1000 or g['util']>5 for g in probe([args.gpu])['gpus']):raise RuntimeError('GPU occupied')
    os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu),OMP_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false')
    import torch
    from omni_av.ltx_runtime import with_text
    from omni_av_p2.config import Config
    from omni_av_p2.ltx import load_renderer,append_conditions,av_loss
    from omni_av_p2.connector import SharedCondition
    from omni_av_p3_staged.data import load_training_sample,load_readouts
    from omni_av_p3_staged.model import RenderTraining,GenerationConnector
    from omni_av_p3_staged.checkpoint import checked
    from omni_av_p3_staged.orchestration import read,write,sha
    from omni_av_p21.residual_performance import configure_grouped_checkpointing
    torch.set_num_threads(2);job=read(args.job);out=Path(job['output']);out.mkdir(parents=True,exist_ok=True)
    if (out/'completed.json').exists():
        if read(out/'completed.json')['job']!=job:raise ValueError('Diagnostic identity changed')
        return
    if sha(job['checkpoint'])!=job['checkpoint_sha256'] or sha(plan['manifest'])!=plan['manifest_sha256']:raise ValueError('Source hash changed')
    manifest=read(plan['manifest']);entry=manifest['dev'][job['index']];wrong_entry=manifest['dev'][job['wrong_index']]
    seal_sha=sha(Path(plan['cache'])/'inputs/complete.json')
    raw_hidden,text,av=load_training_sample(entry,plan['cache'],plan['manifest_sha256'],manifest['config']['data_root'],'pred25','cuda:0')
    allhidden=load_readouts(plan['cache'],entry,plan['manifest_sha256'])
    wrong=load_readouts(plan['cache'],wrong_entry,plan['manifest_sha256'])['pred25']
    wrong_shapes={k:list(v.shape) for k,v in wrong.items()}
    wrong=match_tokens({k:v.cuda() for k,v in wrong.items()},raw_hidden)
    hidden={'pred25':raw_hidden,'oracle':{k:v.cuda() for k,v in allhidden['oracle'].items()},'wrong':wrong}
    renderer,strategy,_=load_renderer(Config());configure_grouped_checkpointing(renderer,1)
    model=RenderTraining(renderer,strategy,'C0',GenerationConnector().cuda())
    cp=checked(job['checkpoint']);model.load_state_dict(cp['model'],strict=True);del cp
    parameters=list(model.connector.named_parameters());values=[p for _,p in parameters]
    before={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    class Sigma:
        def __init__(self,value):self.value=value
        def sample_for(self,x):return torch.full((x.shape[0],),self.value,device=x.device,dtype=x.dtype)
    rows=[];begin=time.time()
    for sigma in job['sigmas']:
        if (args.root/'STOP').exists():raise InterruptedError('Diagnostic STOP')
        torch.manual_seed(job['seed'])
        prepared=strategy.prepare_training_inputs(with_text(av,text),Sigma(sigma))
        for mode in ('text','zero','wrong','oracle','pred25'):
            write(out/'progress.json',dict(sigma=sigma,mode=mode,phase='forward',completed=len(rows),time=time.time()))
            need_grad=mode=='pred25';model.train(need_grad)
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
            with torch.set_grad_enabled(need_grad),torch.autocast('cuda',dtype=torch.bfloat16):
                conditioned=prepared;condition_stats={}
                if mode!='text':
                    condition=model.connector(hidden['pred25'] if mode=='zero' else hidden[mode])
                    if mode=='zero':condition=SharedCondition(torch.zeros_like(condition.video),torch.zeros_like(condition.audio),condition.video_valid,condition.audio_valid)
                    conditioned=append_conditions(prepared,condition)
                    for branch in ('video','audio'):
                        v=getattr(condition,branch);t=getattr(prepared,branch).context
                        condition_stats[branch]=dict(tokens=v.shape[1],rms=float(v.detach().float().square().mean().sqrt()),
                            text_tokens=t.shape[1],text_rms=float(t.detach().float().square().mean().sqrt()))
                loss,parts=av_loss(renderer,strategy,conditioned)
                if not torch.isfinite(loss):raise ValueError('Nonfinite diagnostic loss')
                row=dict(sigma=sigma,mode=mode,loss=float(loss.detach()),video=float(parts['video_loss'].detach()),audio=float(parts['audio_loss'].detach()),condition_stats=condition_stats)
                if need_grad:
                    write(out/'progress.json',dict(sigma=sigma,mode=mode,phase='video_backward',completed=len(rows),time=time.time()))
                    vg=torch.autograd.grad(parts['video_loss'],values,retain_graph=True,allow_unused=True)
                    vg=[None if v is None else v.detach().cpu() for v in vg]
                    write(out/'progress.json',dict(sigma=sigma,mode=mode,phase='audio_backward',completed=len(rows),time=time.time()))
                    ag=torch.autograd.grad(parts['audio_loss'],values,allow_unused=True)
                    row['gradients']=gradient_summary(parameters,vg,ag)
                    del vg,ag
            torch.cuda.synchronize();row.update(seconds=time.perf_counter()-started,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
            rows.append(row);write(out/'partial.json',dict(job=job,rows=rows));print(json.dumps(row),flush=True)
            del loss,parts,conditioned
            if mode!='text':del condition
        del prepared
    if any(not torch.equal(before[k],v.detach().cpu()) for k,v in model.state_dict().items()):raise RuntimeError('Diagnostic modified weights')
    if any(p.grad is not None for p in renderer.parameters()):raise RuntimeError('Unexpected frozen renderer gradients')
    write(out/'completed.json',dict(job=job,sample_id=entry['sample_id'],wrong_sample_id=wrong_entry['sample_id'],
        wrong_original_shapes=wrong_shapes,wrong_length_matched=True,input_integrity_sha256=seal_sha,rows=rows,
        weights_unchanged=True,seconds=time.time()-begin,note='Teacher-forced noised target latent diagnostics; not generation metrics. Fixed GT reply/timing.'))


if __name__=='__main__':main()
