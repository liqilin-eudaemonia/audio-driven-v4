"""Re-decode identical P0299A audio latents to isolate BF16 decoder/vocoder effects."""
import argparse
import json
import os
from pathlib import Path
import sys
import time


def main():
    p=argparse.ArgumentParser();p.add_argument('--job',type=Path,required=True);p.add_argument('--gpu',type=int,required=True)
    a=p.parse_args();job=json.loads(a.job.read_text());source=Path(job['source']);sys.path[:0]=[str(source/'src'),str(source/'scripts')]
    from pipeline3_staged_worker import probe
    if any(g['pids'] or g['memory_mib']>=1000 or g['util']>5 for g in probe([a.gpu])['gpus']):raise RuntimeError('GPU occupied')
    os.environ.update(CUDA_VISIBLE_DEVICES=str(a.gpu),OMP_NUM_THREADS='2')
    import torch
    import soundfile as sf
    from ltx_trainer.model_loader import load_audio_vae_decoder,load_vocoder
    from omni_av_p2.config import Config
    from omni_av_p2.data import unload
    from omni_av_p3_staged.orchestration import sha,write
    torch.set_num_threads(2);out=Path(job['output']);out.mkdir(parents=True,exist_ok=True)
    if (out/'completed.json').exists():
        assert json.loads((out/'completed.json').read_text())['job']==job
        return
    if sha(job['latents'])!=job['latents_sha256']:raise ValueError('Audio latent source changed')
    audio=torch.load(job['latents'],map_location='cpu',weights_only=True)['audio'];cfg=Config();rows=[]
    for name,vd,wd in [('bf16',torch.bfloat16,torch.bfloat16),('fp32_vocoder',torch.bfloat16,torch.float32),('fp32_both',torch.float32,torch.float32)]:
        if (Path(job['root'])/'STOP').exists():raise InterruptedError('Diagnostic STOP')
        started=time.time();decoder=load_audio_vae_decoder(cfg.ltx_path,'cuda:0',vd).requires_grad_(False).eval()
        with torch.no_grad():mel=decoder(audio.to('cuda:0',vd))
        del decoder;unload();vocoder=load_vocoder(cfg.ltx_path,'cuda:0',wd).requires_grad_(False).eval()
        with torch.no_grad():wave=vocoder(mel.to(wd)).squeeze(0).float().cpu()
        rate=int(vocoder.output_sampling_rate);del vocoder,mel;unload()
        if not torch.isfinite(wave).all():raise FloatingPointError(name)
        sf.write(str(out/(name+'.wav')),wave.numpy().T,rate,subtype='FLOAT')
        diff=wave.diff(dim=-1);flat=wave.flatten()
        rows.append(dict(name=name,seconds=time.time()-started,rate=rate,shape=list(wave.shape),peak=float(flat.abs().max()),
            rms=float(flat.square().mean().sqrt()),max_adjacent_jump=float(diff.abs().max()),
            adjacent_jump_p999=float(torch.quantile(diff.abs().flatten(),.999)),over_one_fraction=float((flat.abs()>1).float().mean())))
        write(out/'progress.json',dict(completed=rows,time=time.time()))
    write(out/'completed.json',dict(job=job,rows=rows,time=time.time(),note='Same saved audio latents; no denoising re-run. Signal statistics alone do not measure perceived noise.'))


if __name__=='__main__':main()
