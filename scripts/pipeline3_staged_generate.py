"""Paired C0/A quality checks with native CFG; first-frame-only generation."""
import argparse,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))


def main():
 p=argparse.ArgumentParser();p.add_argument('--job',type=Path,required=True);p.add_argument('--gpu',type=int,required=True);a=p.parse_args()
 from pipeline3_staged_worker import probe
 if any(g['pids'] or g['memory_mib']>=1000 or g['util']>5 for g in probe([a.gpu])['gpus']):raise RuntimeError('GPU occupied')
 os.environ.update(CUDA_VISIBLE_DEVICES=str(a.gpu),OMP_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false')
 import torch
 from PIL import Image
 from omni_av.data import Observation,TextPlan
 from omni_av.ltx_runtime import with_text
 from omni_av_p2.config import Config
 from omni_av_p2.ltx import load_renderer,append_conditions,unpack_latents
 from omni_av_p2.inference import reference_latents
 from omni_av_p2.data import unload
 from omni_av_p2.checkpoint import atomic_save
 from omni_av_p3_staged.data import load_readouts
 from omni_av_p3_staged.integrity import load_record,verified_load
 from omni_av_p3_staged.model import RenderTraining,GenerationConnector
 from omni_av_p3_staged.checkpoint import checked
 from omni_av_p3_staged.sampling import guided_sample
 from omni_av_p3_staged.media import decode_and_save
 from omni_av_p3_staged.orchestration import read,write,sha
 from omni_av_p21.config import LoRAConfig
 torch.set_num_threads(2);job=read(a.job);out=Path(job['output']);out.mkdir(parents=True,exist_ok=True)
 if not job.get('engineering_gate',False) and job.get('input_integrity_sha256')!=sha(Path(job['cache'])/'inputs/complete.json'):
  raise ValueError('Generation input-integrity seal mismatch')
 if (out/'completed.json').exists():
  done=read(out/'completed.json')
  if done['job']!=job or done['video_sha256']!=sha(out/'generated.mp4'):raise ValueError('Completed identity changed')
  return
 m=read(job['manifest']);cfg=Config();e=m['dev'][job['index']];meta=e['metadata']
 geom=dict(frames=meta['target_frames'],height=meta['target_height'],width=meta['target_width'],fps=meta['fps'],duration_s=meta['target_frames']/meta['fps'])
 if sha(job['manifest'])!=job['manifest_sha256']:raise ValueError('Manifest changed')
 with Image.open(e['example']['observation']['listener_image']) as im:
  if im.size!=(geom['width'],geom['height']):raise ValueError('No crop/resize allowed')
 latent=out/'sampled_latents.pt'
 if not latent.exists():
  bound=load_record(Path(job['cache'])/'inputs',e,job['manifest_sha256'],m['config']['data_root'],
                    allow_unsealed=job.get('engineering_gate',False))
  text=verified_load(e['text'],bound['files'],map_location='cuda:0')
  if text['prompt']!=TextPlan(**e['example']['text_plan']).render_prompt():raise ValueError('Text changed')
  if sha(job['negative_text'])!=job['negative_sha256']:raise ValueError('Negative text changed')
  negative=torch.load(job['negative_text'],map_location='cuda:0',weights_only=True)
  av=reference_latents(cfg,Observation(**e['example']['observation']),geom,device='cuda:0')
  renderer,strategy,sampler=load_renderer(cfg);renderer.eval();raw=None;u=None
  if job['mode']!='text':
   cp=checked(job['checkpoint'])
   if sha(job['checkpoint'])!=job['checkpoint_sha256']:raise ValueError('Checkpoint changed')
   stage=cp.get('stage','C0');raw=RenderTraining(renderer,strategy,stage,GenerationConnector().cuda(),LoRAConfig(rank=8,alpha=8))
   if 'model' in cp:raw.load_state_dict(cp['model'],strict=True)
   else:raw.connector.load_state_dict(cp['connector'],strict=True)
   raw.requires_grad_(False).eval();del cp
   source=m['dev'][job['wrong_index']] if job['mode']=='wrong' else e
   mode='oracle' if job['mode']=='oracle' else 'pred25'
   hidden={k:v.cuda() for k,v in load_readouts(job['cache'],source,job['manifest_sha256'])[mode].items()}
   with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):u=raw.connector(hidden)
   del hidden
  torch.manual_seed(job['seed']);prepared=strategy.prepare_training_inputs(with_text(av,text),sampler)
  if u is not None:prepared=append_conditions(prepared,u)
  torch.manual_seed(job['seed']);sampled=guided_sample(renderer,prepared,negative,steps=30)
  video,audio=unpack_latents(sampled,av);atomic_save(latent,dict(video=video.cpu(),audio=audio.cpu(),job=job))
  del renderer,raw,prepared,av,text,negative,u,sampled,video,audio;unload()
 saved=torch.load(latent,map_location='cpu',weights_only=True)
 if saved['job']!=job:raise ValueError('Latent identity changed')
 pending=out/'generated.pending.mp4';pending.unlink(missing_ok=True)
 decoded=decode_and_save(cfg,saved['video'],saved['audio'],geom['fps'],pending,raw_audio_path=out/'raw_audio.pt')
 if any(decoded[k]!=geom[k] for k in ('frames','height','width')):raise ValueError('Decode geometry changed')
 pending.replace(out/'generated.mp4');raw=torch.load(out/'raw_audio.pt',weights_only=True)
 wave=raw['wave'].float();waveform=dict(pre_clamp_peak=float(wave.abs().max()),pre_clamp_over_one_fraction=float((wave.abs()>1).float().mean()),finite=bool(torch.isfinite(wave).all()))
 write(out/'completed.json',dict(job=job,sample_id=e['sample_id'],decoded=decoded,waveform=waveform,video_sha256=sha(out/'generated.mp4'),
  target_features_used=job['mode']=='oracle',target_vae_used=False,sampler='nativeEuler30 CFGv3/a7',fixed_gt_reply_timing=True))
if __name__=='__main__':main()
