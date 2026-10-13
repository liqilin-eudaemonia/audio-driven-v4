#!/usr/bin/env python3
"""Per-video ASR/vocal SyncNet; full-mode legacy visual/identity metrics."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import unicodedata
import zlib
from pipeline21_residual_supervise import write

MODEL='/private/codes/exp/boxian_workspace/models/benchmark/huggingface/hub/models--openai--whisper-large-v3/snapshots/06f233fe06e710322aca913c1bc4249a0d71fce1'

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def normalized(s):
    return ' '.join(''.join(' ' if unicodedata.category(c)[0] in 'PS' else c for c in unicodedata.normalize('NFKC',s).casefold()).split())

def distance(a,b):
    row=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        nxt=[i]
        for j,y in enumerate(b,1):nxt.append(min(nxt[-1]+1,row[j]+1,row[j-1]+(x!=y)))
        row=nxt
    return row[-1]

def errors(ref,hyp):
    r,h=normalized(ref),normalized(hyp);rc,hc=r.replace(' ',''),h.replace(' ','')
    we,ce=distance(r.split(),h.split()),distance(rc,hc)
    return dict(wer=we/len(r.split()) if r else None,cer=ce/len(rc) if rc else None,
                word_errors=we,reference_words=len(r.split()),character_errors=ce,reference_characters=len(rc))

def classify_empty_vocal(values, log_text, output, fingerprint, source_rms):
    """Recognize only the separator's observed empty-stem failure, never invent a sync score."""
    if len(values)!=1 or values[0].get('status')!='error':return None
    item=values[0];reference=item.get('reference')
    if not reference or Path(reference).name!=reference:return None
    stem=Path(output)/'sync_work'/'vocal_tmp'/(reference+'_vocals.wav')
    if ('stem_source array is near-silent or empty' not in log_text
            or str(stem) not in item.get('error','') or stem.exists()):return None
    return [dict(item,status='no_detectable_vocal',reason='separator_empty_vocal_stem',
                 video_sha256=fingerprint,source_rms=source_rms,metric_valid=False,
                 original_status='error',original_error=item.get('error'))]


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--job',type=Path,required=True);p.add_argument('--gpu',type=int,required=True);a=p.parse_args()
    import sys;sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
    from omni_av_p2.generation_metrics import ensure_idle
    ensure_idle(a.gpu)
    os.environ.update(CUDA_VISIBLE_DEVICES=str(a.gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false')
    plan=json.loads(a.plan.read_text());job=json.loads(a.job.read_text());cfg=json.loads(Path(plan['metrics_config']).read_text());out=Path(job['output']);out.mkdir(parents=True,exist_ok=True)
    identity=dict(job=job,input_sha256=sha(job['video'] if job['kind']=='speech' else job['manifest']))
    ident=out/'identity_source.json'
    if ident.exists() and json.loads(ident.read_text())!=identity:raise ValueError('Metric source changed')
    if not ident.exists():write(ident,identity)
    if (out/'completed.json').exists():return
    if job['kind']=='visual':
        env=dict(os.environ,PYTHONPATH=cfg['benchmark_root'],TORCH_HOME=cfg['torch_home'])
        commands={
            'visual':[cfg['avbench_python'],str(Path(cfg['benchmark_root'])/'metrics/visual_distribution.py'),
                '--manifest',job['manifest'],'--output',str(out/'visual.json'),'--cache-dir',str(out/'feature_cache'),
                '--device','cuda:0','--fid-frames',str(cfg['fid_frames']),'--fvd-frames',str(cfg['fvd_frames']),'--metrics','fid,fvd'],
            'identity':[cfg['id_python'],str(Path(cfg['legacy_trainer'])/'scripts/compute_arcface_id.py'),
                '--manifest',job['manifest'],'--output',str(out/'identity.json'),'--model-root',cfg['id_model_root'],
                '--model-name',cfg['id_model_name'],'--sample-frames',str(cfg['id_frames']),'--cpu']}
        if not plan.get('distribution_metrics',True):
            commands.pop('visual');write(out/'visual.json',{})
        for name,command in commands.items():
            if not (out/(name+'.json')).exists():subprocess.run(command,env=env,cwd=cfg['benchmark_root'],check=True,timeout=6500)
        write(out/'completed.json',dict(identity,visual=json.loads((out/'visual.json').read_text()),identity=json.loads((out/'identity.json').read_text())));return
    video=Path(job['video']);fingerprint=identity['input_sha256']
    if not (out/'asr.json').exists():
        import numpy as np,torch
        from transformers import AutoModelForSpeechSeq2Seq,AutoProcessor,pipeline
        torch.set_num_threads(2)
        wave=np.frombuffer(subprocess.check_output(['ffmpeg','-v','error','-threads','1','-i',str(video),'-vn','-ac','1','-ar','16000','-f','f32le','pipe:1']),dtype=np.float32).copy()
        if not len(wave) or not np.isfinite(wave).all():raise ValueError('Empty or nonfinite audio')
        model=AutoModelForSpeechSeq2Seq.from_pretrained(MODEL,torch_dtype=torch.float16,local_files_only=True,attn_implementation='sdpa').to('cuda')
        model.config.forced_decoder_ids=None;model.generation_config.forced_decoder_ids=None
        processor=AutoProcessor.from_pretrained(MODEL,local_files_only=True)
        recognizer=pipeline('automatic-speech-recognition',model=model,tokenizer=processor.tokenizer,feature_extractor=processor.feature_extractor,device='cuda:0',torch_dtype=torch.float16)
        kwargs=dict(task='transcribe',max_new_tokens=128,condition_on_prev_tokens=False,compression_ratio_threshold=1.35,temperature=[0.,.2,.4,.6,.8,1.],logprob_threshold=-1.,no_speech_threshold=.6)
        torch.manual_seed(20261010)
        with torch.inference_mode():result=recognizer({'raw':wave,'sampling_rate':16000},return_timestamps=True,generate_kwargs=kwargs)
        text=result['text'].strip();ratio=len(text.encode())/max(1,len(zlib.compress(text.encode())))
        write(out/'asr.json',dict(text=text,chunks=result.get('chunks'),reference=job['reply'],**errors(job['reply'],text),model=MODEL,
            no_reference_prompt=True,video_sha256=fingerprint,decoding=kwargs,compression_ratio=ratio,repetition_warning=ratio>1.35,
            audio_seconds=len(wave)/16000,rms=float(np.sqrt(np.mean(wave**2))),reference_policy='reply of this mode; GT is calibration; Vocal excluded'))
        del recognizer,processor,model;import gc;gc.collect();torch.cuda.empty_cache()
    sync=out/'syncnet.json'
    if not sync.exists():
        if not job['reply'].strip():values=[dict(status='not_applicable',reason='empty_reply',video_sha256=fingerprint)]
        else:
            script=Path(cfg['syncnet_root'])/'run_vocal_sync.py'
            sync_log=out/'sync_execution.log'
            with sync_log.open('w') as log:
                subprocess.run([cfg['sync_python'],str(script),'--videofile',str(video),'--data_dir',str(out/'sync_work'),
                    '--output_json',str(out/'sync.pending.json'),'--require_single_face','--model_file_dir',str(script.parent/'models/audio-separator')],cwd=script.parent,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=6000)
            values=json.loads((out/'sync.pending.json').read_text())
            empty=classify_empty_vocal(values,sync_log.read_text(errors='replace'),out,fingerprint,
                                       json.loads((out/'asr.json').read_text())['rms'])
            if empty is not None:values=empty
            if len(values)!=1 or values[0]['status']=='error':raise RuntimeError('SyncNet error: '+str(values))
            for v in values:
                v['video_sha256']=fingerprint
                if v.get('av_offset') is not None:v['av_offset_ms']=40*v['av_offset']
        write(sync,values)
    write(out/'completed.json',dict(identity,asr=json.loads((out/'asr.json').read_text()),syncnet=json.loads(sync.read_text()),
        degraded=any(v.get('status')=='no_detectable_vocal' for v in json.loads(sync.read_text())),
        metric_protocol='speech.v2.empty-vocal-explicit'))

if __name__=='__main__':main()
