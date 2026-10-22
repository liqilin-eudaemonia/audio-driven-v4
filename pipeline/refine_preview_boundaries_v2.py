"""Refine the SAME 100 review pairs; source audio, no new models/full scan.

Outputs are separate from v1 and production. Audio energy proposes boundaries,
not speech/laughter semantics. Every uncertain ending remains reviewable.
"""
from __future__ import annotations
import concurrent.futures, copy, io, json, math
from collections import Counter
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.ndimage import maximum_filter1d, uniform_filter1d
from build_head_shoulders_preview import read, run, save

ROOT=Path(__file__).resolve().parent
OLD=ROOT/'outputs_alternate_pairs/plain_screen/visual_quality_v1'
OUT=ROOT/'outputs_alternate_pairs/plain_screen/visual_quality_v2'
REV='boundary-audio-v2-3'
STEP=.01
TARGET_DB=-20.
MAX_GAIN_DB=40.
CEILING_DB=-2.

def db(x):return 20*np.log10(np.maximum(x,1e-10))

def segments(mask):
    edges=np.diff(np.r_[False,mask,False].astype(int))
    return list(zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)))

def frame_rms(a,sr):
    n=round(sr*STEP);nframes=len(a)//n
    return np.sqrt(np.mean(a[:nframes*n].reshape(nframes,n,2).astype(np.float64)**2,axis=1))

def word_masks(r,times):
    masks=np.zeros((len(times),2),bool)
    for w in r['transcript']:
        ch=0 if w['who']=='speaker' else 1
        masks[:,ch]|=(times>=w['start'])&(times<w['end'])
    return masks

def choose_end(r,times,levels,masks):
    refs=[];thresholds=[]
    for ch in range(2):
        speech=levels[masks[:,ch]&(times>=r.get('crop_start',times[0]))&(times<r['end']),ch]
        # Word intervals sometimes contain long silence. Their median frame
        # energy can be the noise floor, making quiet detection impossible.
        ref=float(10*np.log10(np.mean(10**(speech/10)))) if len(speech) else -35.
        floor=float(np.percentile(levels[:,ch],15))
        # Relative to this person's speech. A noisy recording cannot raise the
        # definition of quiet all the way up to speech level.
        threshold=max(-80,min(ref-12,max(ref-24,floor+6,-70)))
        refs.append(ref);thresholds.append(threshold)
    quiet=np.all(levels<=np.array(thresholds),axis=1)
    end=r['end'];nxt=r['next_start'];flags=[];onset=None
    # Only inspect near the annotated NEXT substantive turn. An earlier short
    # speaker backchannel must not become a new floor boundary.
    if nxt is not None:
        active=(levels[:,0]>thresholds[0]+3)&(times>=nxt-.20)&(times<=nxt+.15)
        for a,b in segments(active):
            if (b-a)*STEP>=.04:
                onset=float(times[a]-STEP/2);break
    next_guard=min(nxt,onset) if nxt is not None and onset is not None else nxt
    limit=min(r['context_end']-.05,end+1.5)
    if next_guard is not None:limit=min(limit,next_guard-.08)
    nominal=min(end+.8,limit)
    if limit<end+.12:flags.append('下一轮过近，没有充足的干净收尾空间')
    # Protect the last sustained audible tail before the preferred boundary.
    tail=(times>=end)&(times<=nominal)&(levels[:,1]>thresholds[1]+3)
    last_activity=end
    for a,b in segments(tail):
        if (b-a)*STEP>=.04:last_activity=max(last_activity,float(times[b-1]+STEP/2))
    lower=max(end+.12,last_activity+.06)
    candidates=[]
    for a,b in segments(quiet):
        # At least 60 ms low energy, with 30 ms margin on each side.
        if (b-a)*STEP<.06:continue
        lo=max(lower,float(times[a]+.03))
        hi=min(limit,float(times[b-1]-.03))
        if lo<=hi:candidates.append(float(np.clip(nominal,lo,hi)))
    nearby=[t for t in candidates if nominal-.25<=t<=nominal+.25]
    if nearby:
        chosen=min(nearby,key=lambda t:abs(t-nominal));method='nearby_quiet'
    else:
        later=[t for t in candidates if t>nominal]
        if later:chosen=min(later);method='extended_for_tail'
        else:
            chosen=max(end,nominal);method='fallback_review'
            flags.append('候选边界附近未找到双方持续低能量切口，需要听原声确认')
    chosen=max(end,min(chosen,r['context_end']))
    if nxt is not None and chosen>nxt-.05:
        flags.append('无法同时保证完整回应和下一轮安全间隔')
    return dict(old_end=r['crop_end'],word_end=end,new_end=round(chosen,6),
        tail_seconds=round(chosen-end,6),nominal_end=nominal,next_start=nxt,
        detected_next_onset=onset,next_guard=next_guard,last_tail_activity=last_activity,
        quiet_threshold_dbfs=thresholds,speech_reference_dbfs=refs,
        method=method,needs_review=bool(flags),flags=flags,
        trace=[dict(time=round(float(t),5),speaker_dbfs=round(float(v[0]),2),listener_dbfs=round(float(v[1]),2),quiet=bool(q))
               for t,v,q in zip(times,levels,quiet) if end-.5<=t<=min(end+2,r['context_end'])])

def normalize(a,sr,r,times,rms,masks):
    out=np.empty_like(a);stats=[]
    frame=round(sr*STEP)
    for ch,role in enumerate(['speaker','listener']):
        # Measure only annotated speech inside this pair, then use one gain
        # throughout the entire context. Silence is never individually boosted.
        active=masks[:,ch]&(times>=r['crop_start'])&(times<r['crop_end'])
        values=rms[active,ch]
        if not len(values):raise ValueError('No speech frames '+r['id']+' '+role)
        before=float(db(np.sqrt(np.mean(values**2))))
        gain_db=float(np.clip(TARGET_DB-before,-24,MAX_GAIN_DB))
        x=a[:,ch].astype(np.float64)*10**(gain_db/20)
        ceiling=10**(CEILING_DB/20)
        peaks=maximum_filter1d(np.abs(x),size=max(3,round(.01*sr)|1),mode='nearest')
        envelope=np.minimum(1.,ceiling/np.maximum(peaks,1e-10))
        envelope=np.minimum(envelope,uniform_filter1d(envelope,size=max(3,round(.004*sr)|1),mode='nearest'))
        y=x*envelope
        out[:,ch]=y
        yr=np.sqrt(np.mean(y[:len(rms)*frame].reshape(-1,frame)**2,axis=1))
        after=float(db(np.sqrt(np.mean(yr[active]**2))))
        stats.append(dict(role=role,target_speech_rms_dbfs=TARGET_DB,input_speech_rms_dbfs=before,
            gain_db=gain_db,output_speech_rms_dbfs=after,gain_capped=TARGET_DB-before>MAX_GAIN_DB,
            limited_sample_fraction=float(np.mean(envelope<.999)),output_peak_dbfs=float(db(np.max(np.abs(y))))))
    return out,dict(method='per-person fixed speech RMS gain plus offline lookahead peak limiter',
        channels=stats,target_speech_rms_dbfs=TARGET_DB,max_gain_db=MAX_GAIN_DB,
        peak_ceiling_dbfs=CEILING_DB,fade_ms=10,no_word_gates=True,no_time_shift=True)

def process(original):
    r=copy.deepcopy(original);folder=OUT/'media'/r['id'];folder.mkdir(parents=True,exist_ok=True)
    cached=folder/'completed.json'
    previous=json.loads(cached.read_text()) if cached.exists() else None
    if previous:
        if previous.get('revision')==REV and (folder/'pair.mp4').exists():return previous
    channels=[];sr=None
    for role in ['speaker','listener']:
        with sf.SoundFile(io.BytesIO(read(r['tracks'][role]['refs']['wav']))) as f:
            if sr is not None:assert sr==f.samplerate
            sr=f.samplerate;f.seek(round(r['context_start']*sr))
            a=f.read(round(r['media_duration']*sr),dtype='float32',always_2d=True).mean(axis=1)
            channels.append(np.pad(a,(0,max(0,round(r['media_duration']*sr)-len(a)))))
    a=np.stack(channels,axis=1);rms=frame_rms(a,sr)
    times=r['context_start']+(np.arange(len(rms))+.5)*STEP;masks=word_masks(r,times)
    boundary=choose_end(r,times,db(rms),masks)
    r['old_crop_end']=r['crop_end'];r['crop_end']=boundary['new_end'];r['boundary']=boundary
    y,audit=normalize(a,sr,r,times,rms,masks)
    sf.write(folder/'normalized.wav',y,sr,subtype='PCM_24')
    # Reuse original picture stream: no change of spatial crop or source timing.
    same_audio=bool(previous and previous['revision'] in ['boundary-audio-v2-2',REV]
        and all(abs(x['gain_db']-z['gain_db'])<1e-6 for x,z in zip(previous['audio']['channels'],audit['channels']))
        and (folder/'context.mp4').exists())
    if not same_audio:
        run(['ffmpeg','-v','error','-nostdin','-y','-i',str(OLD/original['preview']),'-i',str(folder/'normalized.wav'),
             '-map','0:v:0','-map','1:a:0','-c:v','copy','-c:a','aac','-b:a','192k','-t',str(r['media_duration']),
             '-movflags','+faststart',str(folder/'context.mp4')])
    offset=r['crop_start']-r['context_start'];duration=r['crop_end']-r['crop_start']
    # A real temporal crop for audition. Fade is encoded here only; continuous
    # context audio remains intact so boundary errors can be heard.
    if not (same_audio and abs(previous['crop_end']-r['crop_end'])<1e-6 and (folder/'pair.mp4').exists()):
        run(['ffmpeg','-v','error','-nostdin','-y','-ss',str(offset),'-i',str(OLD/original['preview']),
             '-ss',str(offset),'-i',str(folder/'normalized.wav'),'-t',str(duration),'-map','0:v:0','-map','1:a:0',
             '-c:v','libx264','-preset','veryfast','-crf','20','-threads','2','-c:a','aac','-b:a','192k',
             '-af',f'afade=t=in:st=0:d=0.01,afade=t=out:st={max(0,duration-.01):.6f}:d=0.01',
             '-movflags','+faststart',str(folder/'pair.mp4')])
    (folder/'normalized.wav').unlink()
    r.update(revision=REV,preview=f'media/{r["id"]}/context.mp4',pair_preview=f'media/{r["id"]}/pair.mp4',
             audio=audit,visual_quality_revision=original['revision'],
             visual_score_interval=[original['crop_start'],original['crop_end']])
    save(OUT/'boundaries'/f'{r["id"]}.json',boundary);save(cached,r)
    print('DONE',r['id'],boundary['method'],round(boundary['tail_seconds'],3),
          'gain',[round(x['gain_db'],1) for x in audit['channels']],flush=True)
    return r

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    old=json.loads((OLD/'manifest.json').read_text())
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:rows=list(ex.map(process,old['rows']))
    summary=dict(revision=REV,samples=len(rows),boundary_methods=dict(Counter(r['boundary']['method'] for r in rows)),
        boundary_review_ids=[r['id'] for r in rows if r['boundary']['needs_review']],
        shortened=sum(r['crop_end']<r['old_crop_end']-.001 for r in rows),
        extended=sum(r['crop_end']>r['old_crop_end']+.001 for r in rows),
        unchanged=sum(abs(r['crop_end']-r['old_crop_end'])<=.001 for r in rows),
        tail_seconds_quantiles=np.quantile([r['boundary']['tail_seconds'] for r in rows],[0,.5,1]).tolist(),
        gain_db_quantiles=np.quantile([x['gain_db'] for r in rows for x in r['audio']['channels']],[0,.5,1]).tolist(),
        output_speech_rms_dbfs_quantiles=np.quantile([x['output_speech_rms_dbfs'] for r in rows for x in r['audio']['channels']],[0,.5,1]).tolist(),
        gain_capped_ids=[r['id'] for r in rows if any(x['gain_capped'] for x in r['audio']['channels'])],
        visual_classification='Prior v1 classification, not recomputed on the new tail',
        source=str(OLD/'manifest.json'))
    save(OUT/'manifest.json',dict(revision=REV,summary=summary,rows=rows));save(OUT/'summary.json',summary)
    with (OUT/'preview_pairs.jsonl').open('w') as f:
        for r in rows:f.write(json.dumps({k:r[k] for k in ['id','pair_key','crop_start','crop_end','old_crop_end','response_start','boundary','audio','preview','pair_preview']},ensure_ascii=False)+'\n')
    print(json.dumps(summary,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
