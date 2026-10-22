"""Small, review-only turn experiment. Never changes production manifests.

Read metadata from the 30 already sampled conversations; export a diagnostic
sample with raw continuous stereo audio and context on both sides.
"""
from __future__ import annotations
import argparse
import concurrent.futures
import io
import json
import random
import re
from pathlib import Path

import numpy as np
import soundfile as sf
from build_head_shoulders_preview import refs, read, uri, run, save, geometry
from scan_alternate_pairs import parse_words, utterances
from scan_turn_pairs import build_floors, MAX_WORD_S
from screen_plain_turns import structure_reason

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'outputs_alternate_pairs/plain_screen/turn_rules_v2_review'
SOURCE = ROOT/'outputs_alternate_pairs/plain_screen/head_shoulders_720p30/manifest.json'
SEED = 20261022
REVISION = 'rules-v2-review-3'
FEEDBACK = {'yeah','yes','yep','yup','okay','ok','right','mhm','mm hmm','uh huh','hmm','oh','oh yeah','oh okay','yeah yeah','right right','i see','sure','exactly','absolutely'}

def normalized(s):
    return re.sub(r'[^a-z ]', '', s.lower()).strip()

def raw_utterances(words, who):
    # No 1 s word cap and no minimum duration deletion in this experiment.
    parts=[]
    for a,b,t in words:
        if b<=a: continue
        if parts and a-parts[-1]['end']<=.6:
            p=parts[-1];p['end']=max(p['end'],b);p['text']+=' '+t;p['nwords']+=1;p['words'].append([a,b,t])
        else:
            parts.append(dict(who=who,start=a,end=b,text=t,nwords=1,words=[[a,b,t]]))
    return parts

def new_floors(groups):
    events=sorted([p for v in groups.values() for p in v],key=lambda x:(x['start'],x['end'],x['who']))
    floors=[]
    for i,u in enumerate(events):
        if not floors:
            floors.append(dict(who=u['who'],start=u['start'],end=u['end'],parts=[u],inserts=[],flags=[]));continue
        f=floors[-1]
        if u['who']==f['who']:
            if u['start']-f['end']>3: f['flags'].append('同一人内部停顿超过3秒，未自动切断')
            f['parts'].append(u);f['end']=max(f['end'],u['end']);continue
        nxt=next((e for e in events[i+1:] if e['who']==f['who']),None)
        quick=nxt is not None and -.05<=nxt['start']-u['end']<=1.6
        short=u['nwords']<=5 and u['end']-u['start']<=1.5
        lexical=normalized(u['text']) in FEEDBACK
        last=f['parts'][-1]['text'].rstrip()
        question=last.endswith('?')
        unfinished=not last.endswith(('.', '?', '!'))
        overlap=u['start']<f['end']-.05
        contained=u['end']<=f['end']+.05
        kind=None
        same_next=next((e for e in events[i+1:] if e['who']==u['who']),None)
        prefix=(not u['text'].rstrip().endswith(('.', '?', '!')) and same_next is not None
                and -.05<=same_next['start']-u['end']<=1.6 and not lexical)
        direct_question=u['text'].rstrip().endswith('?')
        # Temporal containment cannot by itself prove semantic feedback.
        if overlap and contained and u['nwords']<=6 and u['end']-u['start']<=2 and not prefix and not direct_question:
            kind='疑似backchannel' if lexical else '轮内插话（归属待复查）'
        elif short and quick and unfinished and not direct_question and not prefix:
            kind='短插话后原说话人续说（待复查）'
        elif lexical and short and quick and not question:
            kind='疑似backchannel'
        if kind:
            f['inserts'].append(dict(u,kind=kind))
            f['flags'].append('反馈与正式短答仍需听原声确认' if lexical else '插话合并为候选，未验证语义完整性')
        else:
            flags=[]
            if overlap: flags.append('重叠接话，保留复查')
            if prefix: flags.append('疑似下一轮回应前缀，未作为反馈吞并')
            floors.append(dict(who=u['who'],start=u['start'],end=u['end'],parts=[u],inserts=[],flags=flags))
    return floors

def pack(parts):
    return [dict(start_s=p['start'],end_s=p['end'],text=p['text'],nwords=p['nwords']) for p in parts]

def old_candidates(words):
    fs=build_floors({k:utterances(v,MAX_WORD_S) for k,v in words.items()})
    results=[]
    for i,(s,l) in enumerate(zip(fs,fs[1:])):
        bucket=('case2_dropped' if s['end']-l['start']>.5 else 'case2_immediate') if l['case2'] else ('with_backchannel' if s['backchannels'] or l['backchannels'] else 'plain')
        reason=structure_reason(dict(speaker_parts=pack(s['parts']),listener_parts=pack(l['parts']))) if bucket=='plain' else bucket
        results.append(dict(speaker=s['who'],listener=l['who'],start=s['start'],end=l['end'],response_start=l['start'],bucket=bucket,screen_reason=reason or '通过结构筛选（未重跑其他筛选）',speaker_parts=pack(s['parts']),listener_parts=pack(l['parts'])))
    return results

def load_conversation(row,number):
    tracks={};words={};events={}
    for role in ['speaker','listener']:
        source=refs(row[role+'_archive'],row[role+'_id'],['json','wav','mp4'])
        meta=json.loads(read(source['json']));words[role]=parse_words(meta)
        events[role]=raw_utterances(words[role],role)
        tracks[role]=dict(id=row[role+'_id'],archive=row[role+'_archive'],refs=source)
    fs=new_floors(events);old=old_candidates(words);candidates=[]
    for i,(s,l) in enumerate(zip(fs,fs[1:])):
        # Retain the same listener orientation as the existing head/shoulder sample.
        if s['who']!='speaker':continue
        if sum(p['nwords'] for p in s['parts'])<5 or s['end']-s['start']<1.2:continue
        if not (2<=l['end']-s['start']<=35):continue
        if l['start']-s['end']>6:continue
        flags=list(s['flags']+l['flags'])
        if sum(p['nwords'] for p in l['parts'])<5 or l['end']-l['start']<1.2:flags.append('短回应候选保留，未按最少词数删除')
        if not l['parts'][-1]['text'].rstrip().endswith(('.', '?', '!')):flags.append('回应末尾没有句末标点，可能截句或源标点不准')
        prev=fs[i-1] if i else None;nxt=fs[i+2] if i+2<len(fs) else None
        if prev and prev['end']>s['start']+.05: flags.append('起点与上一轮重叠')
        if nxt and nxt['start']<l['end']-.05: flags.append('终点与下一轮重叠')
        suspicious=[dict(who=k,start=a,end=b,text=t) for k,ww in words.items() for a,b,t in ww if b-a>1 and a<l['end'] and b>s['start']]
        if suspicious: flags.append('存在超过1秒的词时间戳；本次未截短，需复查对齐')
        ins=s['inserts']+l['inserts']
        # Classification is diagnostic, not human-verified ground truth.
        categories=[]
        if len(l['parts'])>=2 and any('续说' in p['kind'] for p in l['inserts']):categories.append('插话后续说')
        if any(p['who']=='listener' and p['start']<l['start'] and normalized(p['text']) in FEEDBACK for p in ins):categories.append('保留倾听反馈')
        if sum(p['nwords'] for p in l['parts'])<5 or l['end']-l['start']<1.2:categories.append('保留短回应')
        matching=[o for o in old if o['speaker']=='speaker']
        # Match response onset first, so a merged continuation shows the original
        # first fragment rather than an unrelated nearby response.
        before=min(matching,key=lambda o:abs(o['response_start']-l['start'])+abs(o['start']-s['start'])*.2) if matching else None
        if before and abs(before['response_start']-l['start'])>3:before=None
        candidates.append(dict(source_sample=number,interaction=row['interaction'],label=row['label'],split=row['split'],tracks=tracks,
          start=s['start'],end=l['end'],response_start=l['start'],speaker_end=s['end'],speaker_parts=pack(s['parts']),listener_parts=pack(l['parts']),inserts=ins,
          flags=sorted(set(flags)),suspicious_words=suspicious,categories=categories,old=before,
          previous_end=prev['end'] if prev else None,next_start=nxt['start'] if nxt else None,
          transcript=sorted([dict(who=k,start=a,end=b,text=t) for k,ww in words.items() for a,b,t in ww],key=lambda x:x['start']),
          utterances=sorted(events['speaker']+events['listener'],key=lambda x:x['start']),
          original_roi=row['roi']))
    return candidates

def select():
    rows=json.loads(SOURCE.read_text())['rows'];allc=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for n,cc in enumerate(pool.map(lambda x:load_conversation(x[1],x[0]+1),enumerate(rows))):
            allc.extend(cc);print('METADATA',n+1,'candidates',len(cc),flush=True)
    chosen=[];used=set();rng=random.Random(SEED)
    # Include known old sample 3 and three controls for a direct comparison.
    for number in [3,1,6,12]:
        r=rows[number-1];cs=[c for c in allc if c['source_sample']==number]
        if not cs:continue
        c=min(cs,key=lambda c:abs(c['start']-r['speaker_parts'][0]['start_s'])+abs(c['response_start']-r['listener_parts'][0]['start_s']))
        c['old']=dict(speaker='speaker',listener='listener',start=r['speaker_parts'][0]['start_s'],end=max(p['end_s'] for p in r['listener_parts']),
                      response_start=r['listener_parts'][0]['start_s'],bucket=r['bucket'],screen_reason='原 ready 清单已保留',
                      speaker_parts=r['speaker_parts'],listener_parts=r['listener_parts'],saved_kept=True,
                      actual_start=r['clip_start_s'],actual_end=r['clip_end_s'])
        c['review_group']='原预览对照';chosen.append(c);used.add((c['interaction'],c['start']))
    for category in ['插话后续说','保留倾听反馈','保留短回应']:
        cs=[c for c in allc if category in c['categories'] and (c['interaction'],c['start']) not in used]
        rng.shuffle(cs)
        # Prioritize samples without very long stretched words; keep the known
        # timestamp issue in the original control instead of hiding it.
        cs.sort(key=lambda c: (any(w['end']-w['start']>2 for w in c['suspicious_words']),len(c['flags'])))
        count=0
        # Prefer distinct conversations within each diagnostic group.
        group_interactions=set()
        for c in cs:
            if c['interaction'] in group_interactions:continue
            c['review_group']=category;chosen.append(c);used.add((c['interaction'],c['start']));group_interactions.add(c['interaction']);count+=1
            if count==4:break
        if count<4:print('FEWER',category,count,flush=True)
    for i,c in enumerate(chosen):c['id']=f'{i+1:02d}';c['revision']=REVISION
    save(OUT/'selection.json',dict(seed=SEED,source=str(SOURCE),conversations_read=30,candidate_count=len(allc),rows=chosen))
    return chosen

def render(c):
    folder=OUT/'media'/c['id'];folder.mkdir(parents=True,exist_ok=True)
    cached=folder/'completed.json'
    if cached.exists():
        previous=json.loads(cached.read_text())
        if previous['revision']==REVISION:return previous
    infos={}
    for role,t in c['tracks'].items():
        probe=json.loads(run(['ffprobe','-v','error','-show_streams','-of','json',uri(t['refs']['mp4'])]))
        infos[role]=next(s for s in probe['streams'] if s['codec_type']=='video')
    limit=min(float(x['duration']) for x in infos.values())
    # Context covers both the old and new intervals. Give 5 s minimum, extend
    # to nearby utterance edges up to 12 s; never silently present context as crop.
    old=c['old'];old_start=old.get('actual_start',max(0,old['start']-.2)) if old else c['start']
    old_end=old.get('actual_end',old['end']+.8) if old else c['end']
    start=min(c['start']-.2,old_start);end=max(c['end']+.35,old_end)
    ctx0=max(0,start-5);ctx1=min(limit,end+5)
    for u in c['utterances']:
        if u['start']<ctx0<u['end']:ctx0=max(0,start-12,u['start']-.15)
        if u['start']<ctx1<u['end']:ctx1=min(limit,end+12,u['end']+.15)
    c.update(context_start=ctx0,context_end=ctx1,media_duration=ctx1-ctx0,
             crop_start=max(0,c['start']-.2),crop_end=min(limit,c['end']+.35),
             old_crop_start=old_start,old_crop_end=min(limit,old_end),
             context_before_actual=c['start']-ctx0,context_after_actual=ctx1-c['end'])
    c['flags']=list(c['flags'])
    if c['next_start'] is not None and c['next_start']<c['crop_end']:
        # Keep complete listener words if overlap exists, but don't add silence
        # padding that unnecessarily reaches a later speaker onset.
        c['crop_end']=max(c['end'],min(c['crop_end'],c['next_start']-.03))
    if ctx0==0:c['flags'].append('前文到达源文件起点')
    if abs(ctx1-limit)<.01:c['flags'].append('后文到达源文件终点')
    c['transcript']=[w for w in c['transcript'] if w['end']>=ctx0 and w['start']<=ctx1]
    c['utterances']=[u for u in c['utterances'] if u['end']>=ctx0 and u['start']<=ctx1]
    audios={};sr=None
    for role,t in c['tracks'].items():
        with sf.SoundFile(io.BytesIO(read(t['refs']['wav']))) as f:
            if sr is not None and sr!=f.samplerate:raise ValueError('sample rate mismatch')
            sr=f.samplerate;f.seek(round(ctx0*sr));a=f.read(round((ctx1-ctx0)*sr),dtype='float32',always_2d=True).mean(axis=1)
        count=round((ctx1-ctx0)*sr);audios[role]=np.pad(a,(0,max(0,count-len(a))))[:count]
    stereo=np.stack([audios['speaker'],audios['listener']],axis=1)
    rms=float(np.sqrt(np.mean(stereo**2)));peak=float(np.max(np.abs(stereo)))
    gain=min(10,.10/max(rms,1e-6),.95/max(peak,1e-6))
    c['audio']=dict(common_gain=gain,raw_continuous=True,channels=['speaker','listener'],no_word_gates=True)
    sf.write(folder/'continuous.wav',stereo*gain,sr,subtype='PCM_16')
    # Both full frames, aspect-correct; spatial ROI is only an HTML overlay.
    args=['ffmpeg','-v','error','-nostdin','-y']
    for role in ['speaker','listener']:
        args+=['-threads','1','-ss',str(ctx0),'-t',str(ctx1-ctx0),'-i',uri(c['tracks'][role]['refs']['mp4'])]
    args+=['-i',str(folder/'continuous.wav')]
    chains=[];layout={}
    for i,role in enumerate(['speaker','listener']):
        v=infos[role];sa=v.get('sample_aspect_ratio','1:1');nn,dd=map(int,sa.split(':'));sar=nn/dd
        ratio=int(v['width'])*sar/int(v['height']);w=2*round(min(360,640*ratio)/2);h=2*round(w/ratio/2)
        chains.append(f'[{i}:v]setpts=PTS-STARTPTS,scale={w}:{h},setsar=1,fps=25,pad=360:640:(ow-iw)/2:(oh-ih)/2:black[v{i}]')
        layout[role]=dict(x=i*360+(360-w)/2,y=(640-h)/2,width=w,height=h,source_width=int(v['width']),source_height=int(v['height']),sar=sar)
    chains.append('[v0][v1]hstack=inputs=2[v]')
    args+=['-filter_complex_threads','1','-filter_complex',';'.join(chains),'-map','[v]','-map','2:a','-t',str(ctx1-ctx0),'-c:v','libx264','-threads','2','-preset','veryfast','-crf','22','-pix_fmt','yuv420p','-c:a','aac','-b:a','192k','-movflags','+faststart',str(folder/'context.mp4')]
    run(args);(folder/'continuous.wav').unlink()
    # Compute a new fixed listener ROI over the proposed interval, including
    # any continuation added by the experiment, using existing keypoints.
    try:
        t=c['tracks']['listener'];npz=refs(t['archive'],t['id'],['npz'])['npz'];v=infos['listener']
        with np.load(io.BytesIO(read(npz)),allow_pickle=False) as z:
            k=z['boxes_and_keypoints:keypoints'];nn,dd=map(int,v['r_frame_rate'].split('/'));fps=nn/dd
            if abs(len(k)/fps-float(v['duration']))>.15:raise ValueError('landmark timeline mismatch')
            k=k[int(c['crop_start']*fps):int(np.ceil(c['crop_end']*fps))]
            c['roi']=geometry(k,int(v['width']),int(v['height']),layout['listener']['sar'])
    except ValueError as exc:
        c['roi']=None;c['flags'].append('新范围头肩框不可用：'+str(exc))
    c['layout']=layout;c['preview']=f'media/{c["id"]}/context.mp4'
    # Completed manifest needs no large per-word arrays inside utterance copies.
    c['utterances']=[{k:v for k,v in u.items() if k!='words'} for u in c['utterances']]
    save(cached,c);print('RENDERED',c['id'],c['review_group'],round(ctx1-ctx0,2),flush=True)
    return c

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--render',action='store_true');ap.add_argument('--reselect',action='store_true');args=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    rows=json.loads((OUT/'selection.json').read_text())['rows'] if (OUT/'selection.json').exists() and not args.reselect else select()
    if args.render:
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:rows=list(pool.map(render,rows))
        save(OUT/'manifest.json',dict(revision=REVISION,seed=SEED,rows=rows))
    print('DONE',len(rows),flush=True)

if __name__=='__main__':main()
