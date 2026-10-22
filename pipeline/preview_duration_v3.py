"""Same 100 pairs, <=10 s listen+response candidates, sentence-prefix only."""
import concurrent.futures, copy, json, re, shutil, tempfile
from pathlib import Path
from collections import Counter
import numpy as np
from refine_preview_boundaries_v2 import frame_rms, word_masks, db, segments
from build_head_shoulders_preview import run, save

ROOT=Path(__file__).resolve().parent
OLD=ROOT/'outputs_alternate_pairs/plain_screen/visual_quality_v2'
OUT=ROOT/'outputs_alternate_pairs/plain_screen/duration_v3'
REV='duration-v3-1'
MAX_SECONDS=10.
ENCODE_MARGIN=.05
MICRO={'all','yes','yeah','yep','yup','ok','okay','right','mhm','mm','hmm','uh','huh','oh','sure','exactly','absolutely','ha','haha','hahaha','heh','hehe','well'}
TRAILING={'and','but','because','if','the','a','an','to','of','for','with','my','your','are','is','i','by'}

def sentences(r):
    words=[w for w in r['transcript'] if w['who']=='listener' and w['start']>=r['response_start']-1e-5 and w['end']<=r['end']+1e-5]
    results=[];begin=0
    for i,w in enumerate(words):
        text=w['text'].rstrip().rstrip('\"\'”’)]')
        if not re.search(r'[.!?]$',text):continue
        part=words[begin:i+1];phrase=' '.join(x['text'] for x in part)
        ts=re.findall(r"[a-z]+",phrase.lower());nxt=words[i+1]['start'] if i+1<len(words) else None
        reason=None
        if text.endswith('...') or text.endswith('…'):reason='省略号不作为完整句末'
        elif text.lower() in {'mr.','mrs.','ms.','dr.','prof.','etc.'}:reason='缩写句点不作为句末'
        elif ts and ts[-1] in TRAILING:reason='句末是明显的未完成连接词'
        elif nxt is not None and nxt-w['end']<=1.2 and ((ts and all(t in MICRO for t in ts)) or ' '.join(ts) in {'you know what','well i mean','i mean','you know'}):
            reason='短确认或引导语紧接后续内容，不单独截断'
        results.append(dict(word_index=i,word_end=w['end'],text=phrase,prefix_text=' '.join(x['text'] for x in words[:i+1]),
            next_listener_start=nxt,full_response=i==len(words)-1,rejection=reason))
        # Abbreviations are not sentence separators, ellipses stay with the
        # unfinished phrase until a later real terminal punctuation.
        if reason not in ['缩写句点不作为句末','省略号不作为完整句末']:begin=i+1
    return words,results

def choose_cut(r,c,times,levels,thresholds):
    end=c['word_end'];deadline=r['crop_start']+MAX_SECONDS-ENCODE_MARGIN
    # Any following annotated voice (including the same listener's next
    # sentence) is protected. A word already in progress also needs to finish.
    following=[w['start'] for w in r['transcript'] if w['start']>=end-1e-5 and w['end']>end+1e-5]
    nxt=min(following) if following else None
    upper=min(deadline,end+1.5,r['crop_end'])
    if nxt is not None:upper=min(upper,nxt-.08)
    crossing=[w['end'] for w in r['transcript'] if w['start']<end-1e-5 and w['end']>end+1e-5]
    lower=max([end+.12]+[x+.08 for x in crossing])
    nominal=min(end+.8,upper)
    active=(times>=end)&(times<=nominal)&(levels[:,1]>thresholds[1]+3)
    for a,b in segments(active):
        if (b-a)*.01>=.04:lower=max(lower,float(times[b-1]+.005+.06))
    quiet=np.all(levels<=thresholds,axis=1);choices=[]
    for a,b in segments(quiet):
        if (b-a)*.01<.06:continue
        lo=max(lower,float(times[a]+.03));hi=min(upper,float(times[b-1]-.03))
        if lo<=hi:choices.append(float(np.clip(nominal,lo,hi)))
    if not choices:return dict(accepted=False,reason='句末后没有满足时长与后续发声保护条件的低能量切口',guard_start=nxt)
    chosen=min(choices,key=lambda x:abs(x-nominal))
    return dict(accepted=True,new_end=round(chosen,6),duration=round(chosen-r['crop_start'],6),
        tail_seconds=round(chosen-end,6),guard_start=nxt,quiet_threshold_dbfs=thresholds.tolist())

def process(original):
    r=copy.deepcopy(original);folder=OUT/'media'/r['id'];folder.mkdir(parents=True,exist_ok=True)
    cache=folder/'completed.json'
    if cache.exists():return json.loads(cache.read_text())
    original_duration=r['crop_end']-r['crop_start']
    r.update(revision=REV,previous_crop_end=r['crop_end'],original_response_end=r['end'],original_listener_parts=copy.deepcopy(r['listener_parts']))
    words,candidates=sentences(r)
    audit=dict(original_duration=original_duration,original_end=r['crop_end'],max_seconds=MAX_SECONDS,
        candidates=candidates,response_prefix=False,eligible=False,status='pending',reason='',selected=None)
    context=folder/'context.mp4'
    if not context.exists():context.symlink_to(OLD/original['preview'])
    if original_duration<=MAX_SECONDS+1e-6:
        audit.update(eligible=True,status='unchanged',reason='原样本已在10秒内，保留原边界',duration=original_duration)
    else:
        feasible=[c for c in candidates if not c['rejection'] and c['word_end']+.12<=r['crop_start']+MAX_SECONDS-ENCODE_MARGIN]
        if not feasible:
            audit['reason']='倾听阶段已达到10秒' if r['response_start']-r['crop_start']>=10 else '10秒预算内没有能容纳收尾的完整回应句末'
        else:
            with tempfile.TemporaryDirectory(prefix='seamless_duration_') as tmp:
                local=Path(tmp)/'context.mp4';shutil.copyfile(context,local)
                sr=16000
                audio=np.frombuffer(run(['ffmpeg','-v','error','-nostdin','-threads','1','-i',str(local),'-map','0:a:0','-ar',str(sr),'-ac','2','-f','f32le','pipe:1']),np.float32).reshape(-1,2)
                rms=frame_rms(audio,sr);times=r['context_start']+(np.arange(len(rms))+.5)*.01;levels=db(rms);masks=word_masks(r,times)
                thresholds=[]
                for ch in range(2):
                    speech=rms[masks[:,ch]&(times>=r['crop_start'])&(times<r['end']),ch]
                    ref=float(db(np.sqrt(np.mean(speech**2))))
                    floor=float(np.percentile(levels[:,ch],15));thresholds.append(max(-80,min(ref-12,max(ref-24,floor+6,-70))))
                valid=[]
                for c in feasible:
                    c['acoustic']=choose_cut(r,c,times,levels,np.array(thresholds))
                    if c['acoustic']['accepted']:valid.append(c)
                if valid:
                    # Prefer more complete response content. Never use later
                    # sentences alone or pad silence just to reach eight seconds.
                    chosen=max(valid,key=lambda c:c['word_end']);cut=chosen['acoustic']
                    r['crop_end']=cut['new_end'];r['end']=chosen['word_end']
                    audit.update(eligible=True,status='tail_shortened' if chosen['full_response'] else 'prefix',
                        response_prefix=not chosen['full_response'],selected=chosen,duration=cut['duration'],
                        reason='保留完整回应，仅收紧尾部' if chosen['full_response'] else '保留从回应开头到完整句末的前缀；后续内容不另造配对')
                    included=[w for w in words if w['end']<=r['end']+1e-5]
                    r['listener_parts']=[dict(start=r['response_start'],end=r['end'],text=' '.join(w['text'] for w in included),nwords=len(included))]
                    offset=r['crop_start']-r['context_start'];duration=r['crop_end']-r['crop_start']
                    run(['ffmpeg','-v','error','-xerror','-nostdin','-y','-ss',str(offset),'-i',str(local),'-t',str(duration),
                        '-map','0:v:0','-map','0:a:0','-c:v','libx264','-preset','veryfast','-crf','20','-threads','2',
                        '-c:a','aac','-b:a','192k','-af',f'afade=t=in:st=0:d=0.01,afade=t=out:st={duration-.01:.6f}:d=0.01',
                        '-movflags','+faststart',str(folder/'pair.mp4')])
                else:audit['reason']='有转写句末，但未找到不截尾音、不混入后续发声且在10秒内的切口'
    pair=folder/'pair.mp4'
    if not pair.exists():pair.symlink_to(OLD/original['pair_preview'])
    audit.setdefault('duration',None)
    r.update(duration_review=audit,preview=f'media/{r["id"]}/context.mp4',pair_preview=f'media/{r["id"]}/pair.mp4')
    save(cache,r);print(r['id'],audit['status'],audit['duration'],flush=True);return r

def main():
    OUT.mkdir(parents=True,exist_ok=True);source=json.loads((OLD/'manifest.json').read_text())
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:rows=list(ex.map(process,source['rows']))
    counts=dict(Counter(r['duration_review']['status'] for r in rows))
    summary=dict(revision=REV,samples=100,status_counts=counts,eligible=sum(r['duration_review']['eligible'] for r in rows),
        pending_ids=[r['id'] for r in rows if not r['duration_review']['eligible']],
        changed_ids=[r['id'] for r in rows if r['duration_review']['status'] in ['prefix','tail_shortened']],
        note='Sentence completeness is a punctuation/lexical/energy candidate, not a semantic guarantee. Previous boundary/visual review flags remain applicable.')
    save(OUT/'manifest.json',dict(revision=REV,summary=summary,rows=rows));save(OUT/'summary.json',summary)
    for name,eligible in [('eligible',True),('overlong_pending',False)]:
        with (OUT/(name+'.jsonl')).open('w') as f:
            for r in rows:
                if r['duration_review']['eligible']==eligible:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    print(json.dumps(summary,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
