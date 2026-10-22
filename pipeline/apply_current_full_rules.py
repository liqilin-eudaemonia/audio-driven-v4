"""Apply pinned current turn/10s rules to the raw full dataset.

Uses raw PCM energy only: NO visual filtering, gain normalization or rendering.
Source-global 10 ms bins avoid per-preview resampling/encoding dependencies.
"""
from __future__ import annotations
import argparse, concurrent.futures, gzip, hashlib, json, math, multiprocessing, os, shutil, sqlite3, tempfile, time
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from scan_current_full import ROOT, OUT, REV, save
from preview_listen_response_v3 import floors_from_words, assess
from refine_preview_boundaries_v2 import choose_end, db
from preview_duration_v3 import sentences, choose_cut

DB=None

def init_worker(path):
    global DB
    DB=sqlite3.connect('file:'+str(path)+'?mode=ro',uri=True)

def load_person(key):
    row=DB.execute('SELECT payload FROM people WHERE key=?',(key,)).fetchone()
    if row is None:raise KeyError(key)
    return json.loads(gzip.decompress(row[0]))

def pcm_rms(person):
    """Read WAV data directly in aligned 10-second chunks; retain 10ms RMS."""
    w=person['wav'];sr=w['samplerate'];channels=w['channels'];frame=round(sr*.01)
    if sr%100:raise ValueError('Sample rate not divisible into exact 10ms frames')
    count=w['frames']//frame;result=np.empty(count,np.float64)
    bits=w['bits'];kind=w['format'];width=bits//8
    with open(person['refs']['wav']['archive'],'rb') as f:
        f.seek(w['data_offset'])
        for first in range(0,count,1000):
            n=min(1000,count-first);raw=f.read(n*frame*channels*width)
            if len(raw)!=n*frame*channels*width:raise ValueError('Truncated WAV PCM')
            if kind==3 and bits in [32,64]:a=np.frombuffer(raw,dtype='<f'+str(width)).astype(np.float64)
            elif kind==1 and bits==24:
                b=np.frombuffer(raw,np.uint8).reshape(-1,3).astype(np.int32)
                a=(b[:,0]|b[:,1]<<8|b[:,2]<<16);a=((a^0x800000)-0x800000).astype(np.float64)/8388608.
            elif kind==1 and bits in [8,16,32]:
                a=np.frombuffer(raw,dtype='u1' if bits==8 else '<i'+str(width)).astype(np.float64)
                if bits==8:a-=128
                a/=2**(bits-1)
            else:raise ValueError('Unsupported PCM type')
            if channels>1:a=a.reshape(-1,channels).mean(axis=1)
            if not np.isfinite(a).all():raise ValueError('Non-finite WAV values')
            result[first:first+n]=np.sqrt(np.mean(a.reshape(n,frame)**2,axis=1))
    return result

def word_mask(words,n):
    diff=np.zeros(n+1,np.int32)
    for a,b,_ in words:
        left=max(0,min(n,math.ceil((a-.005)/.01-1e-9)))
        right=max(0,min(n,math.ceil((b-.005)/.01-1e-9)))
        if right>left:diff[left]+=1;diff[right]-=1
    return np.cumsum(diff[:-1])>0

def merge_seconds(intervals):
    total=0.;end=-1e99
    for a,b in sorted(intervals):
        if b>end:total+=b-max(a,end);end=b
    return total

def pack(parts):return [dict(start=p['start'],end=p['end'],text=p['text'],nwords=p['nwords']) for p in parts]

def make_row(conv,fs,i,people,limit):
    s,l=fs[i:i+2];prev=fs[i-1] if i else None;nxt=fs[i+2] if i+2<len(fs) else None
    start=max(0,s['start']-.2)
    if prev:start=max(start,min(s['start'],prev['end']+.03))
    end=min(limit,l['end']+.35)
    if nxt:end=max(l['end'],min(end,nxt['start']-.03))
    return dict(pair_key='/'.join([conv['label'],conv['split'],conv['interaction'],s['turn_id']+'-'+l['turn_id']]),
        interaction=conv['interaction'],label=conv['label'],split=conv['split'],speaker_person=s['who'],listener_person=l['who'],
        speaker_id=people[s['who']]['id'],listener_id=people[l['who']]['id'],speaker_turn=s['turn_id'],listener_turn=l['turn_id'],
        start=s['start'],end=l['end'],response_start=l['start'],speaker_end=s['end'],crop_start=start,crop_end=end,
        previous_end=prev['end'] if prev else None,next_start=nxt['start'] if nxt else None,
        speaker_parts=pack(s['parts']),listener_parts=pack(l['parts']),original_response_end=l['end'],
        micro_listening_count=len(s['micros']),micro_response_count=len(l['micros']),
        micro_reactions=[dict(who=u['who'],start=u['start'],end=u['end'],text=u['text']) for f in [s,l] for u in f['micros']])

def attach_context(r,words,events,limit):
    begin=max(0,r['start']-.2-5);end=min(limit,r['end']+.35+5)
    for u in events:
        if u['start']<begin<u['end']:begin=max(0,r['start']-.2-12,u['start']-.15)
        if u['start']<end<u['end']:end=min(limit,r['end']+.35+12,u['end']+.15)
    r['context_start']=begin;r['context_end']=end
    r['transcript']=sorted([dict(who='speaker' if who==r['speaker_person'] else 'listener',start=a,end=b,text=t)
        for who,ww in words.items() for a,b,t in ww if b>=begin and a<=end],key=lambda w:w['start'])

def preliminary(r,words):
    # Duration-only upper bound, no audio result is implied here.
    if r['end']-r['crop_start']<=10+1e-6:return True,None
    if r['response_start']-r['crop_start']>=10:return False,'listening_at_least_10s'
    r['transcript']=[dict(who='listener',start=a,end=b,text=t) for a,b,t in words[r['listener_person']]
                    if a>=r['response_start']-1e-5 and b<=r['end']+1e-5]
    _,candidates=sentences(r)
    possible=any(not c['rejection'] and c['word_end']+.12<=r['crop_start']+9.95 for c in candidates)
    r.pop('transcript',None)
    return possible,None if possible else 'no_sentence_with_tail_within_budget'

def acoustic_cut(r,words,events,limit,rms,masks):
    attach_context(r,words,events,limit)
    n=min(len(x) for x in rms.values());first=max(0,math.ceil((r['context_start']-.005)/.01));last=min(n,math.ceil((r['context_end']-.005)/.01))
    times=(np.arange(first,last)+.5)*.01
    roles=[r['speaker_person'],r['listener_person']]
    values=np.stack([rms[k][first:last] for k in roles],axis=1)
    mask=np.stack([masks[k][first:last] for k in roles],axis=1)
    levels=db(values)
    boundary=choose_end(r,times,levels,mask)
    r['v2_reference_end']=boundary['new_end'];r['crop_end']=boundary['new_end']
    if r['crop_end']-r['crop_start']<=10+1e-6:
        return dict(eligible=True,kind='full_response',new_end=r['crop_end'],word_end=r['end'],
            response_prefix=False,boundary_review=boundary['needs_review'],boundary_flags=boundary['flags'],tail_seconds=boundary['tail_seconds'])
    _,candidates=sentences(r)
    thresholds=[]
    for ch in range(2):
        speech=values[mask[:,ch]&(times>=r['crop_start'])&(times<r['end']),ch]
        if not len(speech):return dict(eligible=False,reason='no_speech_frames_for_energy_reference')
        ref=float(db(np.sqrt(np.mean(speech**2))));floor=float(np.percentile(levels[:,ch],15))
        thresholds.append(max(-80,min(ref-12,max(ref-24,floor+6,-70))))
    feasible=[c for c in candidates if not c['rejection'] and c['word_end']+.12<=r['crop_start']+9.95]
    valid=[]
    for c in feasible:
        c['acoustic']=choose_cut(r,c,times,levels,np.array(thresholds))
        if c['acoustic']['accepted']:valid.append(c)
    if not valid:
        return dict(eligible=False,reason='no_quiet_sentence_cut' if feasible else 'no_sentence_with_tail_within_budget')
    chosen=max(valid,key=lambda c:c['word_end']);cut=chosen['acoustic']
    return dict(eligible=True,kind='tail_shortened' if chosen['full_response'] else 'response_prefix',
        new_end=cut['new_end'],word_end=chosen['word_end'],response_prefix=not chosen['full_response'],
        boundary_review=False,boundary_flags=[],tail_seconds=cut['tail_seconds'],
        next_voice_guard=cut['guard_start'],selected_sentence=chosen['text'],prefix_text=chosen['prefix_text'])

def format_kept(r,result,people):
    newend=result['new_end'];duration=newend-r['crop_start']
    if not (0<duration<=10+1e-6 and newend>=result['word_end']-1e-6):raise AssertionError('Invalid final interval')
    out={k:r[k] for k in ['pair_key','interaction','label','split','speaker_person','listener_person','speaker_id','listener_id',
        'speaker_turn','listener_turn','speaker_parts','micro_listening_count','micro_response_count','micro_reactions']}
    listener_words=[(a,b,t) for a,b,t in people[r['listener_person']]['words'] if a>=r['response_start']-1e-5 and b<=result['word_end']+1e-5]
    out.update(revision=REV,start_s=round(r['crop_start'],6),end_s=round(newend,6),duration_s=round(duration,6),
        response_start_s=r['response_start'],speaker_end_s=r['speaker_end'],response_word_end_s=result['word_end'],
        original_response_end_s=r['original_response_end'],v2_reference_end_s=r.get('v2_reference_end'),
        listener_text=' '.join(w[2] for w in listener_words),original_listener_parts=r['listener_parts'],
        segmentation=result,audio_normalized=False,visual_filtered=False,media_rendered=False,
        source={role:dict(id=people[r[role+'_person']]['id'],refs=people[r[role+'_person']]['refs'],
            video_duration=people[r[role+'_person']]['video_duration'],audio_duration=people[r[role+'_person']]['wav']['duration']) for role in ['speaker','listener']})
    # Only micro reactions overlapping the retained interval belong to it.
    out['micro_reactions']=[u for u in out['micro_reactions'] if u['start']<newend and u['end']>r['crop_start']]
    out['original_micro_response_count']=out['micro_response_count']
    out['micro_response_count']=sum(u['who']==r['speaker_person'] for u in out['micro_reactions'])
    return out

def process_conversation(conv):
    people={who:load_person(key) for who,key in zip(['A','B'],conv['keys'])}
    words={k:p['words'] for k,p in people.items()};counts=Counter();reasons=Counter();audit=[];pending=[];kept=[]
    counts['conversations']=1
    if not all(words.values()):counts['empty_transcript_conversations']=1;return kept,pending,audit,dict(counts),dict(reasons)
    fs,events=floors_from_words(words);counts['floors']=len(fs);counts['all_adjacent_pairs']=max(0,len(fs)-1)
    long_words={k:[w for w in ww if w[1]-w[0]>1.5] for k,ww in words.items()}
    limit=min(p[k] if k=='video_duration' else p['wav']['duration'] for p in people.values() for k in ['video_duration','audio_duration'])
    candidates=[]
    for i in range(len(fs)-1):
        status,flags,overlaps,suspicious=assess(fs,i,long_words)
        r=make_row(conv,fs,i,people,limit)
        base=dict(pair_key=r['pair_key'],label=conv['label'],split=conv['split'],status=status,start_s=r['crop_start'],
            original_response_end_s=r['end'],response_start_s=r['response_start'],flags=flags)
        if status!='candidate':
            counts['turn_'+status]+=1;reasons.update('turn:'+x.split('（')[0] for x in set(flags));audit.append(base);continue
        counts['structure_candidates']+=1
        if r['start']<0 or r['end']>limit+1e-6 or r['crop_start']>=limit:
            counts['source_time_review']+=1;base.update(status='source_time_review',source_limit=limit);audit.append(base);continue
        possible,why=preliminary(r,words)
        if not possible:
            counts['overlong_pending']+=1;reasons[why]+=1;base.update(status='overlong_pending',reason=why);pending.append(base);audit.append(base);continue
        counts['transcript_duration_candidates']+=1;candidates.append((r,base))
    if candidates:
        counts['audio_read_conversations']=1
        try:
            rms={k:pcm_rms(p) for k,p in people.items()};masks={k:word_mask(words[k],len(rms[k])) for k in people}
            counts['audio_data_bytes']=sum(p['wav']['data_bytes'] for p in people.values())
        except Exception as e:
            counts['audio_error_pairs']+=len(candidates);counts['audio_error_conversations']=1
            for r,base in candidates:base.update(status='audio_error',reason=type(e).__name__+': '+str(e));audit.append(base)
            return kept,pending,audit,dict(counts),dict(reasons)
        for r,base in candidates:
            try:result=acoustic_cut(r,words,events,limit,rms,masks)
            except Exception as e:
                counts['processing_error_pairs']+=1;base.update(status='processing_error',reason=type(e).__name__+': '+str(e));audit.append(base);continue
            if not result['eligible']:
                counts['overlong_pending']+=1;reasons[result['reason']]+=1;base.update(status='overlong_pending',reason=result['reason']);pending.append(base);audit.append(base);continue
            final=format_kept(r,result,people);kept.append(final)
            counts['kept_'+result['kind']]+=1;counts['kept_pairs']+=1;counts['kept_seconds']+=final['duration_s']
            if result['boundary_review']:counts['kept_boundary_review']+=1;counts['kept_boundary_review_seconds']+=final['duration_s']
            else:counts['kept_clean_boundary']+=1;counts['kept_clean_boundary_seconds']+=final['duration_s']
            counts['kept_with_listening_micro']+=bool(r['micro_listening_count'])
            counts['duration_'+('under2' if final['duration_s']<2 else '2to4' if final['duration_s']<4 else '4to6' if final['duration_s']<6 else '6to8' if final['duration_s']<8 else '8to10')]+=1
            base.update(status='kept',end_s=final['end_s'],duration_s=final['duration_s'],kind=result['kind'],boundary_review=result['boundary_review']);audit.append(base)
    if kept:
        counts['conversations_with_kept']=1
        counts['unique_conversation_seconds']=merge_seconds([(r['start_s'],r['end_s']) for r in kept])
        counts['unique_listener_seconds']=sum(merge_seconds([(r['start_s'],r['end_s']) for r in kept if r['listener_id']==p['id']]) for p in people.values())
    return kept,pending,audit,dict(counts),dict(reasons)

def batch(task):
    number,convs=task;folder=OUT/'result_shards'/f'{number:05d}';marker=folder/'summary.json'
    if marker.exists():return json.loads(marker.read_text())
    folder.mkdir(parents=True,exist_ok=True);counts=Counter();reasons=Counter();subsets=defaultdict(Counter)
    paths={k:folder/(k+'.jsonl.gz.pending') for k in ['kept','pending','audit']}
    handles={k:gzip.open(p,'wt',encoding='utf-8',compresslevel=1) for k,p in paths.items()}
    try:
        for conv in convs:
            kept,pending,audit,c,rs=process_conversation(conv);counts.update(c);reasons.update(rs);subsets[conv['label']+'/'+conv['split']].update(c)
            for kind,rows in [('kept',kept),('pending',pending),('audit',audit)]:
                for row in rows:handles[kind].write(json.dumps(row,ensure_ascii=False,separators=(',',':'))+'\n')
    finally:
        for f in handles.values():f.close()
    for kind,p in paths.items():p.replace(folder/(kind+'.jsonl.gz'))
    result=dict(batch=number,counts=dict(counts),reasons=dict(reasons),by_subset={k:dict(v) for k,v in subsets.items()})
    save(marker,result);return result

def run_all(workers,limit_batches=None):
    OUT.mkdir(parents=True,exist_ok=True)
    sources=['scan_current_full.py','apply_current_full_rules.py','preview_listen_response_v3.py','preview_turn_rules_v2.py','refine_preview_boundaries_v2.py','preview_duration_v3.py']
    fingerprint={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in sources}
    provenance=OUT/'rule_fingerprint.json'
    if provenance.exists():assert json.loads(provenance.read_text())['sha256']==fingerprint,'Rules changed; use a new output version instead of mixing cached results'
    else:save(provenance,dict(revision=REV,sha256=fingerprint,notes='Raw source audio; no gain/vision; source-global 10ms energy bins. No preview-only 2-35s sampling gate.'))
    convs=json.loads((OUT/'conversations.json').read_text());tasks=[(i//64,convs[i:i+64]) for i in range(0,len(convs),64)]
    if limit_batches is not None:tasks=tasks[:limit_batches]
    cache=Path('/tmp')/('seamless_current_metadata_'+str(os.getuid())+'.sqlite')
    source=OUT/'metadata.sqlite'
    if not cache.exists() or (cache.stat().st_size,cache.stat().st_mtime_ns)!=(source.stat().st_size,source.stat().st_mtime_ns):shutil.copy2(source,cache)
    total=Counter();started=time.time();done=0
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('fork'),initializer=init_worker,initargs=(cache,)) as ex:
        futures=[ex.submit(batch,task) for task in tasks]
        for fut in concurrent.futures.as_completed(futures):
            result=fut.result();total.update(result['counts']);done+=1
            progress=dict(stage='segmentation',done_batches=done,total_batches=len(tasks),counts=dict(total),elapsed_s=time.time()-started)
            save(OUT/'progress.json',progress)
            print('BATCH',done,'/',len(tasks),'conversations',total['conversations'],'kept',total['kept_pairs'],'hours',round(total['kept_seconds']/3600,3),flush=True)
    if limit_batches is None:aggregate()

def aggregate():
    convs=json.loads((OUT/'conversations.json').read_text());expected=math.ceil(len(convs)/64)
    folders=sorted((OUT/'result_shards').glob('[0-9]*'))
    assert len(folders)==expected and all((f/'summary.json').exists() for f in folders),'Incomplete result shards'
    total=Counter();reasons=Counter();subsets=defaultdict(Counter)
    for folder in folders:
        s=json.loads((folder/'summary.json').read_text());total.update(s['counts']);reasons.update(s['reasons'])
        for k,c in s['by_subset'].items():subsets[k].update(c)
    for kind,target in [('kept','kept.jsonl'),('pending','overlong_pending.jsonl')]:
        temp=OUT/(target+'.pending')
        with temp.open('wb') as out:
            for folder in folders:
                with gzip.open(folder/(kind+'.jsonl.gz'),'rb') as f:shutil.copyfileobj(f,out)
        temp.replace(OUT/target)
    # Keep uncertain acoustic endings available without mixing them into the
    # immediately usable automatic-boundary subset.
    clean=OUT/'kept_clean_boundary.jsonl.pending';review=OUT/'boundary_review.jsonl.pending'
    with (OUT/'kept.jsonl').open() as source,clean.open('w') as a,review.open('w') as b:
        for line in source:
            (b if json.loads(line)['segmentation']['boundary_review'] else a).write(line)
    clean.replace(OUT/'kept_clean_boundary.jsonl');review.replace(OUT/'boundary_review.jsonl')
    def stats(c):
        d=dict(c)
        for key in ['kept_seconds','kept_clean_boundary_seconds','kept_boundary_review_seconds','unique_conversation_seconds','unique_listener_seconds']:
            d[key.replace('seconds','hours')]=c.get(key,0)/3600
        return d
    result=dict(revision=REV,complete=True,counts=stats(total),by_subset={k:stats(v) for k,v in sorted(subsets.items())},
        pending_and_review_reasons=dict(reasons),scope=json.loads((OUT/'database_summary.json').read_text()),
        source_inventory=str(OUT/'source_inventory.json'),
        exclusions='No vision/motion/occlusion/sharpness filtering, no audio normalization, no NPZ completeness requirement, no rendering.',
        interpretation='Counts are automatic segmentation candidates, not human-verified training quality. Summed pair hours reuse adjacent turns. Raw-energy boundaries can differ slightly from normalized-preview boundaries.')
    save(OUT/'summary.json',result);print(json.dumps(result,ensure_ascii=False),flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=16);ap.add_argument('--limit-batches',type=int);ap.add_argument('--aggregate',action='store_true');args=ap.parse_args()
    if args.aggregate:aggregate()
    else:run_all(args.workers,args.limit_batches)

if __name__=='__main__':main()
