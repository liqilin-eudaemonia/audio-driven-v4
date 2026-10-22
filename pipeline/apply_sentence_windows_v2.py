"""Full raw Seamless: sentence-start speaker suffix + sentence-end response prefix.

One contiguous <=10s window per adjacent pair. Evaluate only retained content.
No models, visual filters, normalization, rendering or synthetic concatenation.
"""
from __future__ import annotations
import argparse, concurrent.futures, gzip, hashlib, json, math, multiprocessing, os, re, shutil, time
from collections import Counter, defaultdict
from bisect import bisect_right
from pathlib import Path
import numpy as np
import apply_current_full_rules as old
from scan_current_full import ROOT, save
from preview_listen_response_v3 import floors_from_words, assess
from preview_duration_v3 import sentences, TRAILING
from refine_preview_boundaries_v2 import db, segments

SOURCE=ROOT/'outputs_current_full_v1'
OUT=ROOT/'outputs_sentence_windows_v2'
REV='sentence-windows-10s-raw-v2'
MAX_SECONDS=9.95
OLD_KEYS=set();OLD_CLEAN=set()


def sentence_marks(ww):
    r=dict(transcript=[dict(who='listener',start=a,end=b,text=t) for a,b,t in ww],
           response_start=ww[0][0],end=max(w[1] for w in ww))
    return sentences(r)[1]


def terminal(text):
    text=text.rstrip().rstrip('\"\'”’)]')
    ts=re.findall(r'[a-z]+',text.lower())
    return bool(re.search(r'[.!?]$',text)) and not text.endswith('...') and text.lower() not in {
        'mr.','mrs.','ms.','dr.','prof.','etc.'} and not (ts and ts[-1] in TRAILING)


def floor_words(f):
    return [list(w) for p in f['parts'] for w in p['words']]


def candidate_windows(fs,i,words,limit,word_ends=None):
    """Propose windows before soft assessments; hard overlaps use retained bounds."""
    s,l=fs[i:i+2];sw=floor_words(s);lw=floor_words(l)
    prev=fs[i-1] if i else None;nxt=fs[i+2] if i+2<len(fs) else None
    if len(sw)<5 or s['end']-s['start']<1.2:return [],'speaker_too_short'
    if s['end']-l['start']>.2:return [],'substantive_transition_overlap'
    ends=[c for c in sentence_marks(lw) if not c['rejection']]
    if not ends:return [],'no_confirmed_listener_sentence_end'
    starts=[(0,None)]+[(c['word_index']+1,c['word_end']) for c in sentence_marks(sw)
                       if not c['rejection'] and c['word_index']+1<len(sw)]
    opts=[];fail=Counter()
    for index,preceding_end in starts:
        a=sw[index][0]
        if len(sw)-index<5 or s['end']-a<1.2:
            fail['retained_speaker_too_short']+=1;continue
        if prev and prev['end']-a>.2:
            fail['substantive_previous_overlap']+=1;continue
        # Keep preceding speech out of pre-roll, including the removed sentence.
        if word_ends is None:word_ends=sorted(b for ww in words.values() for x,b,t in ww)
        position=bisect_right(word_ends,a+1e-7)
        before=word_ends[position-1] if position else 0.
        start=max(0.,a-.2,min(a,before+.03))
        for c in reversed(ends):
            b=c['word_end']
            retained_end=max(w[1] for w in lw[:c['word_index']+1])
            if nxt and retained_end-nxt['start']>.2:
                fail['substantive_following_overlap']+=1;continue
            if a<0 or max(retained_end,s['end'])>limit+1e-6:
                fail['source_time_outside']+=1;continue
            # Potential fits even when nominal pre-roll has to be tightened.
            if max(retained_end,s['end'])-a>MAX_SECONDS+1e-6:
                fail['no_sentence_window_within_10s']+=1;continue
            opts.append(dict(speaker_index=index,speaker_word_start=a,pre_start=start,
                preceding_word_end=before,preceding_sentence_end=preceding_end,
                listener_word_end=b,listener_word_index=c['word_index'],listener_sentence=c,
                speaker_words=sw[index:],listener_words=lw[:c['word_index']+1],
                speaker_trimmed=index>0,response_prefix=not c['full_response']))
    return opts,(fail.most_common(1)[0][0] if fail else 'no_sentence_window') if not opts else None


def retained_flags(fs,i,o,words,long_words=None):
    s,l=fs[i:i+2];a=o['speaker_word_start'];b=o['listener_word_end'];flags=[]
    prev=fs[i-1] if i else None;nxt=fs[i+2] if i+2<len(fs) else None
    for name,value in [('previous_overlap',max(0.,prev['end']-a) if prev else 0.),
                       ('transition_overlap',max(0.,s['end']-l['start'])),
                       ('following_overlap',max(0.,b-nxt['start']) if nxt else 0.)]:
        if value>.2:raise AssertionError('Hard overlap reached retained assessment')
        if value>.001:flags.append(name)
    if not terminal(o['speaker_words'][-1][2]):flags.append('speaker_turn_end_unconfirmed')
    # A substantive interruption may leave a new floor starting mid-sentence.
    if not o['speaker_trimmed']:
        prior=[w for w in words[s['who']] if w[0]<a-1e-7 and w[1]<=a+1e-7]
        if prior and not terminal(prior[-1][2]):flags.append('speaker_sentence_start_unconfirmed')
    for name,ww in [('speaker',o['speaker_words']),('listener',o['listener_words'])]:
        if any(v[0]-u[1]>3 for u,v in zip(ww,ww[1:])):flags.append(name+'_pause_over_3s')
    if any(y-x>1.5 and x<b and y>a for ww in (long_words or words).values() for x,y,t in ww):flags.append('word_timestamp_over_1_5s')
    if max(w[1] for w in o['listener_words'])>b+1e-5:flags.append('listener_word_timestamp_order_conflict')
    return flags


def quiet_points(times,levels,thresholds,lower,upper,nominal):
    if lower>upper:return None
    choices=[]
    for a,b in segments(np.all(levels<=thresholds,axis=1)):
        if (b-a)*.01<.06:continue
        lo=max(lower,float(times[a]+.03));hi=min(upper,float(times[b-1]-.03))
        if lo<=hi:choices.append(float(np.clip(nominal,lo,hi)))
    return min(choices,key=lambda x:abs(x-nominal)) if choices else None


def cut_window(o,fs,i,words,rms,masks,limit):
    s,l=fs[i:i+2];a=o['speaker_word_start'];b=o['listener_word_end']
    content_end=max(w[1] for w in o['speaker_words']+o['listener_words'])
    first=max(0,math.ceil((max(0,a-3)-.005)/.01));last=min(min(map(len,rms.values())),math.ceil((min(limit,b+3)-.005)/.01))
    times=(np.arange(first,last)+.5)*.01;roles=[s['who'],l['who']]
    values=np.stack([rms[k][first:last] for k in roles],axis=1)
    levels=db(values);thresholds=[];flags=[]
    for ch,k in enumerate(roles):
        active=masks[k][first:last]&(times>=a)&(times<b)
        speech=values[active,ch]
        if not len(speech):return None,'no_speech_energy_reference'
        ref=float(db(np.sqrt(np.mean(speech**2))));floor=float(np.percentile(levels[:,ch],15))
        thresholds.append(max(-80,min(ref-12,max(ref-24,floor+6,-70))))
    thresholds=np.array(thresholds)
    # Select onset before the first retained speaker word, after earlier words.
    lower=max(0.,a-.35,min(a,o['preceding_word_end']+.03))
    upper=a-.03;nominal=o['pre_start']
    for x,y in segments((times>=a-.2)&(times<=a+.10)&(levels[:,0]>thresholds[0]+3)):
        if (y-x)*.01>=.04:
            upper=min(upper,float(times[x]-.005-.03));break
    start=quiet_points(times,levels,thresholds,lower,upper,nominal)
    if start is None:start=nominal;flags.append('start_no_quiet_cut')
    # Tighten pre-roll only, never delete the first retained word.
    if content_end-start>MAX_SECONDS:
        start=quiet_points(times,levels,thresholds,max(lower,content_end-MAX_SECONDS),upper,max(nominal,content_end-MAX_SECONDS))
        if start is None:
            start=max(nominal,content_end-MAX_SECONDS)
            if start>a+1e-7:return None,'no_sentence_window_within_10s'
            flags.append('start_preroll_tightened_without_quiet_cut')
    context_words=[(k,x,y,t) for k,ww in words.items() for x,y,t in ww if y>=start-.5 and x<=b+1.6]
    # A cut must not silently bisect an annotated micro-reaction or previous word.
    if any(x<start-1e-6 and y>start+1e-6 for _,x,y,_ in context_words):flags.append('start_crosses_annotated_voice')
    following=[x for _,x,y,_ in context_words if x>=b-1e-5 and y>b+1e-5]
    nxt=min(following) if following else None
    guard=nxt
    if nxt is not None:
        next_roles={k for k,x,y,t in context_words if abs(x-nxt)<1e-5 and y>b+1e-5}
        active=np.zeros(len(times),bool)
        for ch,k in enumerate(roles):
            if k in next_roles:active|=levels[:,ch]>thresholds[ch]+3
        active&=(times>=nxt-.2)&(times<=nxt+.15)
        for x,y in segments(active):
            if (y-x)*.01>=.04:guard=min(nxt,float(times[x]-.005));break
    upper=min(limit,b+1.5,start+MAX_SECONDS)
    if guard is not None:upper=min(upper,guard-.08)
    lower=max(b+.12,content_end+.08)
    crossing=[y for _,x,y,_ in context_words if x<b-1e-5 and y>b+1e-5]
    if crossing:lower=max(lower,max(crossing)+.08)
    nominal=min(b+.8,upper)
    for x,y in segments((times>=b)&(times<=nominal)&(levels[:,1]>thresholds[1]+3)):
        if (y-x)*.01>=.04:lower=max(lower,float(times[y-1]+.005+.06))
    end=quiet_points(times,levels,thresholds,lower,upper,nominal)
    if end is None:
        end=max(content_end,nominal);flags.append('end_no_quiet_cut')
        if upper<b+.12:flags.append('insufficient_tail_room')
    if end>limit+1e-6 or end-start>MAX_SECONDS+1e-6 or end<content_end-1e-6:return None,'invalid_boundary_budget'
    if any(x<end-1e-6 and y>end+1e-6 for _,x,y,_ in context_words):flags.append('end_crosses_annotated_voice')
    if nxt is not None and end>nxt-.05:flags.append('next_voice_guard_too_close')
    return dict(start_s=round(start,6),end_s=round(end,6),duration_s=round(end-start,6),
        start_padding_s=round(a-start,6),tail_s=round(end-b,6),next_voice_start_s=nxt,
        next_voice_guard_s=guard,
        quiet_threshold_dbfs=thresholds.tolist(),boundary_flags=sorted(set(flags))),None


def make_output(conv,fs,i,o,cut,flags,people,previous_status):
    s,l=fs[i:i+2];key='/'.join([conv['label'],conv['split'],conv['interaction'],s['turn_id']+'-'+l['turn_id']])
    micros=[dict(who=u['who'],start=u['start'],end=u['end'],text=u['text']) for f in [s,l] for u in f['micros']
            if u['start']<cut['end_s'] and u['end']>cut['start_s']]
    review=sorted(set(flags+cut['boundary_flags']))
    return dict(revision=REV,pair_key=key,interaction=conv['interaction'],label=conv['label'],split=conv['split'],
        speaker_id=people[s['who']]['id'],listener_id=people[l['who']]['id'],speaker_person=s['who'],listener_person=l['who'],
        speaker_turn=s['turn_id'],listener_turn=l['turn_id'],**cut,
        speaker_word_start_s=o['speaker_word_start'],speaker_end_s=s['end'],response_start_s=l['start'],
        response_word_end_s=o['listener_word_end'],original_speaker_start_s=s['start'],original_response_end_s=l['end'],
        speaker_trimmed=o['speaker_trimmed'],response_prefix=o['response_prefix'],
        speaker_text=' '.join(w[2] for w in o['speaker_words']),listener_text=' '.join(w[2] for w in o['listener_words']),
        speaker_words=o['speaker_words'],listener_words=o['listener_words'],
        speaker_start_evidence=dict(kind='after_sentence_terminal' if o['speaker_trimmed'] else 'substantive_turn_start',
                                    preceding_sentence_end=o['preceding_sentence_end']),
        listener_end_evidence=o['listener_sentence']['text'],
        review_flags=review,status='review' if review else 'clean',structure_review_flags=flags,
        micro_reactions=micros,micro_listening_count=sum(u['who']==l['who'] and u['start']<l['start'] for u in micros),
        previous_full_structure_status=previous_status,previous_kept=key in OLD_KEYS,previous_clean=key in OLD_CLEAN,
        source={role:dict(id=people[f['who']]['id'],refs=people[f['who']]['refs'],
            video_duration=people[f['who']]['video_duration'],audio_duration=people[f['who']]['wav']['duration']) for role,f in [('speaker',s),('listener',l)]},
        audio_normalized=False,visual_filtered=False,media_rendered=False)


def process_conversation(conv):
    people={who:old.load_person(key) for who,key in zip(['A','B'],conv['keys'])};words={k:p['words'] for k,p in people.items()}
    counts=Counter(conversations=1);reasons=Counter();kept=[];audit=[];pending=[]
    if not all(words.values()):counts['empty_transcript_conversations']=1;return kept,pending,audit,counts,reasons
    fs,_=floors_from_words(words);counts['all_adjacent_pairs']=max(0,len(fs)-1)
    limit=min(min(p['video_duration'],p['wav']['duration']) for p in people.values())
    long_words={k:[w for w in ww if w[1]-w[0]>1.5] for k,ww in words.items()};possible=[]
    word_ends=sorted(b for ww in words.values() for a,b,t in ww)
    for i in range(len(fs)-1):
        s,l=fs[i:i+2];key='/'.join([conv['label'],conv['split'],conv['interaction'],s['turn_id']+'-'+l['turn_id']])
        status,_,_,_=assess(fs,i,long_words)
        base=dict(pair_key=key,old_structure_status=status,label=conv['label'],split=conv['split'])
        opts,why=candidate_windows(fs,i,words,limit,word_ends)
        if not opts:
            counts['no_window_pairs']+=1;reasons[why]+=1;base.update(status='no_window',reason=why);audit.append(base);pending.append(base)
        else:possible.append((i,opts,base));counts['transcript_window_candidates']+=1
    if possible:
        try:
            rms={k:old.pcm_rms(p) for k,p in people.items()};masks={k:old.word_mask(words[k],len(rms[k])) for k in people}
            counts['audio_read_conversations']=1;counts['audio_data_bytes']=sum(p['wav']['data_bytes'] for p in people.values())
        except Exception as e:
            counts['audio_error_pairs']+=len(possible)
            for i,opts,base in possible:base.update(status='audio_error',reason=repr(e));audit.append(base)
            return kept,pending,audit,counts,reasons
        for i,opts,base in possible:
            best=None;failure=Counter()
            try:
                for o in opts:
                    flags=retained_flags(fs,i,o,words,long_words)
                    cut,why=cut_window(o,fs,i,words,rms,masks,limit)
                    if cut is None:failure[why]+=1;continue
                    result=make_output(conv,fs,i,o,cut,flags,people,base['old_structure_status'])
                    if best is None:best=result
                    if not result['review_flags']:best=result;break
                if best is None:
                    counts['no_cut_pairs']+=1;why=failure.most_common(1)[0][0] if failure else 'no_cut'
                    reasons[why]+=1;base.update(status='no_cut',reason=why);audit.append(base);pending.append(base);continue
            except Exception as e:
                counts['processing_error_pairs']+=1;base.update(status='processing_error',reason=repr(e));audit.append(base);continue
            kept.append(best);counts['kept_pairs']+=1;counts['kept_seconds']+=best['duration_s']
            category=best['status'];counts[category+'_pairs']+=1;counts[category+'_seconds']+=best['duration_s']
            counts['speaker_trimmed']+=best['speaker_trimmed'];counts['response_prefix']+=best['response_prefix']
            counts['both_trimmed']+=best['speaker_trimmed'] and best['response_prefix']
            counts['with_listening_micro']+=bool(best['micro_listening_count'])
            counts['from_old_'+base['old_structure_status']]+=1
            counts['previous_kept_retained']+=best['previous_kept'];counts['previous_clean_retained']+=best['previous_clean']
            counts['new_pairs_vs_v1']+=not best['previous_kept']
            counts['duration_'+('under2' if best['duration_s']<2 else '2to4' if best['duration_s']<4 else '4to6' if best['duration_s']<6 else '6to8' if best['duration_s']<8 else '8to10')]+=1
            reasons.update('kept_review:'+x for x in best['review_flags'])
            base.update(status=category,start_s=best['start_s'],end_s=best['end_s'],duration_s=best['duration_s']);audit.append(base)
    if kept:
        counts['conversations_with_kept']=1
        counts['unique_conversation_seconds']=old.merge_seconds([(r['start_s'],r['end_s']) for r in kept])
        counts['unique_listener_seconds']=sum(old.merge_seconds([(r['start_s'],r['end_s']) for r in kept if r['listener_id']==p['id']]) for p in people.values())
    return kept,pending,audit,counts,reasons


def batch(task):
    number,convs=task;folder=OUT/'result_shards'/f'{number:05d}';marker=folder/'summary.json'
    if marker.exists():return json.loads(marker.read_text())
    folder.mkdir(parents=True,exist_ok=True);counts=Counter();reasons=Counter();subsets=defaultdict(Counter)
    handles={k:gzip.open(folder/(k+'.jsonl.gz.pending'),'wt',encoding='utf-8',compresslevel=1) for k in ['kept','pending','audit']}
    try:
        for conv in convs:
            kept,pending,audit,c,rs=process_conversation(conv);counts.update(c);reasons.update(rs);subsets[conv['label']+'/'+conv['split']].update(c)
            for kind,rows in [('kept',kept),('pending',pending),('audit',audit)]:
                for r in rows:handles[kind].write(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n')
    finally:
        for f in handles.values():f.close()
    for kind in handles:(folder/(kind+'.jsonl.gz.pending')).replace(folder/(kind+'.jsonl.gz'))
    result=dict(batch=number,counts=dict(counts),reasons=dict(reasons),by_subset={k:dict(v) for k,v in subsets.items()})
    save(marker,result);return result


def aggregate():
    convs=json.loads((SOURCE/'conversations.json').read_text());folders=sorted((OUT/'result_shards').glob('[0-9]*'))
    assert len(folders)==math.ceil(len(convs)/64) and all((f/'summary.json').exists() for f in folders)
    counts=Counter();reasons=Counter();subsets=defaultdict(Counter)
    paths={k:OUT/(k+'.jsonl.pending') for k in ['kept','clean','review','pending']}
    handles={k:p.open('w') for k,p in paths.items()}
    try:
        for folder in folders:
            r=json.loads((folder/'summary.json').read_text());counts.update(r['counts']);reasons.update(r['reasons'])
            for k,c in r['by_subset'].items():subsets[k].update(c)
            with gzip.open(folder/'kept.jsonl.gz','rt') as f:
                for line in f:
                    row=json.loads(line);handles['kept'].write(line);handles[row['status']].write(line)
            with gzip.open(folder/'pending.jsonl.gz','rt') as f:shutil.copyfileobj(f,handles['pending'])
    finally:
        for f in handles.values():f.close()
    for k,p in paths.items():p.replace(OUT/(k+'.jsonl'))
    def stats(c):
        d=dict(c)
        for k,v in c.items():
            if k.endswith('_seconds'):d[k[:-8]+'_hours']=v/3600
        return d
    result=dict(revision=REV,complete=True,counts=stats(counts),by_subset={k:stats(v) for k,v in sorted(subsets.items())},
        reasons=dict(reasons),source_inventory=str(SOURCE/'source_inventory.json'),source_metadata=str(SOURCE/'metadata.sqlite'),
        source_conversations=len(convs),note='One window per pair. Soft review is retained separately. No visual/audio normalization filters. Summed hours reuse adjacent turns.')
    save(OUT/'summary.json',result);print('COMPLETE',json.dumps(result,ensure_ascii=False),flush=True)


def run(workers,limit_batches=None):
    global OLD_KEYS,OLD_CLEAN
    OUT.mkdir(parents=True,exist_ok=True)
    names=['apply_sentence_windows_v2.py','apply_current_full_rules.py','scan_current_full.py','scan_alternate_pairs.py',
           'preview_listen_response_v3.py','preview_turn_rules_v2.py','preview_duration_v3.py','refine_preview_boundaries_v2.py']
    hashes={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in names}
    marker=OUT/'rule_fingerprint.json'
    if marker.exists():assert json.loads(marker.read_text())['sha256']==hashes,'Do not mix changed rules with completed batches'
    else:save(marker,dict(revision=REV,sha256=hashes))
    with (SOURCE/'kept.jsonl').open() as f:
        for line in f:
            r=json.loads(line);OLD_KEYS.add(r['pair_key'])
            if not r['segmentation']['boundary_review']:OLD_CLEAN.add(r['pair_key'])
    convs=json.loads((SOURCE/'conversations.json').read_text());tasks=[(i//64,convs[i:i+64]) for i in range(0,len(convs),64)]
    if limit_batches is not None:tasks=tasks[:limit_batches]
    cache=Path('/tmp')/('seamless_current_metadata_'+str(os.getuid())+'.sqlite');src=SOURCE/'metadata.sqlite'
    if not cache.exists() or (cache.stat().st_size,cache.stat().st_mtime_ns)!=(src.stat().st_size,src.stat().st_mtime_ns):shutil.copy2(src,cache)
    started=time.time();counts=Counter();done=0
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('fork'),initializer=old.init_worker,initargs=(cache,)) as ex:
        futures=[ex.submit(batch,t) for t in tasks]
        for fut in concurrent.futures.as_completed(futures):
            r=fut.result();counts.update(r['counts']);done+=1
            save(OUT/'progress.json',dict(done_batches=done,total_batches=len(tasks),elapsed_s=time.time()-started,counts=dict(counts)))
            print('BATCH',done,'/',len(tasks),'conversations',counts['conversations'],'kept',counts['kept_pairs'],'clean',counts['clean_pairs'],'hours',round(counts['kept_seconds']/3600,3),flush=True)
    if limit_batches is None:aggregate()


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=48);ap.add_argument('--limit-batches',type=int);ap.add_argument('--aggregate',action='store_true');args=ap.parse_args()
    if args.aggregate:aggregate()
    else:run(args.workers,args.limit_batches)
