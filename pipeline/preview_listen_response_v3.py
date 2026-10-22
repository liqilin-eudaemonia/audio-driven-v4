"""Adjacent-floor listen-response review, with lexical micro-feedback only.

No substantive-insert stitching. All A->B and B->A adjacent pairs are formed
before screening. Existing full-frame/context renderer is shared with v2.
"""
from __future__ import annotations
import argparse
import concurrent.futures
import copy
import json
import random
import re
from collections import Counter
from pathlib import Path

import preview_turn_rules_v2 as media
from build_head_shoulders_preview import refs, read, save
from scan_alternate_pairs import parse_words

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'outputs_alternate_pairs/plain_screen/listen_response_v3_review'
SOURCE=ROOT/'outputs_alternate_pairs/plain_screen/head_shoulders_720p30/manifest.json'
REVISION='listen-response-v3-1'
SEED=20261022
MICRO={'yeah','yes','yep','yup','okay','ok','right','mhm','mm','hmm','uh','huh','oh','sure','exactly','absolutely','ha','haha','hahaha','heh','hehe'}
MICRO_PHRASES={'uh huh','mm hmm','i see','all right','oh yeah','oh okay'}
LAUGH={'ha','haha','hahaha','heh','hehe','laugh','laughs','laughing','laughter'}
TRAILING={'and','but','because','if','the','a','an','to','of','for','with','my','your','are','is','i'}

def tokens(text):return re.findall(r"[a-z]+",text.lower())

def pure_micro(u):
    ts=tokens(u['text'])
    return bool(ts) and len(ts)<=4 and u['end']-u['start']<=1.5 and (all(t in MICRO|LAUGH for t in ts) or ' '.join(ts) in MICRO_PHRASES)

def is_micro(u,f,events,index):
    if not pure_micro(u):return False
    ts=tokens(u['text']);laugh=all(t in LAUGH for t in ts)
    same=next((x for x in events[index+1:] if x['who']==u['who']),None)
    # A short acknowledgment immediately leading into content may be the onset
    # of the response. Don't throw away that prefix as listening feedback.
    if same and 0<=same['start']-u['end']<=1.2 and not pure_micro(same):return False
    owner_next=next((x for x in events[index+1:] if x['who']==f['who']),None)
    future_words=[w for p in f['parts'] for w in p['words'] if u['end']-.05<=w[0]<=u['end']+1.2]
    continues=bool(future_words) or bool(owner_next and -.05<=owner_next['start']-u['end']<=1.0)
    # After the incumbent finishes, even a one-word Yes can be a full answer.
    if not laugh and u['start']>=f['end']-.15:return False
    if laugh and u['start']>f['end']+1.0:return False
    return continues

def floors_from_words(words):
    groups={k:media.raw_utterances(v,k) for k,v in words.items()}
    events=sorted([u for vv in groups.values() for u in vv],key=lambda u:(u['start'],u['end'],u['who']))
    floors=[]
    for i,u in enumerate(events):
        if floors and u['who']==floors[-1]['who']:
            f=floors[-1];f['parts'].append(u);f['end']=max(f['end'],u['end'])
        elif floors and is_micro(u,floors[-1],events,i):
            floors[-1]['micros'].append(dict(u,kind='词表与时序支持的微反应候选'))
        else:
            # Every substantive switch starts a new floor, even if it is brief
            # and the interrupted person continues shortly afterwards.
            floors.append(dict(who=u['who'],start=u['start'],end=u['end'],parts=[u],micros=[]))
    for i,f in enumerate(floors):f['turn_id']=f'T{i+1:03d}'
    return floors,events

def assess(fs,i,words):
    s,l=fs[i:i+2];prev=fs[i-1] if i else None;nxt=fs[i+2] if i+2<len(fs) else None
    reject=[];review=[]
    overlaps=dict(previous=max(0,prev['end']-s['start']) if prev else 0,
                  transition=max(0,s['end']-l['start']),following=max(0,l['end']-nxt['start']) if nxt else 0)
    names={'previous':'输入开头与上一实质话轮重叠','transition':'输入与正式回应存在实质重叠','following':'回应尚未结束，下一实质话轮已开始'}
    for k,v in overlaps.items():
        if v>.2:reject.append(names[k]+f'（{v:.2f}s）')
        elif v>.001:review.append(names[k]+f'（{v:.2f}s，轻微重叠待确认）')
    if sum(p['nwords'] for p in s['parts'])<5 or s['end']-s['start']<1.2:reject.append('speaker不足5词或1.2秒')
    for role,f in [('输入',s),('回应',l)]:
        pauses=[b['start']-a['end'] for a,b in zip(f['parts'],f['parts'][1:])]
        if pauses and max(pauses)>3:review.append(role+'内部停顿超过3秒，是否同一轮需确认')
        last=f['parts'][-1]['text'].rstrip();ts=tokens(last)
        if not last.endswith(('.', '?', '!')) or (ts and ts[-1] in TRAILING):review.append(role+'末尾疑似未完成（仅依据转写，待听原声）')
    suspicious=[dict(who=k,start=a,end=b,text=t) for k,ww in words.items() for a,b,t in ww if b-a>1.5 and a<l['end'] and b>s['start']]
    if suspicious:review.append('存在超过1.5秒的词时间戳，未强行截短，需核对原声')
    return ('discard' if reject else 'review' if review else 'candidate'),reject+review,overlaps,suspicious

def load_conversation(row,number):
    words={};tracks={}
    for who,role in [('A','speaker'),('B','listener')]:
        rr=refs(row[role+'_archive'],row[role+'_id'],['json','mp4','wav'])
        words[who]=parse_words(json.loads(read(rr['json'])))
        tracks[who]=dict(id=row[role+'_id'],archive=row[role+'_archive'],refs=rr)
    fs,events=floors_from_words(words);pairs=[]
    for i,(s,l) in enumerate(zip(fs,fs[1:])):
        status,reasons,overlaps,suspicious=assess(fs,i,words)
        # Both directions: media renderer always puts the input on the left,
        # target on the right; stable A/B identity is retained separately.
        mapping={s['who']:'speaker',l['who']:'listener'}
        ins=[dict(u,who=mapping[u['who']]) for f in [s,l] for u in f['micros']]
        prev=fs[i-1] if i else None;nxt=fs[i+2] if i+2<len(fs) else None
        pairs.append(dict(pair_key=row['interaction']+':'+s['turn_id']+'-'+l['turn_id'],pair_index=i,
            source_sample=number,interaction=row['interaction'],label=row['label'],split=row['split'],
            speaker_person=s['who'],listener_person=l['who'],speaker_turn=s['turn_id'],listener_turn=l['turn_id'],
            tracks={'speaker':tracks[s['who']],'listener':tracks[l['who']]},
            start=s['start'],end=l['end'],response_start=l['start'],speaker_end=s['end'],
            speaker_parts=media.pack(s['parts']),listener_parts=media.pack(l['parts']),inserts=ins,
            status=status,flags=reasons,overlaps=overlaps,suspicious_words=suspicious,
            previous_end=prev['end'] if prev else None,next_start=nxt['start'] if nxt else None,
            old=None,original_roi={},
            transcript=sorted([dict(who=mapping[k],start=a,end=b,text=t) for k,ww in words.items() for a,b,t in ww],key=lambda w:w['start']),
            utterances=[dict(u,who=mapping[u['who']]) for u in events],
            conversation_turns=[dict(turn_id=f['turn_id'],who=f['who'],start=f['start'],end=f['end'],text=' '.join(p['text'] for p in f['parts'])) for f in fs],
            micro_listening_count=len(s['micros']),micro_response_count=len(l['micros'])))
    return dict(source_sample=number,interaction=row['interaction'],floors=fs,pairs=pairs)

def select():
    OUT.mkdir(parents=True,exist_ok=True)
    cache=OUT/'conversations.json'
    if cache.exists():data=json.loads(cache.read_text())
    else:
        rows=json.loads(SOURCE.read_text())['rows'];data=[]
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            for d in pool.map(lambda x:load_conversation(x[1],x[0]+1),enumerate(rows)):
                data.append(d);print('CONVERSATION',d['source_sample'],'pairs',len(d['pairs']),flush=True)
        save(cache,data)
    candidates=[p for d in data for p in d['pairs']]
    def previewable(p):return 2<=p['end']-p['start']<=35
    rng=random.Random(SEED);chosen=[];used=set();session_used=set()
    chains=[(a,b) for d in data for a,b in zip(d['pairs'],d['pairs'][1:]) if a['status']==b['status']=='candidate' and previewable(a) and previewable(b)]
    rng.shuffle(chains)
    chains.sort(key=lambda ab:-sum(x['micro_listening_count'] for x in ab))
    chain_count=0
    for a,b in chains:
        if a['interaction'] in session_used:continue
        chain_count+=1
        for p in [a,b]:
            c=copy.deepcopy(p);c['review_group']='相邻滑动配对';c['chain']=f'C{chain_count}';chosen.append(c);used.add(c['pair_key'])
        session_used.add(a['interaction'])
        if chain_count==4:break
    if chain_count!=4:raise RuntimeError(f'Only {chain_count} clean chains found; revise sample plan explicitly')
    for group,predicate in [
        ('倾听微反应',lambda p:p['status']=='candidate' and p['micro_listening_count']>0),
        ('完整短回应候选',lambda p:p['status']=='candidate' and sum(x['nwords'] for x in p['listener_parts'])<5),
        ('实质插话舍弃示例',lambda p:p['status']=='discard' and (p['overlaps']['transition']>.2 or p['overlaps']['following']>.2) and sum(x['nwords'] for x in p['speaker_parts'])>=5),
    ]:
        eligible=[p for p in candidates if previewable(p) and predicate(p) and p['pair_key'] not in used]
        rng.shuffle(eligible)
        if group=='实质插话舍弃示例':
            # Include the problematic original sample rather than hiding it.
            eligible.sort(key=lambda p:(p['source_sample']!=3,abs(p['start']-136.023)))
        used_local=set();n=0
        for p in eligible:
            if p['interaction'] in used_local:continue
            c=copy.deepcopy(p);c['review_group']=group;c['chain']=None;chosen.append(c);used.add(c['pair_key']);used_local.add(c['interaction']);n+=1
            if n==4:break
        if n<4:print('AVAILABLE',group,n,'(no threshold relaxation)',flush=True)
    for i,c in enumerate(chosen):c['id']=f'{i+1:02d}';c['revision']=REVISION
    summary=dict(revision=REVISION,conversations_read=len(data),all_adjacent_pairs=len(candidates),
        status_counts=dict(Counter(p['status'] for p in candidates)),preview_count=len(chosen),
        preview_groups=dict(Counter(p['review_group'] for p in chosen)),
        all_direction_counts=dict(Counter(p['speaker_person']+'→'+p['listener_person'] for p in candidates)),
        note='Counts refer only to the 30 previously sampled conversations, not the full dataset. Candidate is a rule result, not semantic ground truth.')
    save(OUT/'summary.json',summary)
    save(OUT/'all_pairs_audit.json',[{k:p[k] for k in ['pair_key','source_sample','speaker_person','listener_person','speaker_turn','listener_turn','start','end','response_start','status','flags','overlaps','micro_listening_count']} for p in candidates])
    save(OUT/'selection.json',dict(revision=REVISION,seed=SEED,source=str(SOURCE),rows=chosen))
    print(json.dumps(summary,ensure_ascii=False),flush=True)
    return chosen

def render(c):
    media.OUT=OUT;media.REVISION=REVISION
    r=media.render(c)
    # For candidates, avoid padding into the previous substantive turn. Never
    # truncate the listener's words to hide a real overlap: those pairs reject.
    if r['status']=='candidate' and r['previous_end'] is not None:
        r['crop_start']=max(r['crop_start'],min(r['start'],r['previous_end']+.03))
    r['review_group']=c['review_group'];r['chain']=c['chain']
    save(OUT/'media'/r['id']/'completed.json',r)
    return r

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--render',action='store_true');ap.add_argument('--reselect',action='store_true');args=ap.parse_args()
    rows=json.loads((OUT/'selection.json').read_text())['rows'] if (OUT/'selection.json').exists() and not args.reselect else select()
    if args.render:
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:rows=list(pool.map(render,rows))
        save(OUT/'manifest.json',dict(revision=REVISION,rows=rows))
    print('DONE',len(rows),flush=True)

if __name__=='__main__':main()
