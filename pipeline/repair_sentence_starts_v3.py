"""Repair sentence starts after V2 selection, preserving pair identities and ends.

V2 is immutable. This required post-selection stage may extend only repaired
windows to 11.95 s (12 s including encoding allowance). No new models.
"""
from __future__ import annotations
import argparse, copy, gzip, hashlib, json, math, sqlite3
from collections import Counter
from functools import lru_cache
from pathlib import Path
import numpy as np
import apply_sentence_windows_v2 as v2
from preview_turn_rules_v2 import raw_utterances
from preview_listen_response_v3 import is_micro

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/'outputs_sentence_windows_v2'
OUT=ROOT/'outputs_sentence_start_repair_v3'
REV='sentence-start-repair-v3'
MAX_REPAIRED_SECONDS=11.95
MAX_ADJACENT_WORD_GAP=3.0
START_FLAGS={'start_no_quiet_cut','start_crosses_annotated_voice','start_preroll_tightened_without_quiet_cut'}
DB=None

def init_db():
 global DB
 DB=sqlite3.connect('file:'+str(ROOT/'outputs_current_full_v1/metadata.sqlite')+'?mode=ro',uri=True)

@lru_cache(maxsize=32)
def person(key):
 row=DB.execute('SELECT payload FROM people WHERE key=?',(key,)).fetchone()
 if row is None:raise KeyError(key)
 return json.loads(gzip.decompress(row[0]))

def people_for(r):return {role:person(f"{r['label']}/{r['split']}/{r[role+'_id']}") for role in ['speaker','listener']}

def propose(r,people):
 """Return a complete sentence prefix or an explicit reason to leave untouched.

Never jump a long gap or substantive speech, never attach just a partial prefix,
never infer missing words from the transcript text. The 12 s allowance applies
only to repairs; listener start/end and original pair identity remain fixed.
 """
 if 'speaker_sentence_start_unconfirmed' not in r['review_flags']:return None,'not_target'
 ww=people['speaker']['words'];first=r['speaker_words'][0]
 indices=[i for i,w in enumerate(ww) if abs(w[0]-first[0])<1e-5 and abs(w[1]-first[1])<1e-5 and w[2]==first[2]]
 if len(indices)!=1:return None,'ambiguous_first_word'
 j=indices[0]
 if not j:return None,'no_preceding_words'
 if v2.terminal(ww[j-1][2]):return None,'preceding_word_already_terminal'
 k=j
 while k>0 and not v2.terminal(ww[k-1][2]):
  prev,cur=ww[k-1],ww[k]
  if cur[0]-prev[1]>MAX_ADJACENT_WORD_GAP:return None,'discontinuous_previous_words'
  if prev[1]>cur[0]+.001 or prev[0]>=cur[0] or prev[1]<=prev[0]:return None,'word_timestamp_conflict'
  k-=1
  if r['end_s']-ww[k][0]>MAX_REPAIRED_SECONDS:return None,'complete_prefix_exceeds_12s'
 prefix=[list(w) for w in ww[k:j]]
 if not prefix:return None,'empty_prefix'
 a=prefix[0][0];allwords={role:p['words'] for role,p in people.items()}
 if any(b-a0>1.5 and a0<first[0] and b>a for w in allwords.values() for a0,b,t in w):return None,'prefix_word_timestamp_over_1_5s'
 # Examine actual utterance intervals, not the previously misordered floors.
 # A substantive utterance that ended before the recovered sentence is allowed.
 # Recognized microfeedback may remain inside the recovered sentence.
 events=sorted([u for role,words in allwords.items() for u in raw_utterances(words,role)],key=lambda u:(u['start'],u['end'],u['who']))
 owner=dict(who='speaker',start=a,end=r['speaker_end_s'],parts=[dict(words=prefix+r['speaker_words'])])
 micros=[];previous_end=0.
 for i,u in enumerate(events):
  if u['who']!='listener':continue
  if u['end']<=a+1e-7:previous_end=max(previous_end,u['end']);continue
  if u['start']>=first[0]-1e-7:continue
  if is_micro(u,owner,events,i):
   micros.append(dict(who=r['listener_person'],start=u['start'],end=u['end'],text=u['text']))
  else:return None,'substantive_voice_in_recovered_prefix'
 all_ends=[b for words in allwords.values() for x,b,t in words if b<=a+1e-7]
 before=max(all_ends,default=0.)
 nominal=max(0.,a-.2,min(a,before+.03))
 if r['end_s']-nominal>MAX_REPAIRED_SECONDS:return None,'complete_prefix_and_preroll_exceed_12s'
 return dict(words=prefix,speaker_word_start_s=a,nominal_start_s=nominal,preceding_word_end_s=before,
             preceding_sentence_end_s=ww[k-1][1] if k else None,micro_reactions=micros,
             previous_substantive_end_s=previous_end,
             proposed_duration_s=round(r['end_s']-nominal,6),added_content_s=round(first[0]-a,6)), 'repairable'

def rms_slice(p,first,last):
 """Read only the needed source-global 10 ms PCM bins directly from tar."""
 w=p['wav'];sr=w['samplerate'];channels=w['channels'];bits=w['bits'];kind=w['format'];width=bits//8
 assert sr%100==0
 frame=sr//100;count=last-first
 with open(p['refs']['wav']['archive'],'rb') as f:
  f.seek(w['data_offset']+first*frame*channels*width);raw=f.read(count*frame*channels*width)
 if len(raw)!=count*frame*channels*width:raise ValueError('Truncated source PCM')
 if kind==3 and bits in [32,64]:a=np.frombuffer(raw,dtype='<f'+str(width)).astype(np.float64)
 elif kind==1 and bits==24:
  b=np.frombuffer(raw,np.uint8).reshape(-1,3).astype(np.int32);a=b[:,0]|b[:,1]<<8|b[:,2]<<16;a=((a^0x800000)-0x800000).astype(np.float64)/8388608.
 elif kind==1 and bits in [8,16,32]:
  a=np.frombuffer(raw,dtype='u1' if bits==8 else '<i'+str(width)).astype(np.float64)
  if bits==8:a-=128
  a/=2**(bits-1)
 else:raise ValueError('Unsupported PCM')
 if channels>1:a=a.reshape(-1,channels).mean(axis=1)
 if not np.isfinite(a).all():raise ValueError('Non-finite source PCM')
 return np.sqrt(np.mean(a.reshape(count,frame)**2,axis=1))

def apply_repair(r,plan,people):
 a=plan['speaker_word_start_s'];limit=min(min(p['video_duration'],p['wav']['duration']) for p in people.values())
 first=max(0,math.ceil((max(0,a-1)-.005)/.01));last=min(min(p['wav']['frames']//(p['wav']['samplerate']//100) for p in people.values()),math.ceil((min(limit,a+.5)-.005)/.01))
 times=(np.arange(first,last)+.5)*.01
 levels=v2.db(np.stack([rms_slice(people[role],first,last) for role in ['speaker','listener']],axis=1))
 thresholds=np.array(r['quiet_threshold_dbfs']) # Preserve V2 end-boundary calibration.
 lower=max(0.,a-.35,min(a,plan['preceding_word_end_s']+.03),r['end_s']-MAX_REPAIRED_SECONDS)
 upper=a-.03;nominal=plan['nominal_start_s'];flags=[]
 for x,y in v2.segments((times>=a-.2)&(times<=a+.1)&(levels[:,0]>thresholds[0]+3)):
  if (y-x)*.01>=.04:upper=min(upper,float(times[x]-.005-.03));break
 start=v2.quiet_points(times,levels,thresholds,lower,upper,nominal)
 if start is None:start=nominal;flags.append('start_no_quiet_cut')
 if any(x<start-1e-6<y-1e-6 for p in people.values() for x,y,t in p['words']):flags.append('start_crosses_annotated_voice')
 out=copy.deepcopy(r);out.update(revision=REV,start_s=round(start,6),duration_s=round(r['end_s']-start,6),speaker_word_start_s=a,
   start_padding_s=round(a-start,6),speaker_words=plan['words']+r['speaker_words'],speaker_start_recovered=True)
 out['speaker_text']=' '.join(w[2] for w in out['speaker_words'])
 out['speaker_start_evidence']=dict(kind='recovered_same_sentence_prefix',preceding_sentence_end=plan['preceding_sentence_end_s'],original_word_start_s=r['speaker_word_start_s'])
 out['boundary_flags']=sorted(set(r['boundary_flags'])-START_FLAGS|set(flags))
 structure=set(r['structure_review_flags'])-{'speaker_sentence_start_unconfirmed','previous_overlap','speaker_pause_over_3s'}
 if any(b[0]-a0[1]>3 for a0,b in zip(out['speaker_words'],out['speaker_words'][1:])):structure.add('speaker_pause_over_3s')
 out['structure_review_flags']=sorted(structure);out['review_flags']=sorted(structure|set(out['boundary_flags']));out['status']='review' if out['review_flags'] else 'clean'
 micro={(u['who'],u['start'],u['end'],u['text']):u for u in r['micro_reactions']+plan['micro_reactions']}
 out['micro_reactions']=sorted(micro.values(),key=lambda u:u['start']);out['micro_listening_count']=sum(u['who']==r['listener_person'] and u['start']<r['response_start_s'] for u in out['micro_reactions'])
 out['start_recovery']=dict(status='repaired',old_start_s=r['start_s'],old_speaker_word_start_s=r['speaker_word_start_s'],old_duration_s=r['duration_s'],
   old_review_flags=r['review_flags'],added_words=plan['words'],added_seconds=round(r['start_s']-start,6),max_allowed_seconds=MAX_REPAIRED_SECONDS,
   over_10_seconds=out['duration_s']>10,over_10_seconds_by=round(max(0,out['duration_s']-10),6),quiet_threshold_policy='reuse V2 thresholds; recalculate start only')
 assert 0<=out['start_s']<=a+1e-6 and out['duration_s']<=MAX_REPAIRED_SECONDS+1e-6 and out['end_s']==r['end_s']
 return out

def process(r,audio=True):
 if 'speaker_sentence_start_unconfirmed' not in r['review_flags']:return r,'not_target'
 people=people_for(r);plan,why=propose(r,people)
 if plan is None:return r,why
 if not audio:return plan,why
 return apply_repair(r,plan,people),'repaired'

def fingerprint(folder):
 names=['repair_sentence_starts_v3.py','apply_sentence_windows_v2.py','preview_listen_response_v3.py','preview_turn_rules_v2.py']
 value=dict(revision=REV,sha256={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in names},max_repaired_seconds=MAX_REPAIRED_SECONDS,max_word_gap=MAX_ADJACENT_WORD_GAP)
 p=folder/'rule_fingerprint.json'
 if p.exists():assert json.loads(p.read_text())==value,'Output contains a different rule revision'
 else:v2.save(p,value)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['preview','audit','apply']);args=ap.parse_args();init_db();OUT.mkdir(exist_ok=True);fingerprint(OUT)
 if args.mode=='preview':
  data=json.loads((SOURCE/'review_preview_v1/manifest.json').read_text());results=[]
  for r in data['rows']:
   if r['sample_group']!='speaker_sentence_start_unconfirmed':continue
   new,why=process(r);new=copy.deepcopy(new);new['repair_outcome']=why;new['before']=r;results.append(new)
   print(r['id'],why,new['speaker_text'],new['duration_s'],flush=True)
  v2.save(OUT/'sample_results.json',dict(revision=REV,rows=results))
 else:
  counts=Counter();added=0.;outfile=OUT/('audit.jsonl' if args.mode=='audit' else 'kept.jsonl');temp=outfile.with_suffix('.jsonl.pending')
  with open(SOURCE/'kept.jsonl') as source,open(temp,'w') as dest:
   for line in source:
    r=json.loads(line);new,why=process(r,audio=args.mode=='apply');counts['source_pairs']+=1;counts[why]+=1
    if args.mode=='apply':dest.write(json.dumps(new,ensure_ascii=False,separators=(',',':'))+'\n')
    elif why!='not_target':dest.write(json.dumps(dict(pair_key=r['pair_key'],reason=why,plan=new if why=='repairable' else None),ensure_ascii=False,separators=(',',':'))+'\n')
    if why in ['repairable','repaired']:
     duration=new['proposed_duration_s'] if args.mode=='audit' else new['duration_s'];counts['over_10s']+=duration>10;added+=duration-r['duration_s']
    if counts['source_pairs']%10000==0:print(dict(counts),flush=True)
  temp.replace(outfile);v2.save(OUT/(args.mode+'_summary.json'),dict(revision=REV,complete=True,counts=dict(counts),added_hours=added/3600,mode=args.mode,source=str(SOURCE/'kept.jsonl')));print('COMPLETE',dict(counts),flush=True)
if __name__=='__main__':main()
