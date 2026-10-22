"""Check full V3 against immutable V2; write verified clean/review partitions."""
import hashlib,json
from collections import Counter
from itertools import zip_longest
import repair_sentence_starts_v3 as v

def main():
 v.init_db();counts=Counter();seconds=Counter();reasons=Counter();keys=set();maximum=0.;added=0.
 allowed={'revision','start_s','duration_s','speaker_word_start_s','start_padding_s','speaker_words','speaker_start_recovered','speaker_text','speaker_start_evidence','boundary_flags','structure_review_flags','review_flags','status','micro_reactions','micro_listening_count','start_recovery'}
 paths={k:v.OUT/(k+'.jsonl.pending') for k in ['clean','review','unresolved_sentence_start']};handles={k:p.open('w') for k,p in paths.items()}
 try:
  with open(v.SOURCE/'kept.jsonl') as old,open(v.OUT/'kept.jsonl') as new:
   for i,(a,b) in enumerate(zip_longest(old,new),1):
    assert a is not None and b is not None
    before=json.loads(a);r=json.loads(b);key=r['pair_key'];assert key==before['pair_key'] and key not in keys;keys.add(key)
    repaired=r.get('speaker_start_recovered',False)
    if not repaired:assert r==before
    else:
     counts['repaired']+=1
     assert 'speaker_sentence_start_unconfirmed' in before['review_flags'] and 'speaker_sentence_start_unconfirmed' not in r['review_flags']
     assert r['revision']==v.REV and r['end_s']==before['end_s'] and r['start_s']<before['start_s']
     assert all(r[k]==val for k,val in before.items() if k not in allowed)
     prefix=r['start_recovery']['added_words'];assert prefix and r['speaker_words']==prefix+before['speaker_words']
     assert r['speaker_text']==' '.join(w[2] for w in r['speaker_words'])
     assert r['start_s']<=prefix[0][0]+1e-6 and r['speaker_word_start_s']==prefix[0][0]
     assert r['start_recovery']['over_10_seconds']==(r['duration_s']>10)
     assert not r['audio_normalized'] and not r['visual_filtered']
     ww=v.people_for(r)['speaker']['words'];positions=[j for j,w in enumerate(ww) if w==prefix[0]];assert len(positions)==1;j=positions[0]
     assert ww[j:j+len(prefix)]==prefix and (j==0 or v.v2.terminal(ww[j-1][2]))
     assert all(y[0]-x[1]<=v.MAX_ADJACENT_WORD_GAP for x,y in zip(prefix+before['speaker_words'][:1],(prefix+before['speaker_words'][:1])[1:]))
     added+=r['duration_s']-before['duration_s']
    assert 0<r['duration_s']<=(v.MAX_REPAIRED_SECONDS if repaired else 9.95)+1e-6
    assert abs(r['duration_s']-(r['end_s']-r['start_s']))<1e-6
    assert r['status']==('review' if r['review_flags'] else 'clean')
    maximum=max(maximum,r['duration_s']);counts['pairs']+=1;counts[r['status']]+=1;counts['over_10s']+=r['duration_s']>10
    seconds[r['status']]+=r['duration_s'];reasons.update(r['review_flags']);handles[r['status']].write(b)
    if 'speaker_sentence_start_unconfirmed' in r['review_flags']:handles['unresolved_sentence_start'].write(b);counts['unresolved_sentence_start']+=1
    if i%100000==0:print('CHECKED',i,flush=True)
 finally:
  for f in handles.values():f.close()
 for k,p in paths.items():p.replace(v.OUT/(k+'.jsonl'))
 assert counts['pairs']==395038 and counts['repaired']==8161 and counts['unresolved_sentence_start']==48559-8161
 for filename,h in json.loads((v.SOURCE/'rule_fingerprint.json').read_text())['sha256'].items():assert hashlib.sha256((v.ROOT/filename).read_bytes()).hexdigest()==h
 result=dict(passed=True,counts=dict(counts),hours={k:s/3600 for k,s in seconds.items()},added_hours=added/3600,max_duration_s=maximum,review_reasons=dict(reasons),
             checks=['all original pairs retained once','only approved prefix fields changed','listener onset/end unchanged','prefix exactly matches consecutive source words after sentence boundary','unrepaired rows exactly unchanged','12 s limit only for actual repairs','clean/review partition','V2 source code fingerprints unchanged'])
 v.v2.save(v.OUT/'validation.json',result);print('PASS',json.dumps(result),flush=True)
if __name__=='__main__':main()
