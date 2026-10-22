"""Behavioral checks and full-manifest accounting for sentence-window cutting."""
import argparse,gzip,hashlib,json
from collections import Counter
import numpy as np
import apply_sentence_windows_v2 as v
from scan_current_full import save


def phrase(text,start,step=.45):
    return [[start+i*step,start+i*step+.3,t] for i,t in enumerate(text.split())]


def floor(ww,who,n):
    return dict(who=who,start=ww[0][0],end=ww[-1][1],turn_id=f'T{n:03d}',micros=[],
                parts=[dict(start=ww[0][0],end=ww[-1][1],text=' '.join(w[2] for w in ww),nwords=len(ww),words=ww)])


def unit_checks():
    checks=[]
    a=phrase('We talked about many things before.',0)+phrase('I would like to hear your answer.',12)
    b=phrase('I think that sounds very good.',16)
    fs=[floor(a,'A',1),floor(b,'B',2)];ww={'A':a,'B':b}
    opts,why=v.candidate_windows(fs,0,ww,30)
    assert opts and all(o['speaker_trimmed'] and o['speaker_word_start']==12 for o in opts)
    assert all('speaker_pause_over_3s' not in v.retained_flags(fs,0,o,ww) for o in opts)
    checks.append('long speaker rescued only at a sentence start; removed pause not assessed')
    a=phrase('Tell me what you think please.',0)
    b=phrase('That sounds good.',4)+phrase('Let me also explain another related point.',8)
    c=phrase('I want to tell you something else.',9)
    fs=[floor(a,'A',1),floor(b,'B',2),floor(c,'A',3)];ww={'A':a+c,'B':b}
    assert v.assess(fs,0,ww)[0]=='discard'
    opts,why=v.candidate_windows(fs,0,ww,30)
    assert opts and all(o['response_prefix'] and o['listener_word_end']<9 for o in opts)
    checks.append('listener prefix recovered before original later substantive overlap')
    bad=phrase('I would like to respond now.',1)
    assert not v.candidate_windows([floor(a,'A',1),floor(bad,'B',2)],0,{'A':a,'B':bad},30)[0]
    checks.append('real retained transition overlap is not admitted')
    no_end=phrase('I cannot finish my thought because',4)
    assert v.candidate_windows([floor(a,'A',1),floor(no_end,'B',2)],0,{'A':a,'B':no_end},30)[1]=='no_confirmed_listener_sentence_end'
    checks.append('no arbitrary 10-second or unconfirmed listener sentence cut')
    b=phrase('That sounds good.',4);a=phrase('Tell me what you think please.',1)
    fs=[floor(a,'A',1),floor(b,'B',2)];ww={'A':a,'B':b};opts,_=v.candidate_windows(fs,0,ww,20)
    rms={k:np.full(2000,1e-6) for k in ww};masks={k:v.old.word_mask(w,2000) for k,w in ww.items()}
    for k in ww:rms[k][masks[k]]=.1
    cut,why=v.cut_window(opts[0],fs,0,ww,rms,masks,20)
    assert cut and not cut['boundary_flags'],cut
    assert 0<cut['duration_s']<=9.95 and cut['start_s']<=1 and cut['end_s']>=b[-1][1]+.12
    checks.append('quiet acoustic boundaries preserve first word and listener tail')
    # Insert an unselected later word crossing an otherwise valid cut; it must
    # remain reviewable, never become a clean word-bisecting output.
    ww['A'].append([b[-1][1]-.05,b[-1][1]+.2,'yeah'])
    masks={k:v.old.word_mask(w,2000) for k,w in ww.items()}
    cut,_=v.cut_window(opts[0],fs,0,ww,rms,masks,20)
    assert cut and (cut['end_s']>=ww['A'][-1][1] or cut['boundary_flags'])
    checks.append('crossing micro voice is protected or explicitly flagged')
    result=dict(passed=True,checks=checks);save(v.OUT/'unit_validation.json',result);print(json.dumps(result),flush=True)


def validate_full():
    summary=json.loads((v.OUT/'summary.json').read_text());c=summary['counts'];assert summary['complete']
    assert c['conversations']==summary['source_conversations']==64638
    assert c['all_adjacent_pairs']==1931463
    assert c['all_adjacent_pairs']==sum(c.get(k,0) for k in ['no_window_pairs','no_cut_pairs','audio_error_pairs','processing_error_pairs','kept_pairs'])
    assert c['kept_pairs']==c['clean_pairs']+c['review_pairs']
    keys=set();clean_keys=set();counts=Counter();seconds=Counter();by_subset=Counter();max_d=0
    old_keys=set();old_clean=set()
    with (v.SOURCE/'kept.jsonl').open() as f:
        for line in f:
            r=json.loads(line);old_keys.add(r['pair_key'])
            if not r['segmentation']['boundary_review']:old_clean.add(r['pair_key'])
    with (v.OUT/'kept.jsonl').open() as f:
        for line in f:
            r=json.loads(line);key=r['pair_key'];assert key not in keys;keys.add(key)
            assert key.startswith('/'.join([r['label'],r['split'],r['interaction']])+'/')
            assert r['previous_kept']==(key in old_keys) and r['previous_clean']==(key in old_clean)
            a,b=r['start_s'],r['end_s'];d=r['duration_s'];max_d=max(max_d,d)
            assert 0<=a<=r['speaker_word_start_s']+1e-6<r['response_start_s']<=r['response_word_end_s']<=b+1e-6
            assert 0<d<=9.950001 and abs(b-a-d)<1.1e-6
            assert len(r['speaker_words'])>=5 and r['speaker_end_s']-r['speaker_word_start_s']>=1.2-1e-6
            assert r['speaker_id']!=r['listener_id'] and r['speaker_person']!=r['listener_person']
            assert r['speaker_trimmed']==(r['speaker_word_start_s']>r['original_speaker_start_s']+1e-6)
            assert r['response_prefix']==(r['response_word_end_s']<r['original_response_end_s']-1e-6)
            if r['speaker_trimmed']:assert r['speaker_start_evidence']['kind']=='after_sentence_terminal'
            assert v.terminal(r['listener_words'][-1][2])
            assert r['status']==('review' if r['review_flags'] else 'clean')
            assert not r['audio_normalized'] and not r['visual_filtered'] and not r['media_rendered']
            for role in ['speaker','listener']:
                src=r['source'][role];assert src['id']==r[role+'_id'];assert b<=min(src['audio_duration'],src['video_duration'])+1e-6
            for name in ['speaker','listener']:
                assert all(x>=a-1e-6 and y<=b+1e-6 for x,y,t in r[name+'_words'])
            if r['status']=='clean':
                clean_keys.add(key);assert r['tail_s']>=.12-1e-6
                if r['next_voice_guard_s'] is not None:assert b<=r['next_voice_guard_s']-.08+1e-6
            counts[r['status']]+=1;seconds[r['status']]+=d;by_subset[r['label']+'/'+r['split']]+=1
    assert len(keys)==c['kept_pairs'] and len(clean_keys)==c['clean_pairs']
    assert len(keys&old_keys)==c.get('previous_kept_retained',0)
    assert len(keys&old_clean)==c.get('previous_clean_retained',0)
    assert len(keys-old_keys)==c.get('new_pairs_vs_v1',0)
    for status in ['clean','review']:
        assert counts[status]==c[status+'_pairs'] and abs(seconds[status]-c[status+'_seconds'])<.001
        got=set()
        with (v.OUT/(status+'.jsonl')).open() as f:
            for line in f:
                r=json.loads(line);assert r['status']==status and r['pair_key'] not in got;got.add(r['pair_key'])
        assert got==(clean_keys if status=='clean' else keys-clean_keys)
    for k,n in by_subset.items():assert n==summary['by_subset'][k]['kept_pairs']
    audit=Counter();old_losses=Counter();lost_keys=old_keys-keys
    for index,p in enumerate(sorted(v.OUT.glob('result_shards/*/audit.jsonl.gz'))):
        with gzip.open(p,'rt') as f:
            for line in f:
                r=json.loads(line);audit[r['status']]+=1
                if r['pair_key'] in lost_keys:old_losses[r.get('reason',r['status'])]+=1
        if (index+1)%200==0:print('AUDIT',index+1,'shards checked',flush=True)
    assert sum(audit.values())==c['all_adjacent_pairs']
    for status,key in [('clean','clean_pairs'),('review','review_pairs'),('no_window','no_window_pairs'),('no_cut','no_cut_pairs'),('audio_error','audio_error_pairs'),('processing_error','processing_error_pairs')]:assert audit[status]==c.get(key,0)
    for name,h in json.loads((v.OUT/'rule_fingerprint.json').read_text())['sha256'].items():assert hashlib.sha256((v.ROOT/name).read_bytes()).hexdigest()==h
    result=dict(passed=True,kept_pairs=len(keys),clean_pairs=len(clean_keys),review_pairs=len(keys-clean_keys),max_duration_s=max_d,audit_counts=dict(audit),
        comparison=dict(old_total=len(old_keys),shared=len(keys&old_keys),new_only=len(keys-old_keys),old_only=len(old_keys-keys),old_only_reasons=dict(old_losses)),
        checks=['one unique window per pair','all source conversations and adjacent pairs accounted','word and media boundaries',
                'speaker sentence-start evidence and listener sentence terminal','retained speaker minimum',
                'clean/review partition','manifest/subset/hour accounting','fingerprints'])
    save(v.OUT/'validation.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--full',action='store_true');args=p.parse_args()
    if args.full:validate_full()
    else:unit_checks()
