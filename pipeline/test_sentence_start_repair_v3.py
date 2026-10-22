"""Regression tests for sentence-prefix recovery and narrowly scoped 12 s budget."""
import copy
import numpy as np
import repair_sentence_starts_v3 as v

def words(text,start,step=.3):return [[start+i*step,start+i*step+.15,t] for i,t in enumerate(text.split())]
def fixture(end=14.):
 prefix=words('I am so',3.);retained=words('lucky to have you as my mentor.',4.6);listener=words('That is very nice.',8.)
 p={'speaker':dict(words=words('Thanks.',2.)+prefix+retained,video_duration=30,wav=dict(duration=30,frames=1440000,samplerate=48000)),
    'listener':dict(words=listener,video_duration=30,wav=dict(duration=30,frames=1440000,samplerate=48000))}
 r=dict(review_flags=['speaker_sentence_start_unconfirmed'],speaker_words=retained,start_s=4.4,end_s=end,duration_s=end-4.4,
        speaker_word_start_s=4.6,speaker_end_s=retained[-1][1],response_start_s=8.,listener_person='B',
        quiet_threshold_dbfs=[-60,-60],boundary_flags=[],structure_review_flags=['speaker_sentence_start_unconfirmed'],micro_reactions=[])
 return r,p

def check():
 checks=[]
 r,p=fixture();plan,why=v.propose(r,p);assert why=='repairable' and [w[2] for w in plan['words']]==['I','am','so']
 # Realistic quiet cut; never shorten the repaired prefix to force 10 s.
 old=v.rms_slice;v.rms_slice=lambda p,a,b:np.full(b-a,1e-6)
 try:
  out=v.apply_repair(r,plan,p);assert out['speaker_text'].startswith('I am so lucky')
  assert 10<out['duration_s']<=11.95 and out['end_s']==r['end_s'] and out['response_start_s']==8
  assert out['start_recovery']['over_10_seconds'] and out['status']=='clean'
 finally:v.rms_slice=old
 checks.append('complete prefix recovered across old floor boundary; >10 s allowed, end preserved')
 r,p=fixture(end=15.2);assert 'exceed' in v.propose(r,p)[1];checks.append('12 s repair cap: no partial prefix fallback')
 r,p=fixture();p['speaker']['words'][3][1]=1. # invalid/backward source timestamp
 assert v.propose(r,p)[0] is None;checks.append('malformed source timestamp cannot be repaired silently')
 r,p=fixture();p['speaker']['words']=words('I',.1)+r['speaker_words'];assert v.propose(r,p)[1]=='discontinuous_previous_words';checks.append('distant stray pronoun not pulled across a long gap')
 r,p=fixture();p['listener']['words']=words('Yes I read the book yesterday.',3.05)+p['listener']['words'];assert v.propose(r,p)[1]=='substantive_voice_in_recovered_prefix';checks.append('substantive interruption blocks recovery')
 r,p=fixture();p['listener']['words']=words('Yeah.',3.25)+p['listener']['words'];plan,why=v.propose(r,p);assert why=='repairable' and len(plan['micro_reactions'])==1;checks.append('listening microfeedback preserved in recovered prefix')
 r,p=fixture();p['listener']['words']=words('I agree with you.',1.3)+p['listener']['words'];assert v.propose(r,p)[1]=='repairable';checks.append('earlier formal speech ending before recovered sentence is not an obstacle')
 r,p=fixture();r['review_flags']=[];assert v.propose(r,p)[1]=='not_target';checks.append('ordinary V2 windows unchanged; no general 12 s relaxation')
 r,p=fixture();p['speaker']['words'][3][2]='so.';assert v.propose(r,p)[1]=='preceding_word_already_terminal';checks.append('terminal punctuation prevents joining previous sentence')
 r,p=fixture();p['speaker']['words'][0][1]=2.99999999999994;p['speaker']['words'][1][0]=2.99999999999994
 plan,why=v.propose(r,p);assert why=='repairable'
 old=v.rms_slice;v.rms_slice=lambda p,a,b:np.full(b-a,1e-6)
 try:out=v.apply_repair(r,plan,p);assert abs(out['start_s']-plan['speaker_word_start_s'])<1e-6
 finally:v.rms_slice=old
 checks.append('microsecond output rounding tolerates binary float representation at first word')
 result=dict(passed=True,checks=checks);v.v2.save(v.OUT/'unit_validation.json',result);print(result)
if __name__=='__main__':check()
