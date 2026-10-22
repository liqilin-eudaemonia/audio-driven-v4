"""Synthetic regression: adjacent pairing swaps roles, preserves listening feedback.

No dataset, GPU, ASR or media rendering required. Uses the actual V2 selection
and output path; V3 recovery is checked as a post-stage preserving role IDs.
"""
import json
from unittest.mock import patch
import numpy as np
import apply_sentence_windows_v2 as v2
import repair_sentence_starts_v3 as v3

def phrase(text,start):
 return [[start+i*.3,start+i*.3+.18,t] for i,t in enumerate(text.split())]

def run_checks():
 words={'A':phrase('Please tell me what you think about this.',.5)+phrase('I think we should try this together.',8.5),
        'B':[[1.25,1.4,'Yeah.']]+phrase('That sounds good and I agree completely.',4.5)}
 people={role:dict(id='person_'+role,words=ww,video_duration=20,refs={},wav=dict(duration=20,data_bytes=0)) for role,ww in words.items()}
 def rms(p):
  values=np.full(2000,1e-6);values[v2.old.word_mask(p['words'],2000)]=.1;return values
 conv=dict(label='synthetic',split='test',interaction='adjacent_roles',keys=['A','B'])
 with patch.object(v2.old,'load_person',side_effect=lambda k:people[k]),patch.object(v2.old,'pcm_rms',side_effect=rms):
  kept,pending,audit,counts,reasons=v2.process_conversation(conv)
 assert counts['all_adjacent_pairs']==2 and len(kept)==2,(counts,pending,audit)
 first,second=kept
 assert (first['speaker_person'],first['listener_person'])==('A','B')
 assert (second['speaker_person'],second['listener_person'])==('B','A')
 assert first['listener_id']==second['speaker_id']=='person_B'
 assert first['listener_turn']==second['speaker_turn']=='T002'
 assert first['micro_listening_count']==1 and first['micro_reactions'][0]['text']=='Yeah.'
 assert 'Yeah.' not in first['listener_text'] # Listening feedback is not the formal response onset.
 assert second['speaker_words']==first['listener_words']
 assert first['pair_key']!=second['pair_key'] and first['split']==second['split']=='test'
 # V3 cannot generate a new pairing or swap roles; non-target rows are unchanged.
 for row in kept:
  repaired,why=v3.process(row,audio=False)
  assert why=='not_target' and repaired==row
 return dict(passed=True,checks=['A-B-A gives two adjacent pairs','listener becomes next speaker with same person/turn IDs',
  'listening Yeah retained without becoming a formal turn','both pairs preserve source split','V3 non-target pass-through preserves pair identities'],
  pairs=[{k:r[k] for k in ['pair_key','speaker_id','listener_id','speaker_turn','listener_turn']} for r in kept])
if __name__=='__main__':print(json.dumps(run_checks(),ensure_ascii=False,indent=2))
