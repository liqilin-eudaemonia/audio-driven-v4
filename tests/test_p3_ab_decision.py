import sys
from pathlib import Path
from unittest.mock import Mock
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from pipeline3_ab_autopilot import Controller
from pipeline3_ab_policy import INDICES
from omni_av_p3_staged.orchestration import write,read


def test_complete_paired_gate_selects_best_and_transitions_only_at_plateau(tmp_path):
 c=Controller.__new__(Controller);c.r=tmp_path;c.db={'decision':None,'stage':'C0'};c.procs={};c.plan={'thread':'test'};c.notify=Mock();c.checkpoint=Mock(return_value={'sha256':'digest'})
 cp={}
 for name in ['warm','fresh']:
  cp[name]={}
  for step in [100,200,300,400,500]:
   folder=tmp_path/name/'attempt000';p=folder/f'step_{step:06d}.pt';cp[name][step]=p
   v=1 if name=='warm' else .8
   write(folder/f'dev_{step:06d}.json',dict(rows=[dict(sample_id=str(i),mode='beta',values=dict(video=v,audio=v)) for i in range(48)]))
   if step<300:continue
   e=tmp_path/'evaluation'/f'{name}_{step:06d}'
   for i in INDICES:
    write(e/f'pred25_{i:03d}'/'completed.json',dict(waveform=dict(finite=True,pre_clamp_over_one_fraction=0)))
    write(e/'scores'/f'{i:03d}'/'completed.json',dict(asr=dict(rms=.01,reference_characters=10,character_errors=5 if name=='warm' else 4,reference_words=2,word_errors=1),syncnet=[dict(status='success',min_dist=v,confidence=1)]))
 c.checkpoints=lambda name:cp[name]
 c.comparison()
 assert c.db['decision']['selected']['route']=='fresh'
 assert c.db['decision']['step']==400
 assert c.db['decision']['selected']['step']==300  # tied checkpoint scores select earlier
 assert c.db['decision']['transition_reason']=='C0 validation plateau'
 assert (tmp_path/'selection.json').exists()
