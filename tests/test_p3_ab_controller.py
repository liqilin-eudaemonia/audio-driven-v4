import sys,json
from pathlib import Path
from unittest.mock import Mock
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from pipeline3_ab_autopilot import Controller
from omni_av_p3_staged.orchestration import write


def bare(tmp_path):
 c=Controller.__new__(Controller);c.r=tmp_path;c.procs={};c.evals={};c.lease=None;c.lease_labels=set()
 c.db={'stage':'C0','decision':None,'routes':{n:dict(step=100,status='ready') for n in ['warm','fresh']},'tasks':{}}
 c.event=Mock();c.start_train=Mock();c.pending=Mock(return_value=[])
 return c


def test_complete_never_restarts_old_routes(tmp_path):
 c=bare(tmp_path);c.db['stage']='completed';c.schedule_train();c.start_train.assert_not_called()


def test_selected_route_does_not_launch_more_c0(tmp_path):
 c=bare(tmp_path);c.db['decision']={'selected':{}};c.schedule_train();c.start_train.assert_not_called()


def test_fallback_keeps_other_group_training(tmp_path):
 c=bare(tmp_path);c.procs={'warm':object()};c.pending.return_value=[('a',{'label':'fresh_000100'})];c.schedule_train()
 assert c.lease=='fresh';c.start_train.assert_not_called()


def test_released_evaluation_group_resumes_training(tmp_path):
 c=bare(tmp_path);c.lease='fresh';c.procs={'warm':object()};c.schedule_train()
 assert c.lease is None;c.start_train.assert_called_once_with('fresh')


def test_resume_recomputed_dev_report_is_discoverable(tmp_path):
 c=bare(tmp_path);write(tmp_path/'warm/attempt001/dev_000100.json',{'step':100})
 assert c.dev_report('warm',100).parent.name=='attempt001'


def test_decision_requires_matched_step_complete_generation(tmp_path):
 c=bare(tmp_path);c.checkpoints=Mock(return_value={300:tmp_path/'step_000300.pt'});c.dev_report=Mock(return_value=tmp_path/'dev.json')
 c.comparison();assert c.db['decision'] is None


def test_new_checkpoint_does_not_extend_lease(tmp_path):
 c=bare(tmp_path);c.lease='fresh';c.lease_labels={'fresh_000100'};c.procs={'warm':object()}
 c.pending.return_value=[('warm/200/generate',{'label':'warm_000200'})]
 c.schedule_train()
 assert c.lease is None and c.lease_labels==set()
 c.start_train.assert_called_once_with('fresh')
 assert c.db['routes']['fresh']['resume_after_eval']


def test_followup_scoring_holds_original_lease(tmp_path):
 c=bare(tmp_path);c.lease='fresh';c.lease_labels={'fresh_000100'};c.procs={'warm':object()}
 c.pending.return_value=[('fresh/100/speech',{'label':'fresh_000100'})]
 c.schedule_train();assert c.lease=='fresh';c.start_train.assert_not_called()


def test_active_owned_worker_holds_lease(tmp_path):
 c=bare(tmp_path);c.lease='fresh';c.lease_labels={'fresh_000100'};c.procs={'warm':object()}
 c.evals={'job':{'lease':'fresh'}}
 c.schedule_train();assert c.lease=='fresh';c.start_train.assert_not_called()


def test_lease_snapshots_at_most_two_labels(tmp_path):
 c=bare(tmp_path)
 c.pending.return_value=[(str(i),{'label':label}) for i,label in enumerate(['a','a','b','c'])]
 assert c.acquire_lease('fresh') and c.lease_labels=={'a','b'}
 c.pending.return_value=[('new',{'label':'d'})]
 assert c.lease_pending()==[] and c.lease_labels=={'a','b'}


def test_long_generation_starts_first(tmp_path):
 c=bare(tmp_path);del c.pending
 c.manifest={'dev':[{'metadata':dict(target_frames=f,target_height=720,target_width=1280)} for f in [25,97]]}
 c.db['tasks']={f'warm_000100/{i}/generate':dict(kind='generate',step=100,status='pending') for i in range(2)}
 assert [k for k,_ in c.pending()]==['warm_000100/1/generate','warm_000100/0/generate']


def test_owned_slots_only_take_leased_labels(tmp_path,monkeypatch):
 import pipeline3_ab_autopilot as module
 c=bare(tmp_path);c.lease='fresh';c.lease_labels={'leased'};c.last_probe=0;c.py='python'
 c.slots=Mock(return_value=({},[({'name':'owned'},4),({'name':'eval194'},0)]))
 for label in ['new','leased']:write(tmp_path/(label+'.json'),{'output':str(tmp_path/label)})
 c.pending.return_value=[(label,dict(label=label,kind='generate',job=str(tmp_path/(label+'.json')),status='pending')) for label in ['new','leased']]
 monkeypatch.setattr(module,'transport',lambda cl,node,cmd:cmd)
 monkeypatch.setattr(module.subprocess,'Popen',Mock())
 c.dispatch_eval()
 assert c.evals['leased']['node']=='owned' and c.evals['leased']['lease']=='fresh'
 assert c.evals['new']['node']=='eval194' and c.evals['new']['lease'] is None
 for item in c.evals.values():item['log'].close()
