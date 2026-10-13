import sys,json
from pathlib import Path
from unittest.mock import Mock
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from pipeline3_ab_autopilot import Controller
from omni_av_p3_staged.orchestration import write


def bare(tmp_path):
 c=Controller.__new__(Controller);c.r=tmp_path;c.procs={};c.evals={};c.lease=None
 c.db={'stage':'C0','decision':None,'routes':{n:dict(step=100,status='ready') for n in ['warm','fresh']},'tasks':{}}
 c.event=Mock();c.start_train=Mock();c.pending=Mock(return_value=[])
 return c


def test_complete_never_restarts_old_routes(tmp_path):
 c=bare(tmp_path);c.db['stage']='completed';c.schedule_train();c.start_train.assert_not_called()


def test_selected_route_does_not_launch_more_c0(tmp_path):
 c=bare(tmp_path);c.db['decision']={'selected':{}};c.schedule_train();c.start_train.assert_not_called()


def test_fallback_keeps_other_group_training(tmp_path):
 c=bare(tmp_path);c.procs={'warm':object()};c.pending.return_value=[('a',{})];c.schedule_train()
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
