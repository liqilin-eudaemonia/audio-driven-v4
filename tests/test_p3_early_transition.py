import copy
import importlib.util
import json
from pathlib import Path

import pytest
import torch

from omni_av_p3_staged.checkpoint import initialize, save
from omni_av_p3_staged.orchestration import read, write, sha
from omni_av_p2.checkpoint import capture_rng
from test_p3_staged_render import model


def test_explicit_transition_is_checkpoint_bound_and_keeps_lora_zero(tmp_path):
    c0,_=model('C0');opt=torch.optim.AdamW(c0.optimizer_groups(.001,.001));cp=tmp_path/'c0.pt'
    save(cp,c0,opt,700,dict(world_size=1,manifest_sha256='data'),[capture_rng()])
    auth=dict(version='p3.user-stage-transition.v1',user_authorized=True,from_stage='C0',to_stage='A',
        from_step=700,a_steps=1000,evaluation_policy='parallel_nonblocking',checkpoint=str(cp),
        checkpoint_sha256=sha(cp),manifest_sha256='data',user_instruction='Switch after step700')
    path=tmp_path/'authorization.json';write(path,auth)
    a,_=model('A');assert initialize(cp,a,'data',transition_authorization=path)==700
    for adapter in a.adapters:assert not adapter.b.weight.any()
    for k,v in c0.connector.state_dict().items():torch.testing.assert_close(a.connector.state_dict()[k],v)
    for field,value in [('from_step',600),('checkpoint_sha256','wrong'),('manifest_sha256','wrong'),
                        ('to_stage','B'),('user_authorized',False),('a_steps',300)]:
        bad=copy.deepcopy(auth);bad[field]=value;write(path,bad)
        with pytest.raises(ValueError,match='authorization'):initialize(cp,a,'data',transition_authorization=path)
    with pytest.raises(ValueError,match='passing evaluation'):initialize(cp,a,'data')


def scheduler():
    path=Path(__file__).resolve().parents[1]/'scripts/pipeline3_schedule_transition.py'
    spec=importlib.util.spec_from_file_location('p3_schedule_test',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m


def setup_schedule(tmp_path):
    folder=tmp_path/'C0_attempt1';folder.mkdir()
    write(tmp_path/'scheduled_transition.json',dict(user_authorized=True,from_stage='C0',to_stage='A',
        from_step=700,a_steps=1000,c0_folder=str(folder),thread='test',diagnostic_root='diagnostic',
        evaluation_policy='parallel_nonblocking',user_instruction='Switch after step700'))
    write(tmp_path/'plan.json',dict(manifest_sha256='data',stage_steps={'C0':2000,'A':1000}))
    write(tmp_path/'cluster.json',dict(python='python',source='original'))
    write(tmp_path/'state.json',dict(pid=-1,status='stopped'))
    cp=folder/'step_000700.pt';cp.write_bytes(b'checkpoint')
    write(cp.with_suffix('.complete.json'),dict(step=700,sha256=sha(cp)))
    write(folder/'completed.json',dict(step=700,stage='C0',stopped=True,engineering_gate=False))
    write(folder/'dev_000700.json',{})
    return folder


def test_handoff_uses_output_stop_preserves_checkpoint_and_no_fake_gate(tmp_path,monkeypatch):
    q=scheduler();folder=setup_schedule(tmp_path);calls=[]
    monkeypatch.setattr(q.sys,'argv',['schedule','--root',str(tmp_path)])
    monkeypatch.setattr(q.signal,'signal',lambda *args:None)
    monkeypatch.setattr(q,'last_step',lambda _:700)
    monkeypatch.setattr(q,'stop_watch',lambda _:calls.append('watch_stopped'))
    monkeypatch.setattr(q,'process_matches',lambda *args:False)
    monkeypatch.setattr(q,'notify',lambda *args:True)
    monkeypatch.setattr(q,'start_diagnostics',lambda *args:dict(status='launched'))
    def launch(python,script,root,log):
        calls.append(script.name)
        if script.name=='pipeline3_staged_queue.py':write(root/'state.json',dict(pid=123,status='training',stage='A'))
        return 123
    monkeypatch.setattr(q,'launch',launch);q.main()
    assert (folder/'STOP').exists() and not (tmp_path/'STOP').exists()
    assert not (tmp_path/'C0_quality_gate.json').exists()
    assert read(tmp_path/'C0_completed.json')['quality_gate_passed'] is None
    assert read(tmp_path/'plan.json')['stage_steps']=={'C0':700,'A':1000}
    assert read(tmp_path/'early_transition_authorization.json')['checkpoint_sha256']==sha(folder/'step_000700.pt')
    assert calls==['watch_stopped','pipeline3_staged_queue.py','pipeline3_staged_watch.py']


@pytest.mark.parametrize('stop,step',[(True,700),(False,701)])
def test_stop_and_missed_boundary_never_launch(tmp_path,monkeypatch,stop,step):
    q=scheduler();folder=setup_schedule(tmp_path)
    if stop:(tmp_path/'STOP').touch()
    monkeypatch.setattr(q.sys,'argv',['schedule','--root',str(tmp_path)])
    monkeypatch.setattr(q.signal,'signal',lambda *args:None)
    monkeypatch.setattr(q,'last_step',lambda _:step)
    monkeypatch.setattr(q,'notify',lambda *args:True)
    monkeypatch.setattr(q,'launch',lambda *args:pytest.fail('Must not launch'))
    with pytest.raises(InterruptedError if stop else ValueError):q.main()
    assert not (tmp_path/'early_transition_authorization.json').exists()
    assert not (folder/'STOP').exists()
