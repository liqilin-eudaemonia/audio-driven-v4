import copy
import json
import os
from pathlib import Path

import pytest
import torch

from omni_av_p3_staged.integrity import (prepare_inputs, load_record, verified_load, record_path)
from omni_av_p3_staged.monitor import progress_issue
from omni_av_p3_staged.orchestration import engineering_complete, write, sha, update_phase
from omni_av_p3_staged.quality import identity_comparison


def identities(count=8,frames=8,score=.9):
    return dict(per_sample=[dict(id=f's{i}',status='success' if i<count else 'no_generated_face',
        id_similarity=score if i<count else None,id_valid_frames=frames if i<count else 0,
        id_frame_coverage=frames/8 if i<count else 0.) for i in range(8)])


@pytest.mark.parametrize('count,frames',[(1,8),(8,1),(6,8)])
def test_identity_gate_rejects_lost_faces(count,frames):
    result=identity_comparison(identities(),identities(count,frames),[f's{i}' for i in range(8)],dict(id_tolerance=.05))
    assert not result['nonregression_passed']


def test_identity_uses_same_samples_and_frame_coverage():
    baseline=identities();baseline['per_sample'][-1]['id_similarity']=0.
    result=identity_comparison(baseline,identities(7,7,.8),[f's{i}' for i in range(8)],dict(id_tolerance=.05))
    assert result['coverage_passed'] and result['frame_coverage_passed']
    assert result['baseline_paired_mean']==pytest.approx(.9)
    assert not result['nonregression_passed']
    good=identity_comparison(identities(),identities(7,7,.88),[f's{i}' for i in range(8)],dict(id_tolerance=.05))
    assert good['nonregression_passed']
    bad=identities();bad['per_sample'][0]['id']='wrong'
    with pytest.raises(ValueError,match='ID mismatch'):
        identity_comparison(identities(),bad,[f's{i}' for i in range(8)],dict(id_tolerance=.05))


def gate(folder,phase,step=None,stage='A'):
    step=step if step is not None else (2 if phase=='initial' else 3)
    write(folder/'completed.json',dict(stage=stage,step=step,stopped=step<3,engineering_gate=True))
    cp=folder/f'step_{step:06d}.pt';cp.write_bytes(b'checkpoint bytes')
    write(cp.with_suffix('.complete.json'),dict(step=step,sha256=sha(cp)))
    write(folder/'run.json',dict(identity=dict(stage=stage,engineering_gate=True,manifest_sha256='data',world_size=24)))
    return cp


def test_gate_rejects_stopped_resume_missing_or_corrupt_checkpoint(tmp_path):
    gate(tmp_path/'initial','initial')
    assert engineering_complete(tmp_path/'initial','A','initial','data')
    gate(tmp_path/'stopped','resume',step=2)
    assert not engineering_complete(tmp_path/'stopped','A','resume','data')
    cp=gate(tmp_path/'resume','resume')
    assert engineering_complete(tmp_path/'resume','A','resume','data')
    assert not engineering_complete(tmp_path/'resume','A','resume','data',1)
    assert not engineering_complete(tmp_path/'resume','A','resume','wrong-data')
    cp.write_bytes(b'changed')
    assert not engineering_complete(tmp_path/'resume','A','resume','data')
    cp.unlink()
    assert not engineering_complete(tmp_path/'resume','A','resume','data')


def sources(tmp_path):
    data=tmp_path/'dataset';base=data/'ltx23_precomputed_720p_area_qualified_20260920'
    entry=dict(sample_id='s1',split='train',text=str(tmp_path/'text.pt'))
    text=dict(prompt='same',video=torch.zeros(1,2,4));torch.save(text,entry['text'])
    for name in ('latents','audio_latents'):
        p=base/name/'train/s1.pt';p.parent.mkdir(parents=True);torch.save(dict(latents=torch.zeros(2)),p)
    manifest=dict(config=dict(data_root=str(data)),train=[entry],dev=[])
    return data,entry,manifest


@pytest.mark.parametrize('field',['text','latents','audio_latents'])
def test_changed_payload_rejected_even_with_same_metadata(tmp_path,field):
    data,entry,manifest=sources(tmp_path);root=tmp_path/'inputs'
    prepare_inputs(manifest,root,'manifest',tmp_path,workers=2)
    record=load_record(root,entry,'manifest',data)
    path=record['files'][field]['path'];before=verified_load(path,record['files'])
    name='video' if field=='text' else 'latents';before[name]+=1;torch.save(before,path)
    with pytest.raises(ValueError,match='content hash mismatch'):verified_load(path,record['files'])


def test_input_seal_rejects_marker_tampering_and_absence(tmp_path):
    data,entry,manifest=sources(tmp_path);root=tmp_path/'inputs'
    with pytest.raises(ValueError,match='seal required'):load_record(root,entry,'manifest',data)
    first=load_record(root,entry,'manifest',data,allow_unsealed=True)
    prepare_inputs(manifest,root,'manifest',tmp_path)
    assert load_record(root,entry,'manifest',data)==first
    record_path(root,entry).write_text('{}')
    with pytest.raises(ValueError,match='seal mismatch'):load_record(root,entry,'manifest',data)


def test_interrupted_input_preparation_refuses_changed_baseline(tmp_path):
    data,entry,manifest=sources(tmp_path);root=tmp_path/'inputs'
    load_record(root,entry,'manifest',data,allow_unsealed=True)
    torch.save(dict(prompt='same',video=torch.ones(1,2,4)),entry['text'])
    with pytest.raises(ValueError,match='content hash mismatch'):prepare_inputs(manifest,root,'manifest',tmp_path)
    assert not (root/'complete.json').exists()


def test_input_preparation_stop_never_seals(tmp_path):
    _,_,manifest=sources(tmp_path);(tmp_path/'STOP').touch()
    with pytest.raises(InterruptedError):prepare_inputs(manifest,tmp_path/'inputs','manifest',tmp_path)
    assert not (tmp_path/'inputs/complete.json').exists()


def test_missing_first_heartbeat_and_cache_stall(tmp_path):
    state=dict(status='training',folder=str(tmp_path),phase_started=100.,time=999.)
    issue=progress_issue(tmp_path,state,{},1401.)
    assert issue and issue['heartbeat_files']==0
    state.update(status='extracting_full_cache',phase_started=100.)
    path=tmp_path/'progress/worker.json';write(path,dict(completed=10,time=200.));os.utime(path,(200,200))
    history={};assert progress_issue(tmp_path,state,history,300.) is None
    write(path,dict(completed=10,time=1500.));os.utime(path,(1500,1500))
    assert progress_issue(tmp_path,state,history,1500.) is not None
    write(path,dict(completed=11,time=1501.));os.utime(path,(1501,1501))
    assert progress_issue(tmp_path,state,history,1501.) is None
    state.update(status='sealing_cache',phase_started=1600.)
    assert progress_issue(tmp_path,state,history,1601.) is None


def test_polling_preserves_phase_start(monkeypatch):
    import omni_av_p3_staged.orchestration as o
    state=dict(status='training',folder='same',phase_started=1.,time=1.)
    monkeypatch.setattr(o.time,'time',lambda:50.)
    update_phase(state,status='training',folder='same',active=3)
    assert state['phase_started']==1. and state['time']==50.
    update_phase(state,status='sealing_cache')
    assert state['phase_started']==50.


def test_queue_restarts_incomplete_resume_instead_of_publishing_false_pass(tmp_path,monkeypatch):
    import sys
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts'))
    import pipeline3_staged_queue as q
    manifest=tmp_path/'manifest.json';write(manifest,dict(config=dict(initial_connector=dict(path='pilot'))))
    digest=sha(manifest)
    write(tmp_path/'plan.json',dict(manifest=str(manifest),prerequisite='unused',thread='unused'))
    write(tmp_path/'cluster.json',{})
    write(tmp_path/'local_gate/completed.json',dict(passed=True))
    write(tmp_path/'local_gate/indices_0.json',[0])
    for stage in ('C0','A'):
        for phase in ('initial','resume'):
            folder=tmp_path/f'gate24_{stage}_{phase}'
            gate(folder,phase,step=2 if stage=='A' and phase=='resume' else None,stage=stage)
            run=json.loads((folder/'run.json').read_text());run['identity']['manifest_sha256']=digest;write(folder/'run.json',run)
    write(tmp_path/'gpu_gate_passed.json',dict(passed=True))
    write(tmp_path/'render_gate/completed.json',dict(passed=True))
    calls=[]
    def fake_train(root,plan,cluster,stage,out,**kwargs):
        calls.append((stage,out.name))
        assert not (root/'gpu_gate_passed.json').exists()
        gate(out,'resume',stage=stage)
        run=json.loads((out/'run.json').read_text());run['identity']['manifest_sha256']=digest;write(out/'run.json',run)
    class ReachedRender(Exception):pass
    def render(*args):raise ReachedRender()
    monkeypatch.setattr(sys,'argv',['queue','--root',str(tmp_path)])
    monkeypatch.setattr(q.signal,'signal',lambda *args:None)
    monkeypatch.setattr(q,'prerequisite',lambda _:True)
    monkeypatch.setattr(q,'wait_idle',lambda *args:None)
    monkeypatch.setattr(q,'notify',lambda *args,**kwargs:True)
    monkeypatch.setattr(q,'cluster_training',fake_train)
    monkeypatch.setattr(q,'render_gate',render)
    with pytest.raises(ReachedRender):q.main()
    assert calls==[('A','gate24_A_resume')]
    assert engineering_complete(tmp_path/'gate24_A_resume','A','resume',digest,24)
    assert list(tmp_path.glob('gate24_A_resume_failed_*'))
    assert list(tmp_path.glob('render_gate_superseded_*'))
