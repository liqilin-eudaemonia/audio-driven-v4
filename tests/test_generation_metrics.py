import json
from pathlib import Path

import pytest

from omni_av_p2 import generation_metrics as m


def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))


def fixture(tmp_path):
    generation=tmp_path/'generated';generation.mkdir()
    cases=[];meta=[]
    for i in range(3):
        sid=f'dev{i}';ref=tmp_path/f'ref{i}.mp4';ref.write_bytes(b'CPU protocol fixture')
        geometry=dict(frames=17,height=96,width=64,fps=30.)
        cases.append(dict(sample_id=sid,split='dev',seed=1045+i,geometry=geometry,
            gt_plan=dict(reply='GT words',delivery='voice',behavior='wave',vocal_events='')))
        meta.append(dict(sample_id=sid,video=str(ref),split='dev',target_frames=17,target_height=96,target_width=64,fps=30.))
    metadata=tmp_path/'metadata.jsonl';metadata.write_text(''.join(json.dumps(x)+'\n' for x in meta))
    suite=tmp_path/'suite.json';put(suite,dict(cases=cases,modes=['gt','omni'],geometry_metadata_sha256=m.digest(metadata)))
    put(generation/'worker_000.json',dict(step=300,checkpoint_sha256='checkpoint',suite_sha256=m.digest(suite)))
    def generated(i,mode,reply=None):
        case=cases[i];folder=generation/case['sample_id'];branch=folder/mode
        plan=dict(case['gt_plan'])
        if mode=='omni': plan.update(reply='Omni words' if reply is None else reply)
        put(folder/'case.json',case);put(branch/'text_plan.json',plan)
        put(branch/'completed.json',dict(plan=plan,decoded=case['geometry'],gt_plan_used=mode=='gt',paired_target_media_used=False))
        (branch/'generated.mp4').write_bytes(f'{i} {mode}'.encode())
        if mode=='omni': put(folder/'planning_result.json',dict(status='valid',plan=plan,gt_fallback=False))
    return suite,generation,metadata,generated


def test_modes_use_their_own_reply_and_keep_silent_visual_samples(tmp_path):
    suite,gen,meta,generated=fixture(tmp_path)
    generated(0,'gt');generated(0,'omni',reply='');generated(1,'gt');generated(1,'omni')
    report=m.prepare(suite,gen,meta,tmp_path/'metrics')
    omni=[json.loads(x) for x in (tmp_path/'metrics/omni_available.jsonl').read_text().splitlines()]
    assert omni[0]['transcript']=='' and not omni[0]['metric_applicability']['sync_c']['eligible']
    assert omni[0]['metric_applicability']['fid']['eligible']
    assert omni[1]['transcript']=='Omni words'
    assert report['coverage']['dataset_count']==3 and report['coverage']['omni_plan_valid_count']==2
    assert report['coverage']['generation_success']['omni']['rate_all']==2/3
    assert report['common_alias']==dict(gt='gt_available',omni='omni_available')


def test_different_success_sets_get_common_manifests_and_failure_denominators(tmp_path):
    suite,gen,meta,generated=fixture(tmp_path)
    generated(0,'gt');generated(0,'omni');generated(1,'gt')
    put(gen/'dev1/planning_result.json',dict(status='rejected',error='bad plan',gt_fallback=False))
    report=m.prepare(suite,gen,meta,tmp_path/'metrics')
    assert report['coverage']['omni_plan_valid_rate_attempted']==.5
    assert report['common_sample_ids']==['dev0']
    jobs={j['name']:j for j in report['jobs']}
    assert jobs['gt_available']['sample_ids']==['dev0','dev1']
    assert jobs['gt_common']['sample_ids']==['dev0']
    assert report['cases'][1]['planning_error']=='bad plan'


def test_zero_outputs_not_attempted_is_not_planner_rejection(tmp_path):
    suite,gen,meta,_=fixture(tmp_path)
    report=m.prepare(suite,gen,meta,tmp_path/'metrics')
    assert report['coverage']['omni_plan_status_counts']['not_attempted']==3
    assert report['coverage']['omni_plan_valid_rate_attempted'] is None
    result=m.score_job(report['jobs'][0],{},tmp_path/'empty',0,runner=lambda *a:pytest.fail('Must not launch models'))
    assert result['status']=='no_generated_samples' and all(v is None for v in result['values'].values())


@pytest.mark.parametrize('kind',['metadata','checkpoint','plan','geometry','fallback'])
def test_rejects_inconsistent_provenance(tmp_path,kind):
    suite,gen,meta,generated=fixture(tmp_path);generated(0,'gt');generated(0,'omni')
    if kind=='metadata': meta.write_text(meta.read_text()+'\n')
    if kind=='checkpoint': put(gen/'worker_001.json',dict(step=200,checkpoint_sha256='other',suite_sha256=m.digest(suite)))
    if kind=='plan':
        p=gen/'dev0/omni/text_plan.json';v=m.read(p);v['reply']='changed';put(p,v)
    if kind=='geometry':
        p=gen/'dev0/gt/completed.json';v=m.read(p);v['decoded']['width']=96;put(p,v)
    if kind=='fallback': put(gen/'dev0/planning_result.json',dict(status='valid',gt_fallback=True))
    with pytest.raises(ValueError): m.prepare(suite,gen,meta,tmp_path/'metrics')


def raw(count=2,eligible=1):
    return dict(id_similarity=dict(mean=.9,count=count,coverage=1.)),dict(fid=12.,fvd=18.),dict(
        sync_c=dict(mean=6. if eligible else None,dataset_count=count,eligible_count=eligible,count=eligible),
        sync_d=dict(mean=7. if eligible else None),status_counts={'success':eligible,'not_applicable':count-eligible})


def test_partial_metrics_do_not_pass_or_replace_missing_with_zero():
    a,b,c=raw();a['id_similarity']['count']=1;b['fid']=float('nan');c['sync_c']['count']=0
    result=m.summarize_raw(a,b,c,2)
    assert result['status']=='partial' and result['values']['fid'] is None
    assert len(result['issues'])>=3


def test_no_speech_and_single_sample_have_explicit_not_applicable():
    a,b,c=raw(1,0);b.pop('fvd')
    result=m.summarize_raw(a,b,c,1)
    assert result['status']=='computed' and result['values']['sync_c'] is None
    assert result['fvd_status']=='not_computable' and result['values']['fvd'] is None


def test_command_gpu_binding_and_single_sample_fvd_exclusion(tmp_path):
    manifest=tmp_path/'m.jsonl';manifest.write_text('{"id":"one"}\n')
    c=dict(benchmark_root='/benchmark',legacy_trainer='/legacy',id_python='/id/python',avbench_python='/av/python',
        sync_python='/sync/python',torch_home='/torch',id_model_root='/models',id_model_name='buffalo_l',
        id_frames=8,fid_frames=4,fvd_frames=16,timeout_seconds=7200,syncnet_root='/sync')
    commands=[]
    def runner(command,cwd,log,env,timeout):
        commands.append(command);assert env['CUDA_VISIBLE_DEVICES']=='3'
        out=Path(command[command.index('--output')+1])
        a,b,d=raw(1,0);b.pop('fvd');put(out,{'identity.json':a,'visual.json':b,'sync.json':d}[out.name])
    result=m.score_job(dict(manifest=str(manifest)),c,tmp_path/'result',3,runner)
    assert commands[1][commands[1].index('--metrics')+1]=='fid'
    assert commands[2][commands[2].index('--gpus')+1]=='3'
    assert '--cpu' in commands[0] and result['status']=='computed'


def test_busy_gpu_refuses_before_any_metric_models(monkeypatch):
    monkeypatch.setattr(m.subprocess,'check_output',lambda *a,**k:'0, 100000, 100\n3, 1, 0\n')
    with pytest.raises(RuntimeError,match='occupied'): m.ensure_idle(0)
    m.ensure_idle(3)


def test_corrupt_media_rejected_before_preflight_or_gpu(tmp_path,monkeypatch):
    suite,gen,meta,generated=fixture(tmp_path);generated(0,'gt');generated(0,'omni')
    m.prepare(suite,gen,meta,tmp_path/'plan')
    (gen/'dev0/gt/generated.mp4').write_bytes(b'changed')
    monkeypatch.setattr(m,'preflight',lambda *a:pytest.fail('Media validation must come first'))
    with pytest.raises(ValueError,match='video changed'): m.score(tmp_path/'plan/evaluation_plan.json',{},tmp_path/'out',0)


def test_end_to_end_report_csv_and_tensorboard_keep_both_modes(tmp_path,monkeypatch):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    suite,gen,meta,generated=fixture(tmp_path)
    for i in range(3):
        generated(i,'gt');generated(i,'omni')
    m.prepare(suite,gen,meta,tmp_path/'plan')
    monkeypatch.setattr(m,'preflight',lambda c:dict(config=c,asset_sha256={}))
    devices=[];monkeypatch.setattr(m,'ensure_idle',devices.append)
    calls=[]
    def score_job(job,config,output,gpu):
        calls.append(job['name']);assert gpu==3
        return m.summarize_raw(*raw(3,1),3)
    monkeypatch.setattr(m,'score_job',score_job)
    out=tmp_path/'scores'
    result=m.score(tmp_path/'plan/evaluation_plan.json',{},out,3)
    assert devices==[3] and calls==['gt_available','omni_available']
    assert result['complete_dev_set'] and result['results']['gt_common']['alias_of']=='gt_available'
    assert len((out/'metrics.csv').read_text().splitlines())==5
    events=EventAccumulator(str(out/'tensorboard'));events.Reload()
    assert events.Scalars('generation/omni_available/sync_c')[0].value==6
    assert events.Scalars('generation/omni/plan_valid_rate_all')[0].value==1
