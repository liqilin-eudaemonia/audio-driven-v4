import importlib.util
from pathlib import Path


spec=importlib.util.spec_from_file_location('staged_watch',Path(__file__).resolve().parents[1]/'scripts/pipeline3_staged_watch.py')
watch=importlib.util.module_from_spec(spec)
spec.loader.exec_module(watch)


def nodes(low=True):
    return [dict(node='node1',ok=True,gpus=[dict(index=i,util=0 if i==0 and low else 100) for i in range(8)])]


def test_single_idle_gpu_alerts_despite_high_cluster_average():
    state=dict(status='training',folder='C0',phase_started=0);history={}
    for t in range(0,300,60):
        assert watch.utilization_issues(nodes(),state,history,t)==[]
    issue,=watch.utilization_issues(nodes(),state,history,300)
    assert issue['affected']==[dict(node='node1',gpu=0,util=0,seconds_low=300)]


def test_recovery_and_stage_change_reset_low_utilization_timer():
    state=dict(status='training',folder='C0',phase_started=0);history={}
    watch.utilization_issues(nodes(),state,history,0)
    assert watch.utilization_issues(nodes(False),state,history,240)==[]
    assert watch.utilization_issues(nodes(),state,history,300)==[]
    assert watch.utilization_issues(nodes(),dict(state,folder='A'),history,600)==[]


def test_cpu_preparation_and_finished_cache_shard_do_not_alert():
    for status in ['preparing_input_integrity','sealing_cache','waiting_gpus','extracting_full_cache']:
        state=dict(status=status,folder='inputs',phase_started=0);history={}
        watch.utilization_issues(nodes(),state,history,0)
        assert watch.utilization_issues(nodes(),state,history,600)==[]
