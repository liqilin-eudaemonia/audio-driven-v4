import importlib.util
from pathlib import Path
import sys
from unittest.mock import patch

scripts = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(scripts))
spec = importlib.util.spec_from_file_location('gpu4_monitor', scripts / 'pipeline2_gpu4_monitor.py')
monitor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(monitor)


def snapshot(ok):
    return dict(nodes=[dict(node='p2_gpu4', ok=ok, **({} if ok else {'error': 'SSH connection closed'}))])


def test_transient_probe_recovers_without_sleeping_after_success():
    with patch.object(monitor, 'gpu_snapshot', side_effect=[snapshot(False), snapshot(True)]) as probe, patch.object(monitor.time, 'sleep') as sleep:
        result = monitor.snapshot_with_retry({})
    assert result['nodes'][0]['ok']
    assert result['probe_attempts'] == 2 and len(result['prior_probe_failures']) == 1
    assert probe.call_count == 2 and sleep.call_count == 1


def test_persistent_failure_is_preserved_after_bounded_retries():
    with patch.object(monitor, 'gpu_snapshot', side_effect=[snapshot(False) for _ in range(3)]) as probe, patch.object(monitor.time, 'sleep'):
        result = monitor.snapshot_with_retry({})
    assert not result['nodes'][0]['ok']
    assert result['probe_attempts'] == 3 and probe.call_count == 3


def test_healthy_probe_has_no_extra_calls():
    with patch.object(monitor, 'gpu_snapshot', return_value=snapshot(True)) as probe, patch.object(monitor.time, 'sleep') as sleep:
        result = monitor.snapshot_with_retry({})
    assert result['probe_attempts'] == 1 and probe.call_count == 1
    sleep.assert_not_called()
