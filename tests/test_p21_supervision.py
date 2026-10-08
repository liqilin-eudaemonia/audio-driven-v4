"""Health/recovery decisions are tested without touching any live job or GPU."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location('p21_supervisor_test', SCRIPTS / 'pipeline21_supervise.py')
supervision = importlib.util.module_from_spec(spec)
spec.loader.exec_module(supervision)


def gpu(util=100, ok=True):
    return dict(time=2000, nodes=[dict(node='node12', ok=ok, error='offline',
        gpus=[dict(index=0, util=util, used_mib=40000, total_mib=140000, watts=600, temp_c=65)])])


def training(modified=1990, loss=1.0, stop=False, completed=False):
    return dict(ranks=[dict(rank=0, step=20, modified=modified, loss=loss, av=.5, grad_norm=1)],
                failures=[], stop=stop, completed=completed)


def kinds(issues):
    return {i['kind'] for i in issues}


def test_health_window_does_not_confuse_startup_or_checkpoint_dips():
    assert supervision.health_issues(training(), gpu(0), [gpu(100)] * 4 + [gpu(0)], 2000, 0, 1) == []
    assert supervision.health_issues(training(0), gpu(0), [gpu(0)] * 5, 2000, 1900, 1) == []
    issues = supervision.health_issues(training(), gpu(0), [gpu(0)] * 5, 2000, 0, 1)
    assert kinds(issues) == {'sustained_low_utilization'}


def test_stall_unreachable_and_nonfinite_are_distinct():
    issues = supervision.health_issues(training(100), gpu(), [gpu()] * 5, 2000, 0, 1)
    assert kinds(issues) == {'training_stalled'}
    issues = supervision.health_issues(training(loss=float('nan')), gpu(ok=False), [], 2000, 0, 1)
    assert kinds(issues) == {'nonfinite_metrics', 'node_unreachable'}
    assert supervision.health_issues(training(0, stop=True), gpu(0), [gpu(0)] * 5, 2000, 0, 1) == []
    assert supervision.health_issues(training(0, completed=True), gpu(0), [gpu(0)] * 5, 2000, 0, 1) == []


def test_high_util_low_power_can_expose_communication_wait():
    value = gpu()
    value['nodes'][0]['gpus'][0]['watts'] = 130
    issues = supervision.health_issues(training(), value, [value] * 5, 2000, 0, 1)
    assert kinds(issues) == {'sustained_high_util_low_power'}


@pytest.mark.parametrize('message,expected', [
    ('torch.distributed.DistBackendError: ncclSystemError', True),
    ('Connection reset by peer', True),
    ('Input/output error', True),
    ('NCCL initialized\nValueError: checkpoint identity/architecture mismatch', False),
    ('NCCL warning\nCUDA out of memory', False),
    ('watchdog timeout\nNonfinite training loss', False),
])
def test_recovery_allowlist(message, expected):
    assert supervision.transient_failure(message) == expected


def test_only_fully_published_checkpoint_is_recoverable(tmp_path):
    folder = tmp_path / 'attempt_000/checkpoints'
    folder.mkdir(parents=True)
    marker = folder / 'extractor_step_000100.complete.json'
    marker.write_text(json.dumps(dict(step=100)))
    full = folder / 'extractor_step_000100.pt'
    full.write_bytes(b'simulated full checkpoint')
    assert supervision.complete_checkpoints(tmp_path) == []
    (folder / 'extractor_step_000100_deployment.pt').write_bytes(b'simulated deployment')
    assert supervision.complete_checkpoints(tmp_path) == [(100, full)]


def test_exact_stop_arms_before_save_and_tolerates_partial_log_line(tmp_path):
    folder = tmp_path / 'old/training'
    folder.mkdir(parents=True)
    log = folder / 'train_rank0.jsonl'
    log.write_text('{"step": 200}\n{"step":')
    assert supervision.last_row(log)['step'] == 200
    record = tmp_path / 'record.json'
    subprocess.run([sys.executable, str(SCRIPTS / 'stop_training_at_step.py'), '--run', str(folder.parent),
                    '--step', '200', '--record', str(record)], check=True, timeout=10)
    assert (folder / 'STOP').exists()
    assert json.loads(record.read_text())['observed_step'] == 200
