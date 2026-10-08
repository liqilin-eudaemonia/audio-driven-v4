#!/usr/bin/env python3
"""Own one P2.1 job, check all GPUs every 15 minutes, queue scoped Codex alerts.

No arbitrary PID killing, no metric-driven learning-rate changes, no cache edits.
The guarded cluster launcher owns/reaps only its own local and remote children.
"""
import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import datetime
import fcntl
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import signal
import socket
import statistics
import subprocess
import time

from stop_training_at_step import last_row

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.tmp.{os.getpid()}')
    with temp.open('w') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write('\n')
    os.replace(temp, path)


def tail(path, limit=65536):
    if not path.is_file():
        return ''
    with path.open('rb') as stream:
        stream.seek(max(0, path.stat().st_size - limit))
        return stream.read().decode(errors='replace')


def gpu_snapshot(cluster):
    key = cluster.get('ssh_key', f'/private/ssh_keys/root_{socket.gethostname()}_ed25519')
    query = ['nvidia-smi', '--query-gpu=index,memory.used,memory.total,utilization.gpu,power.draw,temperature.gpu',
             '--format=csv,noheader,nounits']
    def one(node):
        command = query if node.get('local') else ['ssh', '-i', key, '-o', 'BatchMode=yes',
            '-o', 'ConnectTimeout=10', '-o', 'StrictHostKeyChecking=yes', 'root@' + node['host'], shlex.join(query)]
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=20, check=True)
            rows = []
            for line in result.stdout.splitlines():
                values = [float(x.strip()) for x in line.split(',')]
                rows.append(dict(zip(('index', 'used_mib', 'total_mib', 'util', 'watts', 'temp_c'), values)))
            rows = [r for r in rows if int(r['index']) in node['gpus']]
            if len(rows) != len(node['gpus']):
                raise ValueError('Requested GPU missing from nvidia-smi')
            return dict(node=node['name'], host=node['host'], gpus=rows, ok=True)
        except Exception as exc:
            return dict(node=node['name'], host=node['host'], ok=False, error=str(exc))
    with ThreadPoolExecutor(max_workers=len(cluster['nodes'])) as pool:
        return dict(time=time.time(), nodes=list(pool.map(one, cluster['nodes'])))


def idle(snapshot):
    return all(n['ok'] and all(g['used_mib'] <= 1000 and g['util'] <= 5 for g in n['gpus'])
               for n in snapshot['nodes'])


def training_snapshot(folder, world):
    rows = []
    for rank in range(world):
        path = folder / f'train_rank{rank}.jsonl'
        row = last_row(path)
        if row:
            rows.append(dict(rank=rank, modified=path.stat().st_mtime, **{k: v for k, v in row.items() if k != 'rank'}))
    return dict(ranks=rows, failures=[str(p) for p in folder.glob('failed_rank*.json')],
                completed=(folder / 'completed.json').exists(), stop=(folder / 'STOP').exists())


def health_issues(training, snapshot, history, now, started, world, stale_seconds=900):
    issues = []
    for node in snapshot['nodes']:
        if not node['ok']:
            issues.append(dict(kind='node_unreachable', node=node['node'], error=node['error']))
    if training['failures']:
        issues.append(dict(kind='worker_failed', paths=training['failures']))
    rows = training['ranks']
    if any(not math.isfinite(row[key]) for row in rows
           for key in ('loss', 'av', 'grad_norm') if key in row):
        issues.append(dict(kind='nonfinite_metrics'))
    if training['completed'] or training['stop']:
        return issues
    if now - started > stale_seconds:
        if len(rows) != world or now - min(r['modified'] for r in rows) > stale_seconds:
            issues.append(dict(kind='training_stalled', ranks_reporting=len(rows)))
        elif max(r['step'] for r in rows) - min(r['step'] for r in rows) > 1:
            issues.append(dict(kind='rank_step_gap'))
        # Check a window, not the transient dip during IO/checkpoint publication.
        if len(history) >= 5:
            for node in snapshot['nodes']:
                if not node['ok']:
                    continue
                for gpu in node['gpus']:
                    values = [g['util'] for s in list(history)[-5:] for n in s['nodes']
                              if n['ok'] and n['node'] == node['node'] for g in n['gpus']
                              if g['index'] == gpu['index']]
                    if len(values) == 5 and statistics.mean(values) < 70:
                        issues.append(dict(kind='sustained_low_utilization', node=node['node'],
                                           gpu=int(gpu['index']), mean_util=statistics.mean(values)))
    return issues


def complete_checkpoints(root):
    result = []
    for marker in root.glob('attempt_*/checkpoints/extractor_step_*.complete.json'):
        try:
            data = json.loads(marker.read_text())
            stem = marker.name.removesuffix('.complete.json')
            full, deploy = marker.with_name(stem + '.pt'), marker.with_name(stem + '_deployment.pt')
            if full.is_file() and deploy.is_file() and min(full.stat().st_size, deploy.stat().st_size) > 0:
                result.append((data['step'], full))
        except (OSError, ValueError, KeyError):
            continue
    return sorted(result)


def transient_failure(text):
    value = text.lower()
    if any(s in value for s in ('out of memory', 'nonfinite', 'nan loss', 'identity mismatch', 'invalid tensor')):
        return False
    return any(s in value for s in ('ncclsystemerror', 'ncclremoteerror', 'distbackenderror',
                                    'connection reset', 'connection closed', 'timed out',
                                    'watchdog caught collective operation timeout',
                                    'socket error', 'input/output error'))


class Supervisor:
    def __init__(self, args):
        self.args, self.root = args, args.root.resolve()
        self.cluster = json.loads(args.cluster.read_text())
        self.world = sum(len(n['gpus']) for n in self.cluster['nodes'])
        self.process = None
        self.attempt = None
        self.attempt_started = None
        self.restarts = 0
        self.last_notified = {}
        self.history = deque(maxlen=8)
        self.last_check = 0
        self.pending_recovery = False
        self.launch_stream = None
        self.stop_requested_at = None

    def event(self, kind, **values):
        record = dict(time=time.time(), utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), kind=kind, **values)
        with (self.root / 'events.jsonl').open('a') as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')
        print(json.dumps(record, ensure_ascii=False), flush=True)
        return record

    def notify(self, kind, **values):
        if time.time() - self.last_notified.get(kind, 0) < self.args.check_seconds:
            return
        record = self.event(kind, **values)
        with (self.root / 'alerts.jsonl').open('a') as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')
        message = (f'[三机GPU监控自动报告] {kind}；记录目录：{self.root}。'
            '这是用户授权的训练监控。请先读最新state.json、health.json、alerts.jsonl确认问题是否仍存在，'
            '再诊断并解决本次P2.1任务的问题。只操作本次任务，不停止其他作业；保持P2.1从头初始化、'
            '原始720p-area、目标1000步、每100步保存、不开Dev推理。若已恢复，仅报告状态。'
            + json.dumps(values, ensure_ascii=False)[:800])
        try:
            result = subprocess.run([self.args.codex, 'queue', '--thread', self.args.thread,
                '--message', message], capture_output=True, text=True, timeout=30, cwd=str(ROOT))
            self.event('notification_delivery', alert=kind, returncode=result.returncode,
                       stdout=result.stdout[-1000:], stderr=result.stderr[-1000:])
            if result.returncode == 0:
                self.last_notified[kind] = time.time()
        except Exception as exc:
            self.event('notification_delivery_failed', alert=kind, error=str(exc))

    def stop_owned_process(self):
        if self.process and self.process.poll() is None:
            self.event('terminate_owned_launcher', pid=self.process.pid)
            self.process.terminate()  # SIGTERM handler closes/reaps only its guard children.
            try:
                self.process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=10)
            self.event('owned_launcher_exited', returncode=self.process.returncode)

    def launch(self, checkpoint=None):
        attempt = self.root / f'attempt_{self.restarts:03d}'
        attempt.mkdir(exist_ok=False)
        start = checkpoint[0] if checkpoint else 0
        command = [self.cluster['python'], '-u', str(ROOT / 'scripts/pipeline21_cluster.py'),
            '--cluster', str(self.args.cluster.resolve()), '--log-dir', str(attempt / 'agents'), '--',
            '--config', str(self.args.config.resolve()), '--cache', str(self.args.cache.resolve()),
            '--output', str(attempt / 'training'), '--checkpoint-dir', str(attempt / 'checkpoints'),
            '--network-config', str(ROOT / 'configs/p2_roce.json'), '--run-id', f'p21_fresh_attempt_{self.restarts:03d}',
            '--recipe', 'joint', '--steps', str(self.args.steps - start), '--accumulation', '3',
            '--balance-workload', '--save-every', '100', '--startup-save-step', '3', '--cache-samples', '3',
            '--checkpoint-every', '1', '--seed', '45', '--allow-terminal-trim', '--tensorboard']
        if checkpoint:
            command += ['--resume', str(checkpoint[1])]
        assert '--init-p2' not in command
        self.launch_stream = (attempt / 'launcher.log').open('x')
        self.process = subprocess.Popen(command, cwd=ROOT, stdout=self.launch_stream,
                                        stderr=subprocess.STDOUT, start_new_session=True)
        self.attempt, self.attempt_started = attempt, time.time()
        self.history.clear()
        write_json(attempt / 'launch.json', dict(command=command, pid=self.process.pid,
            started=self.attempt_started, initial_source='fresh' if checkpoint is None else 'P2.1 recovery',
            p2_weights_loaded=False, target_steps=self.args.steps, global_batch=72,
            model_initialization_seed=45, validation_enabled=False))
        self.event('launched', attempt=str(attempt), pid=self.process.pid, resume_step=start)

    def run(self):
        self.root.mkdir(parents=True, exist_ok=True)
        with (self.root / 'supervisor.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            write_json(self.root / 'supervisor.json', dict(pid=os.getpid(), thread=self.args.thread,
                interval_seconds=self.args.check_seconds, sample_seconds=self.args.sample_seconds,
                max_auto_restarts=self.args.max_restarts, p2_wait=str(self.args.wait_for_p2),
                initial_source='fresh original LTX2.3 and fresh Extractor/LoRA', started=time.time()))
            self.loop()

    def loop(self):
        while True:
            if (self.root / 'STOP').exists():
                self.stop_requested_at = self.stop_requested_at or time.time()
                if self.attempt and (self.attempt / 'training').is_dir():
                    (self.attempt / 'training/STOP').touch()
                    if self.process.poll() is None:
                        if time.time() - self.stop_requested_at > 600:
                            self.stop_owned_process()
                        time.sleep(self.args.sample_seconds)
                        continue
                self.event('user_stop_complete')
                return
            snapshot = gpu_snapshot(self.cluster)
            self.history.append(snapshot)
            with (self.root / 'gpu_telemetry.jsonl').open('a') as stream:
                stream.write(json.dumps(snapshot) + '\n')
            if self.process is None:
                old = self.args.wait_for_p2
                done = old / 'training/completed.json'
                marker = old / 'checkpoints/extractor_step_000200.complete.json'
                ready = self.root / 'READY.json'
                if not (done.is_file() and marker.is_file() and ready.is_file()):
                    write_json(self.root / 'state.json', dict(phase='waiting_for_p2_step200_and_code_ready',
                        time=time.time(), p2_last=last_row(old / 'training/train_rank0.jsonl')))
                    if list((old / 'training').glob('failed_rank*.json')):
                        self.notify('p2_failed_before_transition')
                    time.sleep(self.args.sample_seconds)
                    continue
                result = json.loads(done.read_text())
                if result['training_completed_steps'] != 200 or not result['ranks_identical']:
                    self.notify('p2_stop_verification_failed')
                    return
                if not idle(snapshot):
                    write_json(self.root / 'state.json', dict(phase='waiting_for_gpu_release', time=time.time()))
                    time.sleep(self.args.sample_seconds)
                    continue
                # The operator validates checkpoint integrity before creating this marker.
                if not (self.root / 'P2_CHECKPOINT_VERIFIED.json').is_file():
                    write_json(self.root / 'state.json', dict(phase='waiting_for_p2_checkpoint_integrity', time=time.time()))
                    time.sleep(self.args.sample_seconds)
                    continue
                self.launch()
                continue
            training = training_snapshot(self.attempt / 'training', self.world)
            now = time.time()
            status = dict(time=now, phase='training', attempt=str(self.attempt), pid=self.process.pid,
                          returncode=self.process.poll(), restarts=self.restarts,
                          rank_steps={str(r['rank']): r['step'] for r in training['ranks']})
            write_json(self.root / 'state.json', status)
            code = self.process.poll()
            if code is not None:
                self.launch_stream.close()
                if code == 0 and training['completed']:
                    result = json.loads((self.attempt / 'training/completed.json').read_text())
                    completed = max((r['step'] for r in training['ranks']), default=0)
                    write_json(self.root / 'completed.json', dict(step=completed, result=result,
                        attempt=str(self.attempt), target_reached=completed >= self.args.steps))
                    write_json(self.root / 'state.json', dict(status, phase='completed_or_stopped'))
                    self.notify('training_completed_or_stopped', step=completed)
                    return
                logs = tail(self.attempt / 'launcher.log') + ''.join(tail(p) for p in (self.attempt / 'agents').glob('*.log'))
                checkpoints = complete_checkpoints(self.root)
                can_recover = (not training['stop'] and checkpoints and self.restarts < self.args.max_restarts
                    and not (self.root / 'DISABLE_AUTO_RECOVERY').exists()
                    and (self.pending_recovery or transient_failure(logs)))
                if not can_recover:
                    write_json(self.root / 'state.json', dict(status, phase='failed_needs_diagnosis'))
                    self.notify('training_failed_needs_diagnosis', returncode=code,
                                latest_checkpoint_step=checkpoints[-1][0] if checkpoints else None)
                    return
                if not idle(snapshot):
                    self.notify('recovery_waiting_for_idle_gpus', returncode=code)
                    time.sleep(self.args.sample_seconds)
                    continue
                self.restarts += 1
                self.pending_recovery = False
                self.notify('automatic_resume', checkpoint_step=checkpoints[-1][0], attempt=self.restarts)
                self.launch(checkpoints[-1])
                continue
            if now - self.last_check >= self.args.check_seconds:
                issues = health_issues(training, snapshot, self.history, now, self.attempt_started, self.world)
                free = shutil.disk_usage(self.root).free
                if free < 20 * 1024**3:
                    issues.append(dict(kind='low_checkpoint_disk_space', free_bytes=free))
                report = dict(status=status, gpu=snapshot, issues=issues, checkpoint_free_bytes=free,
                    checkpoint_steps=[step for step, _ in complete_checkpoints(self.root)])
                write_json(self.root / 'health.json', report)
                with (self.root / 'health_history.jsonl').open('a') as stream:
                    stream.write(json.dumps(report) + '\n')
                self.last_check = now
                if issues:
                    self.notify('health_check_alert', issues=issues)
                kinds = {i['kind'] for i in issues}
                if ('training_stalled' in kinds and 'nonfinite_metrics' not in kinds
                        and all(n['ok'] for n in snapshot['nodes']) and complete_checkpoints(self.root)
                        and self.restarts < self.args.max_restarts and not training['stop']
                        and not (self.root / 'DISABLE_AUTO_RECOVERY').exists()):
                    self.pending_recovery = True
                    self.stop_owned_process()
            time.sleep(self.args.sample_seconds)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--wait-for-p2', type=Path, required=True)
    p.add_argument('--cluster', type=Path, default=ROOT / 'configs/p21_three_nodes.json')
    p.add_argument('--config', type=Path, default=ROOT / 'configs/pipeline21_text_lora.json')
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--steps', type=int, default=1000)
    p.add_argument('--check-seconds', type=int, default=900)
    p.add_argument('--sample-seconds', type=int, default=15)
    p.add_argument('--max-restarts', type=int, default=2)
    p.add_argument('--thread', required=True)
    p.add_argument('--codex', default=shutil.which('codex'))
    args = p.parse_args()
    if min(args.steps, args.check_seconds, args.sample_seconds) < 1 or args.max_restarts < 0 or not args.codex:
        p.error('Invalid supervisor configuration')
    supervisor = Supervisor(args)
    def stop(signum, frame):
        (args.root / 'STOP').touch()
    signal.signal(signal.SIGTERM, stop)
    supervisor.run()


if __name__ == '__main__':
    main()
