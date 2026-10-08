#!/usr/bin/env python3
"""Observe the resumed P2 four-GPU job; queue alerts without restarting it."""
import argparse
from collections import deque
import json
import os
from pathlib import Path
import subprocess
import time

from pipeline21_supervise import gpu_snapshot, health_issues, training_snapshot, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--thread', required=True)
    p.add_argument('--once', action='store_true')
    a = p.parse_args()
    cluster = dict(ssh_key='/private/ssh_keys/root_dedicated-developjob-js-public-huvlx_ed25519',
        nodes=[dict(name='p2_gpu4', host='172.29.104.194', gpus=[0, 1, 2, 3])])
    started = json.loads((a.root / 'controller.json').read_text())['started']
    history = deque(maxlen=8)
    last_check = 0
    last_alert = 0
    write_json(a.root / 'monitor.json', dict(pid=os.getpid(), sample_seconds=15,
        check_seconds=900, stale_seconds=1800, auto_restart=False))
    while not (a.root / 'STOP_MONITOR').exists():
        snapshot = gpu_snapshot(cluster)
        history.append(snapshot)
        with (a.root / 'gpu_telemetry.jsonl').open('a') as stream:
            stream.write(json.dumps(snapshot) + '\n')
        training = training_snapshot(a.root / 'training', 4)
        now = time.time()
        write_json(a.root / 'state.json', dict(time=now, rank_steps={r['rank']:r['step'] for r in training['ranks']},
            completed=training['completed'], failures=training['failures']))
        if now-last_check >= 900 or training['failures'] or a.once:
            issues = health_issues(training, snapshot, history, now, started, 4, stale_seconds=1800)
            report = dict(time=now, gpu=snapshot, training=training, issues=issues)
            write_json(a.root / 'health.json', report)
            with (a.root / 'health_history.jsonl').open('a') as stream:
                stream.write(json.dumps(report) + '\n')
            last_check = now
            if issues and now-last_alert >= 900:
                message = ('[P2四卡续训监控] 请核查 ' + str(a.root) + '/health.json 和最新训练日志，'
                    '诊断并解决仍存在的问题。仅限172.29.104.194的0–3卡本次P2任务；'
                    '保留step200续训、global batch72、每100步保存、不验证的配置，不停止其他任务。'
                    + json.dumps(issues, ensure_ascii=False)[:1000])
                result = subprocess.run(['/usr/local/nvm/versions/node/v20.20.2/bin/codex','queue',
                    '--thread',a.thread,'--message',message],capture_output=True,text=True,timeout=30)
                with (a.root/'alerts.jsonl').open('a') as stream:
                    stream.write(json.dumps(dict(time=now,issues=issues,returncode=result.returncode,
                        stdout=result.stdout[-1000:],stderr=result.stderr[-1000:]))+'\n')
                last_alert=now
        if training['completed'] or a.once:
            return
        time.sleep(15)


if __name__ == '__main__':
    main()
