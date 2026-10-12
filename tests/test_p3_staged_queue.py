import json
import os
from pathlib import Path
import subprocess
import sys
import time

from omni_av_p3_staged.orchestration import prerequisite, free_slots, write
from omni_av_p3_staged.media import audio_tail_padding


def test_prerequisite_requires_all_generation_and_scoring(tmp_path):
    write(tmp_path/'completed.json',dict(status='completed',checkpoint_step=1000))
    write(tmp_path/'generation/completed.json',dict(passed=True))
    summary={m:dict(scored=48) for m in ('reference','gt','omni')}
    write(tmp_path/'report.json',dict(cases=list(range(48)),summary=summary))
    assert prerequisite(tmp_path)
    summary['omni']['scored']=47
    write(tmp_path/'report.json',dict(cases=list(range(48)),summary=summary))
    assert not prerequisite(tmp_path)


def test_never_allocate_another_jobs_gpu():
    cluster=dict(nodes=[dict(name='local',gpus=[0,1,2,3])])
    status=[dict(node='local',ok=True,gpus=[
        dict(index=0,memory_mib=0,util=0,pids=[123]),
        dict(index=1,memory_mib=0,util=0,pids=[]),
        dict(index=2,memory_mib=2000,util=0,pids=[]),
        dict(index=3,memory_mib=0,util=80,pids=[])])]
    assert [g for _,g in free_slots(cluster,status)]==[1]


def test_guard_eof_stops_only_owned_child(tmp_path):
    worker=Path(__file__).resolve().parents[1]/'scripts/pipeline3_staged_worker.py'
    target=tmp_path/'pid'
    code='import os,time;from pathlib import Path;Path('+repr(str(target))+').write_text(str(os.getpid()));time.sleep(120)'
    sentinel=subprocess.Popen([sys.executable,'-c','import time;time.sleep(120)'])
    child=subprocess.Popen([sys.executable,str(worker),'guard','--',sys.executable,'-c',code],stdin=subprocess.PIPE)
    try:
        deadline=time.monotonic()+10
        while not target.exists() and time.monotonic()<deadline:time.sleep(.05)
        assert target.exists()
        pid=int(target.read_text());child.stdin.close();child.wait(timeout=25)
        assert not Path(f'/proc/{pid}').exists()
        assert sentinel.poll() is None
    finally:
        if child.poll() is None:child.terminate();child.wait(timeout=25)
        sentinel.terminate();sentinel.wait(timeout=5)


def test_native_audio_causal_tail_bound():
    # 25 latent ticks decode to 97 mel frames, not 100.
    padding,bound=audio_tail_padding(target_samples=48000,decoded_samples=46560,
        output_rate=48000,latent_frames=25,mel_frames=97,sample_rate=16000,
        hop_length=160,downsample_factor=4,causal=True)
    assert padding==1440 and padding<=bound
