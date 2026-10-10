"""Cooperating extraction lanes must publish every sample exactly once."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
from pathlib import Path
import time
from types import SimpleNamespace

import torch


def test_extraction_lanes_claim_each_sample_once_and_resume(tmp_path,monkeypatch):
    script=Path(__file__).resolve().parents[1]/'scripts/pipeline3_full_cache.py'
    spec=importlib.util.spec_from_file_location('p3_full_cache_test',script)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    from omni_av_p3 import data
    monkeypatch.setattr(torch,'set_num_threads',lambda n:None)
    monkeypatch.setattr(torch.cuda,'empty_cache',lambda:None)
    monkeypatch.setattr(data,'contract',lambda cfg:{'test':True})
    monkeypatch.setattr(data,'load_native',lambda cfg,device:(SimpleNamespace(
        thinker=SimpleNamespace(model=SimpleNamespace()),config=cfg),None))
    seen=[]
    def extract(runtime,processor,row):
        seen.append(row['observation']['sample_id'])
        time.sleep(.01)
        return dict(sample_id=row['observation']['sample_id'],layout=dict(prefix=torch.zeros(1,2,4)),
                    video_tokens=2,audio_tokens=2,duration_s=1.)
    monkeypatch.setattr(data,'extract',extract)
    (tmp_path/'manifests').mkdir();(tmp_path/'shards').mkdir()
    (tmp_path/'config.json').write_text('{}')
    rows=[dict(observation=dict(sample_id=f'sample{i}'),split='train') for i in range(12)]
    (tmp_path/'manifests/000.json').write_text(json.dumps(rows))
    args=SimpleNamespace(output=tmp_path,shard=0,gpu=0)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda _:module.worker(args),range(4)))
    assert Counter(seen)==Counter(r['observation']['sample_id'] for r in rows)
    markers=list((tmp_path/'shards/000').glob('*.json'))
    assert len(markers)==12
    for marker in markers:
        value=json.loads(marker.read_text())
        assert Path(value['path']).is_file()
    module.worker(args)
    assert len(seen)==12
