#!/usr/bin/env python3
"""P2/P2.1 existing-video metrics using the historical four-metric protocol."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    prepare=sub.add_parser('prepare',help='CPU-only reference manifest and coverage audit')
    prepare.add_argument('--suite',type=Path,required=True)
    prepare.add_argument('--generation',type=Path,required=True)
    prepare.add_argument('--metadata',type=Path,default=Path('/private/liqi_workspace/data/seamless/subset_300h_pure/ltx23_precomputed_720p_area_qualified_20260920/metadata_dev.jsonl'))
    prepare.add_argument('--output',type=Path,required=True)
    for name in ('preflight','score'):
        q=sub.add_parser(name)
        q.add_argument('--config',type=Path,default=ROOT/'configs/p2_generation_metrics.json')
        q.add_argument('--output',type=Path,required=True)
        if name=='score':
            q.add_argument('--plan',type=Path,required=True)
            q.add_argument('--gpu',type=int,required=True)
            q.add_argument('--tensorboard',action=argparse.BooleanOptionalAction,default=True)
    a=p.parse_args(argv)
    # The parent never needs a GPU. Score subprocesses explicitly select one idle GPU.
    os.environ['CUDA_VISIBLE_DEVICES']=''
    sys.path.insert(0,str(ROOT/'src'))
    from omni_av_p2 import generation_metrics as metrics
    if a.command=='prepare':
        r=metrics.prepare(a.suite,a.generation,a.metadata,a.output)
        print(json.dumps(r['coverage'],ensure_ascii=False))
    elif a.command=='preflight':
        r=metrics.preflight(json.loads(a.config.read_text()))
        if a.output.exists(): raise FileExistsError(a.output)
        metrics.write_json(a.output,r)
        print(json.dumps(dict(assets=len(r['asset_sha256']),report=str(a.output))))
    else:
        r=metrics.score(a.plan,json.loads(a.config.read_text()),a.output,a.gpu,a.tensorboard)
        print(json.dumps(dict(complete_dev_set=r['complete_dev_set'],status={k:v['status'] for k,v in r['results'].items()})))
        if any(v['status'] in ('failed','partial') for v in r['results'].values()):
            raise SystemExit(1)


if __name__=='__main__':
    main()
