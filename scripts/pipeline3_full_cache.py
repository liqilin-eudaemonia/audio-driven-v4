#!/usr/bin/env python3
"""P3 full manifests, immutable per-sample records, multiple extraction lanes per GPU."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def write_new(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)


def transport(cfg, node, command):
    key = cfg.get('ssh_key', f'/private/ssh_keys/root_{socket.gethostname()}_ed25519')
    multiplex = ['-o','ControlMaster=auto','-o','ControlPersist=600',
                 '-o','ControlPath='+str(Path(cfg['ssh_control_dir'])/'%C')] if cfg.get('ssh_control_dir') else []
    return ['ssh', *multiplex, '-i', key, '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
            '-o', 'StrictHostKeyChecking=accept-new', '-p', str(node.get('ssh_port', 22)),
            f"{node.get('ssh_user','root')}@{node['host']}", shlex.join(command)]


def worker(a):
    os.environ.update(CUDA_VISIBLE_DEVICES=str(a.gpu), OMP_NUM_THREADS='2',
                      TOKENIZERS_PARALLELISM='false', OPENBLAS_NUM_THREADS='1')
    import fcntl
    import torch
    from omni_av_p3.data import contract, extract, load_native
    from omni_av_p3.model import Config
    from omni_av_p3.distributed import sha
    torch.set_num_threads(2)
    cfg = Config(**json.loads((a.output/'config.json').read_text()))
    rows = json.loads((a.output/'manifests'/f'{a.shard:03d}.json').read_text())
    dest = a.output/'shards'/f'{a.shard:03d}'
    dest.mkdir(exist_ok=True)
    claims=dest/'claims'
    claims.mkdir(exist_ok=True)
    runtime = processor = None
    for i, row in enumerate(rows):
        with (claims/f'{i:06d}.lock').open('a') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                continue
            marker = dest/f'{i:06d}.json'
            if marker.exists():
                entry = json.loads(marker.read_text())
                if entry['sample_id'] != row['observation']['sample_id'] or not Path(entry['path']).is_file():
                    raise ValueError('Existing cache record changed')
                # Integrity hashes are retained; the trainer rechecks bytes before first use.
                # Do not reread hundreds of GB in every cooperating extraction lane.
                continue
            if runtime is None:
                runtime, processor = load_native(cfg, 'cuda:0')
                # Extraction uses towers, embedding lookup and MRoPE only, never decoder blocks.
                runtime.thinker.model.layers = torch.nn.ModuleList()
                runtime.thinker.lm_head = torch.nn.Identity()
                torch.cuda.empty_cache()
            start = time.monotonic()
            payload = extract(runtime, processor, row)
            payload['contract'] = contract(cfg)
            path = dest/f'{i:06d}_{time.time_ns()}.pt'
            with path.open('xb') as f:
                torch.save(payload, f)
                f.flush()
                os.fsync(f.fileno())
            entry = dict(path=str(path), sha256=sha(path), sample_id=payload['sample_id'],
                         split=row['split'], prefix_tokens=payload['layout']['prefix'].shape[1],
                         video_tokens=payload['video_tokens'], audio_tokens=payload['audio_tokens'],
                         duration_s=payload['duration_s'], seconds=time.monotonic()-start)
            write_new(marker, entry)
            print(json.dumps(dict(shard=a.shard, index=i, **entry)), flush=True)
    print(json.dumps(dict(completed=True, shard=a.shard, count=len(rows))), flush=True)


def setup(a, cfg, slots):
    from omni_av_p3.model import Config
    from omni_av_p3.data import contract
    from omni_av_p3.distributed import sha
    train, dev = (json.loads(p.read_text()) for p in (a.train_manifest, a.dev_manifest))
    for split, rows, expected in [('train',train,91009),('dev',dev,48)]:
        if len(rows)!=expected or len({r['observation']['sample_id'] for r in rows})!=expected or any(r['split']!=split for r in rows):
            raise ValueError('Full manifest count/identity mismatch')
        if any(not 0 < r['duration_s'] <= 24 for r in rows):
            raise ValueError('Invalid full duration')
    if {r['observation']['sample_id'] for r in train} & {r['observation']['sample_id'] for r in dev}:
        raise ValueError('Train/Dev overlap')
    conf = Config(train_samples=len(train), dev_samples=len(dev), max_duration=24., steps=1000)
    identity = dict(train_sha256=sha(a.train_manifest), dev_sha256=sha(a.dev_manifest),
                    slots=len(slots), config=asdict(conf))
    if a.resume:
        if json.loads((a.output/'identity.json').read_text()) != json.loads(json.dumps(identity)):
            raise ValueError('Cache resume identity mismatch')
        return
    a.output.mkdir(parents=True, exist_ok=False)
    for d in ('manifests','shards','logs'):
        (a.output/d).mkdir()
    write_new(a.output/'identity.json',identity)
    write_new(a.output/'config.json',asdict(conf))
    write_new(a.output/'contract.json',contract(conf))
    write_new(a.output/'selection.json',dict(train=train,dev=dev))
    # Fixed duration-balanced shards, without dropping empty replies or long examples.
    groups=[[] for _ in slots]
    loads=[0.]*len(slots)
    for row in sorted(train+dev,key=lambda r:-r['duration_s']):
        j=min(range(len(slots)),key=lambda k:loads[k])
        groups[j].append(row)
        loads[j]+=row['duration_s']
    for j,rows in enumerate(groups):
        write_new(a.output/'manifests'/f'{j:03d}.json',rows)


def merge(a):
    from omni_av_p3.distributed import sha
    selected=json.loads((a.output/'selection.json').read_text())
    entries={}
    for marker in (a.output/'shards').glob('*/*.json'):
        e=json.loads(marker.read_text())
        if e['sample_id'] in entries:
            raise ValueError('Duplicate extracted sample')
        if not Path(e['path']).is_file():
            raise FileNotFoundError(e['path'])
        entries[e['sample_id']]=e
    index=dict(contract=json.loads((a.output/'contract.json').read_text()))
    for split in ('train','dev'):
        index[split]=[entries[r['observation']['sample_id']] for r in selected[split]]
        if any(e['split']!=split for e in index[split]):
            raise ValueError('Split mismatch')
    if len(entries)!=sum(len(index[s]) for s in ('train','dev')):
        raise ValueError('Unexpected sample coverage')
    write_new(a.output/'index.json',index)
    write_new(a.output/'completed.json',dict(train=len(index['train']),dev=len(index['dev']),
        index_sha256=sha(a.output/'index.json'),bytes=sum(Path(e['path']).stat().st_size for e in entries.values()),
        note='Full exact manifest coverage; per-sample SHA stored at extraction; trainer verifies before first use'))


def controller(a):
    cfg=json.loads(a.cluster.read_text())
    slots=[(n,g,l) for n in cfg['nodes'] for g in n['gpus'] for l in range(a.lanes)]
    # Preflight all nodes before starting any extraction process.
    def check(node):
        code="import sys;sys.path.insert(0,"+repr(str(ROOT/'src'))+");from omni_av_p2.launch import idle_gpus;idle_gpus("+repr(node['gpus'])+")"
        for attempt in range(6):
            result=subprocess.run(transport(cfg,node,[cfg['python'],'-c',code]),timeout=60)
            if result.returncode==0:return
            if result.returncode!=255 or attempt==5:
                raise RuntimeError(f'Preflight failed on {node["name"]}: code {result.returncode}')
            time.sleep(2*(attempt+1))
    with ThreadPoolExecutor(max_workers=len(cfg['nodes'])) as pool:
        list(pool.map(check,cfg['nodes']))
    setup(a,cfg,slots)
    processes=[]
    try:
        for node in cfg['nodes']:
            for gpu in node['gpus']:
                shards=[j for j,(n,g,l) in enumerate(slots) if n['host']==node['host'] and g==gpu]
                cmd=[cfg['python'],str(ROOT/'scripts/pipeline3_cluster.py'),'--guard',cfg['python'],
                     str(Path(__file__).resolve()),'--gpu-manager','--output',str(a.output),'--gpu',str(gpu),
                     '--shard-list',','.join(map(str,shards)),'--workers-per-shard',str(a.workers_per_shard)]
                for attempt in range(5):
                    logfile=a.output/'logs'/f'manager_{node["name"]}_{gpu}_{time.time_ns()}.log'
                    with logfile.open('x') as log:
                        p=subprocess.Popen(transport(cfg,node,cmd),stdin=subprocess.PIPE,stdout=log,stderr=subprocess.STDOUT)
                    deadline=time.monotonic()+90
                    while p.poll() is None and 'CACHE_MANAGER_READY' not in logfile.read_text():
                        if time.monotonic()>deadline:
                            processes.append(p)
                            raise TimeoutError('GPU manager did not acknowledge SSH startup')
                        time.sleep(.2)
                    if 'CACHE_MANAGER_READY' in logfile.read_text():
                        processes.append(p)
                        break
                    p.stdin.close()
                    if p.returncode!=255 or attempt==4:
                        raise RuntimeError(f'GPU manager startup failed: {logfile.read_text()}')
                    time.sleep(2*(attempt+1))
        while any(p.poll() is None for p in processes):
            if any(p.poll() not in (None,0) for p in processes):
                raise RuntimeError('Cache worker failed; inspect shard logs; incomplete files preserved')
            time.sleep(2)
        if any(p.returncode!=0 for p in processes):
            raise RuntimeError('Cache worker failed')
        merge(a)
    finally:
        for p in processes:
            p.stdin.close()
        for p in processes:
            try: p.wait(timeout=30)
            except subprocess.TimeoutExpired: p.terminate()


def gpu_manager(a):
    """One SSH transport per GPU; local children remain in the owned guard process group."""
    print('CACHE_MANAGER_READY',flush=True)
    processes=[]
    try:
        for replica in range(a.workers_per_shard):
            for shard in map(int,a.shard_list.split(',')):
                command=[sys.executable,str(Path(__file__).resolve()),'--worker','--output',str(a.output),
                         '--shard',str(shard),'--gpu',str(a.gpu)]
                with (a.output/'logs'/f'{shard:03d}_{time.time_ns()}.log').open('x') as log:
                    processes.append(subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT))
                time.sleep(.15)
        while any(p.poll() is None for p in processes):
            if any(p.poll() not in (None,0) for p in processes):
                raise RuntimeError(f'Local extraction failed: {[p.poll() for p in processes]}')
            time.sleep(1)
        if any(p.returncode!=0 for p in processes): raise RuntimeError('Local extraction failed')
    finally:
        for p in processes:
            if p.poll() is None: p.terminate()
        for p in processes:
            try: p.wait(timeout=20)
            except subprocess.TimeoutExpired: p.kill()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cluster',type=Path)
    p.add_argument('--train-manifest',type=Path)
    p.add_argument('--dev-manifest',type=Path)
    p.add_argument('--lanes',type=int,default=4)
    p.add_argument('--workers-per-shard',type=int,default=1)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--worker',action='store_true')
    p.add_argument('--gpu-manager',action='store_true')
    p.add_argument('--shard-list')
    p.add_argument('--gpu',type=int)
    p.add_argument('--shard',type=int)
    a=p.parse_args()
    if a.worker: worker(a)
    elif a.gpu_manager: gpu_manager(a)
    else: controller(a)


if __name__=='__main__': main()
