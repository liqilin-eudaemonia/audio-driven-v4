#!/usr/bin/env python3
"""Explicitly migrate P2 DDP topology while preserving global batch and optimizer.

This is not bitwise continuation: floating-point reduction order changes. P2
re-seeds AV/aux noise by global sample ordinal, so sample/noise membership stays
unchanged when world * accumulation is constant. Never modify the source file.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def repartition(data, world, accumulation):
    from omni_av_p2 import VERSION
    identity = data['identity']
    old_world = identity['world_size']
    if data['version'] != VERSION or identity.get('trainer') != 'p2_ddp_v1':
        raise ValueError('Only full P2 DDP checkpoints are supported')
    if min(world, accumulation) < 1 or world * accumulation != old_world * identity['accumulation']:
        raise ValueError('Global batch must remain unchanged')
    if 'optimizer' not in data or len(data.get('rank_rng') or []) != old_world:
        raise ValueError('Full optimizer and per-rank RNG states required')
    result = dict(data)
    result['identity'] = dict(identity, world_size=world, accumulation=accumulation)
    # Rank-local state has no one-to-one continuation across topology changes.
    # Select evenly spaced old ranks; sample-local Torch RNG is re-seeded before
    # each stochastic training forward, including the auxiliary generator.
    mapping = [rank * old_world // world for rank in range(world)]
    result['rank_rng'] = [copy.deepcopy(data['rank_rng'][rank]) for rank in mapping]
    result['topology_migration'] = dict(original_identity=identity, rank_rng_mapping=mapping,
        global_batch=world * accumulation, bitwise_equivalent=False,
        noise_policy='unchanged per-global-sample ordinal seed')
    return result


def main():
    import torch
    from omni_av_p2.checkpoint import atomic_save, atomic_json_new
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--world-size', type=int, required=True)
    p.add_argument('--accumulation', type=int, required=True)
    a = p.parse_args()
    if a.source.resolve() == a.output.resolve():
        p.error('Source must remain immutable')
    torch.set_num_threads(8)
    data = torch.load(a.source, map_location='cpu', weights_only=True, mmap=True)
    migrated = repartition(data, a.world_size, a.accumulation)
    if not all(torch.isfinite(t).all() for t in data['model'].values()):
        raise ValueError('Nonfinite model tensor')
    for state in data['optimizer']['state'].values():
        if any(torch.is_tensor(v) and not torch.isfinite(v).all() for v in state.values()):
            raise ValueError('Nonfinite optimizer tensor')
        if int(state['step']) != data['step']:
            raise ValueError('Optimizer step does not match checkpoint')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_save(a.output, migrated)
    check = torch.load(a.output, map_location='cpu', weights_only=True, mmap=True)
    def same(x, y):
        if torch.is_tensor(x):
            return torch.equal(x, y)
        if isinstance(x, dict):
            return x.keys() == y.keys() and all(same(x[k], y[k]) for k in x)
        if isinstance(x, (list, tuple)):
            return len(x) == len(y) and all(same(i, j) for i, j in zip(x, y))
        return x == y
    assert same(check['model'], data['model']) and same(check['optimizer'], data['optimizer'])
    with a.source.open('rb') as stream:
        source_sha256 = hashlib.file_digest(stream, 'sha256').hexdigest()
    record = dict(migrated['topology_migration'], source=str(a.source.resolve()),
        source_sha256=source_sha256, output=str(a.output.resolve()), step=data['step'],
        new_world_size=a.world_size, new_accumulation=a.accumulation,
        model_and_optimizer_exactly_equal=True)
    atomic_json_new(a.output.with_suffix('.migration.json'), record)
    print(json.dumps({k:v for k,v in record.items() if k != 'original_identity'}))


if __name__ == '__main__':
    main()
