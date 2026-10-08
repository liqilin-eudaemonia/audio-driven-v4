import importlib.util
from pathlib import Path

import pytest

from omni_av_p2 import VERSION
from omni_av_p2.performance import balanced_sample_indices

spec = importlib.util.spec_from_file_location('migration', Path(__file__).resolve().parents[1] / 'scripts/pipeline2_repartition_checkpoint.py')
migration = importlib.util.module_from_spec(spec)
spec.loader.exec_module(migration)


def checkpoint():
    return dict(version=VERSION, step=200, identity=dict(trainer='p2_ddp_v1', world_size=24,
        accumulation=3, seed=45), rank_rng=[dict(rank=i) for i in range(24)],
        model={'weight': 7}, optimizer={'step': 200})


def test_resume_preserves_global_sample_and_noise_membership():
    original = checkpoint()
    moved = migration.repartition(original, 4, 18)
    assert original['identity']['world_size'] == 24
    assert len(original['rank_rng']) == 24
    assert moved['model'] == original['model'] and moved['optimizer'] == original['optimizer']
    assert moved['step'] == 200 and len(moved['rank_rng']) == 4
    costs = [9, 2, 5, 7, 100, 4, 6]
    for step in (201, 300, 1000):
        old = sorted(i for rank in balanced_sample_indices(step, 24, 3, costs) for i in rank)
        new = sorted(i for rank in balanced_sample_indices(step, 4, 18, costs) for i in rank)
        assert old == new == list(range((step-1)*72, step*72))
        assert [45+i+1 for i in old] == [45+i+1 for i in new]


def test_rejects_batch_change_or_missing_resume_state():
    with pytest.raises(ValueError, match='Global batch'):
        migration.repartition(checkpoint(), 4, 3)
    partial = checkpoint()
    del partial['optimizer']
    with pytest.raises(ValueError, match='Full optimizer'):
        migration.repartition(partial, 4, 18)
