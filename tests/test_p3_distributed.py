from copy import deepcopy
import json

import pytest
import torch

from test_p3 import tiny
from omni_av_p3.distributed import BatchAssignments, GlobalOrder, Samples, TrainableDDP, ordinal_for, read_checkpoint, tiny_rows, validate_index
from omni_av_p3.launch import cluster_nodes, signature


def test_ddp_wrapper_registers_only_trainable_aliases():
    raw = tiny("lora")
    wrapped = TrainableDDP(raw)
    assert {id(p) for p in wrapped.parameters()} == {id(p) for p in raw.parameters() if p.requires_grad}
    assert not any(not p.requires_grad for p in wrapped.parameters())
    assert all(not name.startswith("raw") for name in wrapped.state_dict())


def test_global_order_resume_and_rank_partition_match_serial_exposure():
    order = GlobalOrder(17, 23)
    serial = [order[i] for i in range(120)]
    reconstructed = []
    for step in range(1, 11):
        for micro in range(3):
            for rank in range(4):
                reconstructed.append(order[ordinal_for(step, micro, rank, 4, 3)])
    assert reconstructed == serial
    resumed = GlobalOrder(17, 23)
    assert [resumed[i] for i in range(60, 120)] == serial[60:]
    assert sorted(serial[:17]) == list(range(17))
    assert sorted(serial[17:34]) == list(range(17))


def test_remote_master_sorted_first_and_rejects_heterogeneous_topology():
    nodes = [dict(name="local", host="1", local=True, role="slave", gpus=[2, 3]),
             dict(name="remote", host="2", role="master", gpus=[0, 1])]
    assert cluster_nodes(dict(nodes=nodes))[0]["name"] == "remote"
    bad = deepcopy(nodes)
    bad[1]["gpus"] = [0]
    with pytest.raises(ValueError, match="equal GPU"):
        cluster_nodes(dict(nodes=bad))
    bad = deepcopy(nodes)
    bad[0]["role"] = "master"
    with pytest.raises(ValueError, match="Exactly one"):
        cluster_nodes(dict(nodes=bad))


def test_checkpoint_rejects_unpublished_or_wrong_identity(tmp_path):
    from omni_av_p3.distributed import sha, FORMAT
    path = tmp_path / "test.pt"
    torch.save(dict(format=FORMAT, identity=dict(world_size=2), step=3), path)
    with pytest.raises(ValueError, match="incomplete"):
        read_checkpoint(path, dict(world_size=2))
    path.with_suffix(".complete.json").write_text(json.dumps(dict(sha256=sha(path))))
    assert read_checkpoint(path, dict(world_size=2))["step"] == 3
    with pytest.raises(ValueError, match="contract"):
        read_checkpoint(path, dict(world_size=4))


def test_index_and_shared_source_contract():
    good = dict(train=[dict(sample_id="a")], dev=[dict(sample_id="b")])
    validate_index(good)
    with pytest.raises(ValueError, match="overlap"):
        validate_index(dict(train=good["train"], dev=good["train"]))
    with pytest.raises(ValueError, match="duplicate"):
        validate_index(dict(train=good["train"] * 2, dev=good["dev"]))
    assert "src/omni_av_p3/distributed.py" in signature()


def test_balancing_preserves_global_exposure_noise_and_reduces_skew():
    order=list(range(12))
    costs=[100,90,10,8,80,70,7,6,60,50,5,4]
    plain=BatchAssignments(order,costs,4,3)
    balanced=BatchAssignments(order,costs,4,3,True)
    a,b=plain[1],balanced[1]
    assert sorted(j for row in b for j in row)==list(range(12))
    assert len(b)==3 and all(len(row)==4 for row in b)
    load=lambda rows:[sum(costs[rows[m][r]] for m in range(3)) for r in range(4)]
    assert max(load(b))<max(load(a))
    assert b==BatchAssignments(order,costs,4,3,True)[1]


def test_prefetch_matches_sync_and_keeps_lru_bounded():
    rows=tiny_rows(7,'train')
    entries=[dict(sample_id=r['sample_id']) for r in rows]
    a=Samples(entries,'train',torch.device('cpu'),2,rows,prefetch=True)
    b=Samples(entries,'train',torch.device('cpu'),0,rows)
    try:
        a.schedule([0,1,0,2])
        assert len(a.pending)==3
        for i in [0,1,0,2,4]:
            torch.testing.assert_close(a[i].targets,b[i].targets,atol=0,rtol=0)
            assert len(a.cache)<=2
        assert not a.pending
    finally: a.close()
