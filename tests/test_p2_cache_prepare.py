from concurrent.futures import ThreadPoolExecutor
import threading
import time

import pytest
import torch

from omni_av_p2.cache_prepare import read_complete, text_slot
from omni_av_p2.checkpoint import atomic_json_new


def test_partial_cache_is_preserved_and_valid_cache_is_reused(tmp_path):
    path = tmp_path/'features.pt'
    assert read_complete(path) is None
    path.write_bytes(b'partial torch archive')
    assert read_complete(path) is None
    assert not path.exists()
    assert next(tmp_path.glob('*.interrupted.*')).read_bytes() == b'partial torch archive'
    torch.save({'value': torch.tensor([2.0])}, path)
    torch.testing.assert_close(read_complete(path)['value'], torch.tensor([2.0]))
    assert path.is_file()


def test_json_publish_never_replaces_previous_index(tmp_path):
    import json
    path = tmp_path/'index.json'
    with pytest.raises(TypeError):
        atomic_json_new(path, {'invalid': object()})
    assert not path.exists() and not list(tmp_path.iterdir())
    atomic_json_new(path, {'samples': ['a']})
    with pytest.raises(FileExistsError):
        atomic_json_new(path, {'samples': ['b']})
    assert json.loads(path.read_text()) == {'samples': ['a']}
    assert list(tmp_path.iterdir()) == [path]


def test_gemma_slots_bound_concurrency_and_release_on_error(tmp_path):
    lock = threading.Lock()
    active, maximum = 0, 0
    def task(i):
        nonlocal active, maximum
        with text_slot(tmp_path, gpu=0, slots=2):
            with lock:
                active += 1
                maximum = max(maximum, active)
            time.sleep(.05)
            with lock:
                active -= 1
    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(task,range(6)))
    assert 1 <= maximum <= 2 and active == 0
    with pytest.raises(RuntimeError):
        with text_slot(tmp_path, gpu=0, slots=1):
            raise RuntimeError('Interrupted extraction')
    with text_slot(tmp_path, gpu=0, slots=1):
        pass
