import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch

from omni_av.data import Observation, TextPlan, file_hash
import omni_av.omni as omni
import omni_av_p2.cache_refresh as refresh
from omni_av_p2.connector import ConditionInputs


def test_user_message_has_no_timing_and_is_independent_of_bounds(tmp_path, monkeypatch):
    image = tmp_path / 'ref.png'
    Image.new('RGB', (32, 32)).save(image)
    calls = []
    monkeypatch.setattr(omni, 'read_audio', lambda *a, **kw: np.zeros(16000, dtype=np.float32))
    class Processor:
        def apply_chat_template(self, messages, **kwargs):
            calls.append(messages)
            return json.dumps(messages)
        def __call__(self, **kwargs):
            return kwargs
    for end, budget in [(2.125, 8), (4.875, 16)]:
        obs = Observation('s', 'audio.wav', str(image), 0, end, budget)
        omni.prepare_observation(Processor(), obs, SimpleNamespace(vision_max_pixels=65536))
    assert calls[0] == calls[1]
    user = calls[0][1]
    assert [v['type'] for v in user['content']] == ['image', 'audio', 'text']
    text = user['content'][-1]['text']
    assert text == omni.OBSERVATION_USER_PROMPT
    assert not any(c.isdigit() for c in text)
    assert not any(word in text.lower() for word in ['seconds', 'duration', 'finishes', 'window', 'timestamp'])


def test_prompt_fingerprint_tracks_system_and_user_changes(monkeypatch):
    original = omni.observation_prompt_contract()
    monkeypatch.setattr(omni, 'OBSERVATION_USER_PROMPT', omni.OBSERVATION_USER_PROMPT + ' Changed.')
    assert omni.observation_prompt_contract() != original
    monkeypatch.setattr(omni, 'SYSTEM', omni.SYSTEM + ' Changed.')
    assert omni.observation_prompt_contract()['system_sha256'] != original['system_sha256']


def fixture_job(tmp_path, monkeypatch):
    source = tmp_path / 'old'
    source.mkdir()
    gt = tmp_path / 'gt.json'
    gt.write_text(json.dumps(dict(avatar_plan=dict(sample_id='s', response=dict(present=True, text='hello')))))
    example = dict(split='dev', duration_s=2, gt_path=str(gt),
        text_plan=dict(reply='hello', delivery='', behavior='', vocal_events=''),
        observation=dict(sample_id='s', speaker_audio='audio.wav', listener_image='image.jpg',
                         speaker_start_s=0, speaker_end_s=1, output_duration_s=8))
    current = dict(version='test', config=dict(target_dim=2), observation_prompt=omni.observation_prompt_contract())
    old = refresh.legacy_contract(current)
    condition = dict(hidden=torch.ones(1, 4, 3, 2), hidden_valid=torch.ones(1, 3, dtype=torch.bool),
        layer_ids=torch.tensor([8, 18, 28, 35]), video_coordinates=torch.zeros(1, 2, 3),
        audio_coordinates=torch.zeros(1, 2, 3), video_valid=torch.ones(1, 2, dtype=torch.bool),
        audio_valid=torch.ones(1, 2, dtype=torch.bool))
    payload = dict(contract=old, example=example, split='dev', sample_id='s', condition=condition,
        targets=dict(video=torch.ones(1, 2, 2), audio=torch.ones(1, 2, 2),
                     video_valid=torch.ones(1, 2, dtype=torch.bool), audio_valid=torch.ones(1, 2, dtype=torch.bool)),
        gt_sha256=file_hash(gt), window=dict(duration=2), source='paired_gt_text_frozen_omni')
    feature = source / 'features_0000.pt'
    torch.save(payload, feature)
    text = source / 'text_0000.pt'
    text.write_bytes(b'text sentinel')
    monkeypatch.setattr(refresh, 'contract', lambda cfg: current)
    monkeypatch.setattr(refresh, 'checked_text', lambda src, dst, plan: dst.write_bytes(src.read_bytes()))
    monkeypatch.setattr(refresh, 'load_omni', lambda cfg: (None, None))
    new = copy.deepcopy(condition)
    new['hidden'] = torch.full((1, 4, 2, 2), 2.)
    new['hidden_valid'] = torch.ones(1, 2, dtype=torch.bool)
    monkeypatch.setattr(refresh, 'extract_condition', lambda *a: ConditionInputs(**new))
    job = dict(split='dev', source_contract=old, items=[dict(example=example,
        entry=dict(features=str(feature), text=str(text), sample_id='s', split='dev'))])
    return source, feature, payload, job, current, new


def test_refresh_deletes_only_after_verified_replacement_and_resumes(tmp_path, monkeypatch):
    source, feature, payload, job, current, new = fixture_job(tmp_path, monkeypatch)
    output = tmp_path / 'new'
    result = refresh.refresh_job(job, output, source, None, delete_old=True)
    assert not feature.exists()
    saved = torch.load(result['entries'][0]['features'], weights_only=True)
    assert saved['contract'] == current
    assert torch.equal(saved['condition']['hidden'], new['hidden'])
    assert all(torch.equal(saved['targets'][k], v) for k, v in payload['targets'].items())
    assert refresh.refresh_job(job, output, source, None, delete_old=True) == result


def test_layout_change_preserves_old_feature(tmp_path, monkeypatch):
    source, feature, payload, job, current, new = fixture_job(tmp_path, monkeypatch)
    new['video_coordinates'] += 1
    before = feature.read_bytes()
    with pytest.raises(ValueError, match='layout changed'):
        refresh.refresh_job(job, tmp_path / 'new', source, None, delete_old=True)
    assert feature.read_bytes() == before
    assert not (tmp_path / 'new/features_0000.pt').exists()


def test_failed_atomic_save_keeps_source(tmp_path, monkeypatch):
    source, feature, payload, job, current, new = fixture_job(tmp_path, monkeypatch)
    def fail(*args):
        raise OSError('disk full')
    monkeypatch.setattr(refresh, 'atomic_save', fail)
    with pytest.raises(OSError, match='disk full'):
        refresh.refresh_job(job, tmp_path / 'new', source, None, delete_old=True)
    assert feature.is_file()


def test_corrupt_new_archive_is_repaired_only_with_intact_source(tmp_path, monkeypatch):
    source, feature, payload, job, current, new = fixture_job(tmp_path, monkeypatch)
    output = tmp_path / 'new'
    output.mkdir()
    path = output / 'features_0000.pt'
    path.write_bytes(b'PK\x03\x04incomplete zip archive')
    refresh.refresh_job(job, output, source, None, delete_old=True)
    assert not feature.exists()
    assert len(list(output.glob('*.invalid.*'))) == 1
    assert torch.load(path, weights_only=True)['contract'] == current


def test_corrupt_new_archive_without_source_is_never_discarded(tmp_path, monkeypatch):
    source, feature, payload, job, current, new = fixture_job(tmp_path, monkeypatch)
    feature.unlink()
    output = tmp_path / 'new'
    output.mkdir()
    path = output / 'features_0000.pt'
    path.write_bytes(b'PK\x03\x04incomplete zip archive')
    with pytest.raises(FileNotFoundError):
        refresh.refresh_job(job, output, source, None, delete_old=True)
    assert path.exists() and not list(output.glob('*.invalid.*'))


def test_retirement_rejects_changed_and_outside_files(tmp_path):
    root = tmp_path / 'cache'
    root.mkdir()
    path = root / 'features_0000.pt'
    path.write_bytes(b'old')
    before = path.stat()
    path.write_bytes(b'changed-size')
    with pytest.raises(ValueError, match='Source changed'):
        refresh.delete_verified_source(path, root, before)
    with pytest.raises(ValueError, match='outside'):
        refresh.delete_verified_source(path, root / 'another', path.stat())
    assert path.exists()


def test_reusable_text_is_hardlinked_and_prompt_checked(tmp_path):
    plan = TextPlan('hello')
    source, destination = tmp_path / 'text.pt', tmp_path / 'linked.pt'
    torch.save(dict(prompt=plan.render_prompt(), video=torch.zeros(1, 1024, 4096, dtype=torch.bfloat16),
        audio=torch.zeros(1, 1024, 2048, dtype=torch.bfloat16), mask=torch.ones(1, 1024, dtype=torch.bool)), source)
    refresh.checked_text(source, destination, plan)
    assert source.stat().st_ino == destination.stat().st_ino
    with pytest.raises(ValueError, match='Gemma'):
        refresh.checked_text(source, tmp_path / 'bad.pt', TextPlan('different'))
