from contextlib import nullcontext
from dataclasses import asdict
import json
from types import SimpleNamespace

from PIL import Image
import pytest
import torch

from omni_av.data import Observation, TextPlan
from omni_av_p2 import VERSION
from omni_av_p2.connector import ConditionInputs, SharedCondition
import omni_av_p2.paired_validation as paired


def media_case(tmp_path):
    image = tmp_path / 'reference.png'
    Image.new('RGB', (64, 96)).save(image)
    audio = tmp_path / 'speaker.wav'
    audio.write_bytes(b'input waveform placeholder; CPU preparation only hashes it')
    gt = tmp_path / 'gt.json'
    gt.write_text(json.dumps(dict(avatar_plan=dict(sample_id='dev1', response=dict(present=True, text='GT words'),
        semantics=dict(tone='CALM'), visual_action_description='GT gesture',
        vocal_track=[dict(state='PARALINGUISTIC', subtype='INHALE')]))))
    example = dict(split='dev', duration_s=7.8, gt_path=str(gt),
        observation=asdict(Observation('dev1', str(audio), str(image), 0, 2.5, 8)),
        text_plan=asdict(TextPlan('GT words', 'calm tone', 'GT gesture', 'inhale')),
        target_height=96, target_width=64, target_video='/not/read/target.mp4', target_audio='/not/read/listener.wav')
    metadata = dict(sample_id='dev1', split='dev', target_frames=233, fps=30., target_height=96, target_width=64)
    manifest, meta = tmp_path / 'dev.json', tmp_path / 'metadata.jsonl'
    manifest.write_text(json.dumps([example]))
    meta.write_text(json.dumps(metadata) + '\n')
    return example, metadata, manifest, meta


def make_condition(n=3, value=1.):
    return ConditionInputs(hidden=torch.full((1, 4, n, 2), value),
        hidden_valid=torch.ones(1, n, dtype=torch.bool), layer_ids=torch.tensor([8, 18, 28, 35]),
        video_coordinates=torch.zeros(1, 2, 3), audio_coordinates=torch.zeros(1, 2, 3),
        video_valid=torch.ones(1, 2, dtype=torch.bool), audio_valid=torch.ones(1, 2, dtype=torch.bool))


def test_prepare_has_full_gt_but_no_target_media_or_training_cache(tmp_path, monkeypatch):
    example, metadata, manifest, meta = media_case(tmp_path)
    monkeypatch.setattr(paired, 'contract', lambda cfg: {'prompt': 'no-time'})
    cfg = SimpleNamespace(to_dict=lambda: {})
    suite = paired.prepare_suite(manifest, meta, tmp_path / 'suite', cfg)
    case = suite['cases'][0]
    assert suite['modes'] == ['gt', 'omni'] and not suite['generated']
    assert case['gt_plan'] == example['text_plan']
    assert case['geometry']['frames'] == 233 and case['geometry']['duration_s'] == 233 / 30
    assert case['geometry']['external_budget_s'] == 8
    assert case['geometry']['height'] != case['geometry']['width']
    serialized = json.dumps(suite)
    assert '/not/read/' not in serialized and 'target_audio' not in serialized and 'features_' not in serialized


def test_prepare_refuses_crop_and_train_split(tmp_path, monkeypatch):
    example, metadata, manifest, meta = media_case(tmp_path)
    monkeypatch.setattr(paired, 'contract', lambda cfg: {})
    cfg = SimpleNamespace(to_dict=lambda: {})
    Image.new('RGB', (96, 96)).save(example['observation']['listener_image'])
    with pytest.raises(ValueError, match='resize/crop'):
        paired.prepare_suite(manifest, meta, tmp_path / 'bad', cfg)
    example['split'] = 'train'
    manifest.write_text(json.dumps([example]))
    with pytest.raises(ValueError, match='Dev only'):
        paired.prepare_suite(manifest, meta, tmp_path / 'train', cfg)


def test_full_omni_plan_is_reencoded_and_never_completed_from_gt(monkeypatch):
    gt = TextPlan('GT reply', 'GT delivery', 'GT behavior', 'GT vocal')
    predicted = TextPlan('Omni reply', 'Omni delivery', 'Omni behavior', '')
    obs = Observation('dev1', 'speaker.wav', 'ref.png', 0, 2, 8)
    marker = object()
    seen = []
    monkeypatch.setattr(paired, 'prepare_observation', lambda processor, observation, cfg: marker)
    def generate(runtime, processor, prepared, max_new_tokens):
        assert prepared is marker and max_new_tokens == 512
        return predicted, predicted.serialize()
    def extract(runtime, processor, observation, plan):
        assert observation is obs
        seen.append(plan)
        return make_condition(3 if plan == gt else 5)
    monkeypatch.setattr(paired, 'generate_reply', generate)
    monkeypatch.setattr(paired, 'extract_condition', extract)
    plans, conditions, raw = paired.plan_conditions(None, None, obs, gt, None, 512)
    assert seen == [gt, predicted]
    assert plans['omni'] == predicted and plans['omni'].vocal_events == ''
    assert conditions['gt'].hidden.shape[2] == 3 and conditions['omni'].hidden.shape[2] == 5
    assert raw == predicted.serialize()


def test_invalid_omni_plan_fails_without_gt_fallback(monkeypatch):
    monkeypatch.setattr(paired, 'prepare_observation', lambda *a: {})
    def fail(*a, **kw):
        raise ValueError('Incomplete plan')
    monkeypatch.setattr(paired, 'generate_reply', fail)
    monkeypatch.setattr(paired, 'extract_condition', lambda *a: pytest.fail('Must not replace failed Omni with GT'))
    with pytest.raises(ValueError, match='Incomplete'):
        paired.plan_conditions(None, None, None, TextPlan('GT'), None, 512)


def test_paired_noise_ignores_text_length_and_prior_rng(monkeypatch):
    class Strategy:
        def prepare_training_inputs(self, text, sampler):
            torch.rand(len(text))
            return text
    monkeypatch.setattr(paired, 'with_text', lambda av, text: text)
    monkeypatch.setattr(paired, 'append_conditions', lambda prepared, u: prepared)
    monkeypatch.setattr(paired, 'sample', lambda renderer, inputs, steps: (torch.randn(5), torch.randn(7)))
    monkeypatch.setattr(paired, 'unpack_latents', lambda sampled, av: sampled)
    a = paired.sample_branch(None, Strategy(), None, None, 'short', None, 1045, 30)
    torch.rand(1000)
    b = paired.sample_branch(None, Strategy(), None, None, 'much longer text plan', None, 1045, 30)
    assert all(torch.equal(x, y) for x, y in zip(a, b))


def test_checkpoint_requires_publication_and_disjoint_dev(tmp_path):
    path = tmp_path / 'extractor_step_000100_connector.pt'
    suite = dict(contract={'new_prompt': True}, cases=[dict(sample_id='dev1')])
    with pytest.raises(ValueError, match='published'):
        paired.checkpoint_state(path, suite)
    path.with_name('extractor_step_000100.complete.json').write_text(json.dumps(dict(step=100)))
    value = dict(version=VERSION, step=100, identity=dict(contract=suite['contract'],
                 entries=[dict(sample_id='train1')]), model={'connector.weight': torch.ones(1)})
    torch.save(value, path)
    state, step = paired.checkpoint_state(path, suite)
    assert step == 100 and state['weight'].item() == 1
    value['identity']['entries'][0]['sample_id'] = 'dev1'
    torch.save(value, path)
    with pytest.raises(ValueError, match='overlaps'):
        paired.checkpoint_state(path, suite)
    value['identity']['contract'] = {'old_prompt': True}
    torch.save(value, path)
    with pytest.raises(ValueError, match='contract'):
        paired.checkpoint_state(path, suite)


def test_pair_execution_uses_matching_hidden_and_gemma_plans(tmp_path, monkeypatch):
    example, metadata, manifest, meta = media_case(tmp_path)
    monkeypatch.setattr(paired, 'contract', lambda cfg: {'prompt': 'no-time'})
    cfg = SimpleNamespace(to_dict=lambda: {})
    suite = paired.prepare_suite(manifest, meta, tmp_path / 'suite', cfg)
    case = suite['cases'][0]
    plans = dict(gt=TextPlan(**case['gt_plan']), omni=TextPlan('new words', 'new voice', 'new action', 'laugh'))
    monkeypatch.setattr(paired, 'collect_conditions', lambda *a: (
        plans, dict(gt=make_condition(3, 1), omni=make_condition(5, 2)), plans['omni'].serialize()))
    monkeypatch.setattr(paired, 'unload', lambda: None)
    monkeypatch.setattr(torch, 'autocast', lambda *a, **kw: nullcontext())
    class TextRuntime:
        def __init__(self, cfg):
            pass
        def encode(self, plan):
            return dict(prompt=plan.render_prompt())
        def close(self):
            pass
    monkeypatch.setattr(paired, 'NativeTextConditioner', TextRuntime)
    monkeypatch.setattr(paired, 'reference_latents', lambda *a, **kw: {'reference_only': True})
    monkeypatch.setattr(paired, 'load_renderer', lambda *a, **kw: (None, None, None))
    seen = []
    def branch(renderer, strategy, sampler, av, text, u, seed, steps):
        assert av == {'reference_only': True}
        seen.append((text['prompt'], u.audio.flatten()[0].item(), seed))
        return torch.ones(1), torch.ones(1)
    monkeypatch.setattr(paired, 'sample_branch', branch)
    def decode(cfg, video, audio, fps, path, device):
        path.write_bytes(b'mocked video; not a GPU generation test')
        return {k: case['geometry'][k] for k in ('frames', 'height', 'width')}
    monkeypatch.setattr(paired, 'decode_and_save', decode)
    def connector(condition):
        value = condition.hidden.flatten()[0].item()
        return SharedCondition(torch.full((1, 2, 4), value), torch.full((1, 2, 2), value),
                               condition.video_valid, condition.audio_valid)
    output = tmp_path / 'generated'
    result = paired.run_case(case, suite, cfg, connector, output, device='cpu')
    assert seen == [(plans['gt'].render_prompt(), 1, 1045), (plans['omni'].render_prompt(), 2, 1045)]
    for mode in paired.MODES:
        hidden = torch.load(output / mode / 'hidden.pt', weights_only=True)
        assert hidden['text_plan'] == asdict(plans[mode])
        assert result['modes'][mode]['gt_plan_used'] == (mode == 'gt')
        assert not result['modes'][mode]['paired_target_media_used']
    assert result['modes']['gt']['hidden_sha256'] != result['modes']['omni']['hidden_sha256']


def test_comparison_report_escapes_plan_text(tmp_path):
    folder = tmp_path / 'dev1/gt'
    folder.mkdir(parents=True)
    (folder / 'generated.mp4').write_bytes(b'placeholder')
    (folder / 'text_plan.json').write_text(json.dumps(asdict(TextPlan('<script>alert(1)</script>'))))
    paired.write_comparison_report(tmp_path, dict(cases=[dict(sample_id='dev1')]))
    text = (tmp_path / 'comparison.html').read_text()
    assert 'dev1/gt/generated.mp4' in text
    assert '<script>' not in text and '&lt;script&gt;' in text
