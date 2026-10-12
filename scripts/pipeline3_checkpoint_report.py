"""Summarize a matched checkpoint comparison and audio-specific ablations."""
import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);root=p.parse_args().root
    read=lambda p:json.loads(Path(p).read_text());plan=read(root/'plan.json')
    deadline=time.monotonic()+7200
    while not (root/'cpu_identity_completed.json').exists():
        state=read(root/'cpu_identity_state.json')
        if state['status']=='failed':raise RuntimeError(state)
        if (root/'STOP').exists():raise InterruptedError('Diagnostic STOP')
        if time.monotonic()>deadline:raise TimeoutError('Identity scoring')
        time.sleep(10)
    generation={}
    for mode in dict.fromkeys(x['mode'] for x in plan['records']):
        records=[x for x in plan['records'] if x['mode']==mode]
        scores=[read(root/'scores'/f'{mode}_{r["index"]:03d}'/'completed.json') for r in records]
        sync=[v for s in scores for v in s['syncnet'] if v.get('status')=='success']
        words=sum(s['asr']['reference_words'] for s in scores);chars=sum(s['asr']['reference_characters'] for s in scores)
        generation[mode]=dict(samples=len(records),wer=sum(s['asr']['word_errors'] for s in scores)/max(words,1),
            cer=sum(s['asr']['character_errors'] for s in scores)/max(chars,1),sync_valid=len(sync),
            sync_c_mean=statistics.mean(x['confidence'] for x in sync) if sync else None,
            sync_d_mean=statistics.mean(x['min_dist'] for x in sync) if sync else None,
            identity=read(root/'scores'/f'{mode}_identity'/'identity.json'),
            videos=[dict(index=r['index'],path=str(Path(r['output'])/'generated.mp4')) for r in records])
    dev=[]
    for step in range(100,plan['checkpoint_step']+1,100):
        path=Path(plan['parent'])/'C0_attempt1'/f'dev_{step:06d}.json'
        if not path.exists():continue
        rows=read(path)['rows']
        for mode in ['oracle','pred25']:
            group=[r for r in rows if r['mode']==mode]
            dev.append(dict(step=step,mode=mode,samples=len(group),loss=[statistics.mean(r['loss'][i] for r in group) for i in range(3)]))
    previous_report=read(Path(plan['previous'])/'report.json')
    previous=previous_report['generation']['pred25']
    previous_step=previous_report.get('checkpoint_step',int(Path(previous_report['checkpoint']).stem.split('_')[-1]))
    current_ids={r['index'] for r in plan['records'] if r['mode']=='pred25'}
    if current_ids!={v['index'] for v in previous['videos']}:raise ValueError('Checkpoint comparison sample mismatch')
    comparison=dict(previous_step=previous_step,current_step=plan['checkpoint_step'],same_samples=True,
        changes={k:dict(previous=previous[k],current=generation['pred25'][k],delta=generation['pred25'][k]-previous[k])
                 for k in ['wer','cer','sync_c_mean','sync_d_mean']})
    comparison['identity']=dict(previous=previous['identity']['id_similarity']['mean'],current=generation['pred25']['identity']['id_similarity']['mean'])
    report=dict(checkpoint=plan['checkpoint'],checkpoint_sha256=plan['checkpoint_sha256'],checkpoint_step=plan['checkpoint_step'],
        samples=len(plan['indices']),generation=generation,dev_curves=dev,
        matched_checkpoint_comparison=comparison,
        previous_report=str(Path(plan['previous'])/'report.json'),
        limitations=[f'Matched Dev8 and seeds versus step{previous_step}; this is a small single-seed comparison.',
            'Ground-truth text/timing fixed; inference does not train or switch stages.'])
    if (root/'audio_precision/completed.json').exists():
        report['audio_precision']=read(root/'audio_precision/completed.json')
        report['limitations'].extend(['Audio CFG3/5 ablations only cover P0299A; do not compare their means directly to Dev8.',
            'Precision ablations re-decode step400 audio latents; raw waveforms are retained for listening.'])
    for key,path in [('connector_stability',root/'connector_stability.json'),('audio_signal_comparison',root/'audio_precision/signal_comparison.json')]:
        if path.exists():report[key]=read(path)
    (root/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    gallery=Path(plan['gallery_source'])
    subprocess.run([sys.executable,str(gallery/'build.py')],check=True)
    subprocess.run([sys.executable,str(gallery/'prepare.py')],check=True)
    print(json.dumps(dict(report=str(root/'report.json'),samples=len(plan['indices'])),ensure_ascii=False))


if __name__=='__main__':main()
