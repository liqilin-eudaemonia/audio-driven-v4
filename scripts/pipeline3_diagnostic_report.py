"""Aggregate matched loss/gradient checks and paired generation metrics."""
import argparse
import json
from pathlib import Path
import statistics
import time


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);root=parser.parse_args().root
    read=lambda p:json.loads(p.read_text())
    plan=read(root/'plan.json');results=[read(p) for p in sorted((root/'gradients').glob('*/completed.json'))]
    if plan.get('cpu_identity_parallel'):
        deadline=time.monotonic()+7200
        while not (root/'cpu_identity_completed.json').exists():
            status=read(root/'cpu_identity_state.json')
            if status.get('status')=='failed':raise RuntimeError('CPU identity scoring failed: '+str(status))
            if (root/'STOP').exists():raise InterruptedError('Diagnostic STOP')
            if time.monotonic()>deadline:raise TimeoutError('CPU identity scoring timed out')
            time.sleep(10)
    if len(results)!=len(plan['indices']):raise ValueError('Incomplete diagnostic sample coverage')
    losses=[]
    for sigma in plan['sigmas']:
        for mode in ['text','zero','wrong','oracle','pred25']:
            pairs=[(next(x for x in r['rows'] if x['sigma']==sigma and x['mode']==mode),
                    next(x for x in r['rows'] if x['sigma']==sigma and x['mode']=='text')) for r in results]
            losses.append(dict(sigma=sigma,mode=mode,samples=len(pairs),**{
                metric:dict(mean=statistics.mean(x[metric] for x,_ in pairs),
                    paired_delta_vs_text=statistics.mean(x[metric]-base[metric] for x,base in pairs),
                    samples_better_than_text=sum(x[metric]<base[metric] for x,base in pairs))
                for metric in ['loss','video','audio']}))
    gradients=[]
    for sigma in plan['sigmas']:
        for branch in ['video','audio']:
            values=[next(x for x in r['rows'] if x['mode']=='pred25' and x['sigma']==sigma)['gradients'][branch] for r in results]
            gradients.append(dict(sigma=sigma,connector_branch=branch,samples=len(values),
                mean_video_loss_grad_norm=statistics.mean(x['video_loss_grad_norm'] for x in values),
                mean_audio_loss_grad_norm=statistics.mean(x['audio_loss_grad_norm'] for x in values),
                cosine_mean=statistics.mean(x['cosine'] for x in values if x['cosine'] is not None) if any(x['cosine'] is not None for x in values) else None,
                opposing_samples=sum(x['cosine'] is not None and x['cosine']<0 for x in values)))
    generation={}
    for mode in ['text','pilot','oracle','pred25','wrong']:
        records=[r for r in plan['records'] if r['mode']==mode]
        scores=[read(root/'scores'/f'{mode}_{r["index"]:03d}'/'completed.json') for r in records]
        identity=read(root/'scores'/f'{mode}_identity'/'identity.json')
        sync=[v for s in scores for v in s['syncnet'] if v.get('status')=='success']
        words=sum(s['asr']['reference_words'] for s in scores);chars=sum(s['asr']['reference_characters'] for s in scores)
        generation[mode]=dict(samples=len(records),wer=sum(s['asr']['word_errors'] for s in scores)/max(words,1),
            cer=sum(s['asr']['character_errors'] for s in scores)/max(chars,1),sync_valid=len(sync),
            sync_c_mean=statistics.mean(x['confidence'] for x in sync) if sync else None,
            sync_d_mean=statistics.mean(x['min_dist'] for x in sync) if sync else None,identity=identity,
            videos=[dict(index=r['index'],path=str(Path(r['output'])/'generated.mp4')) for r in records])
    report=dict(checkpoint=plan['checkpoint'],checkpoint_sha256=plan['checkpoint_sha256'],samples=len(results),
        loss_comparisons=losses,gradient_comparisons=gradients,generation=generation,
        limitations=['Dev8 diagnosis, not a population-level quality claim.',
            'Ground-truth reply and timing fixed in every condition; Oracle uses target-derived planning features.',
            'Loss diagnostics use matched noise and length-matched wrong features; generation wrong features retain source length.',
            'Gradient norms describe this checkpoint, not training outcomes after changing loss weights.',
            'No training weights or original three-node queue changed.'])
    if plan.get('performance_jobs'):
        performance=[read(p) for p in sorted((root/'performance').glob('*/completed.json'))]
        if len(performance)!=plan['performance_jobs']:raise ValueError('Incomplete performance benchmark coverage')
        report['performance']=performance
        report['kernel_screen']=[read(p) for p in sorted((root/'kernel_screen').glob('*/completed.json'))]
        report['limitations'].append('Kernel compatibility was screened on shared GPUs. Benchmarks checked GPU idleness at launch only; later competing jobs can invalidate timing. Single-GPU speedups do not establish 24-GPU gains.')
        audit_path=root/'performance_timing_audit.json'
        if audit_path.exists():
            audit=read(audit_path)
            report['performance_timing_audit']=str(audit_path)
            for result in performance:
                if result['job']['name']==audit['benchmark']:
                    result['timing_audit']=audit
                    if not audit['speed_measurement_valid']:
                        result['raw_speedup']=result.get('speedup');result['speedup']=None
                        result['note']+=' Post-run audit: candidate timing overlapped another GPU process and is invalid for speed comparisons.'
            report['limitations'].append(audit['reason'])
    (root/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(report=str(root/'report.json'),samples=len(results)),ensure_ascii=False))


if __name__=='__main__':main()
