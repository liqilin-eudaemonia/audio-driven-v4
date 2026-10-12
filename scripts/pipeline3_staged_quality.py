"""Measure C0 transition gate; never manufacture a passing quality marker."""
import json
from pathlib import Path
from omni_av_p3_staged.orchestration import read,write,sha,run_pool
from omni_av_p3_staged.quality import QUALITY_VERSION, IDENTITY_DEFAULTS, identity_comparison


def selected(manifest):
 order=sorted(range(len(manifest['dev'])),key=lambda i:manifest['dev'][i]['metadata']['target_frames'])
 return [order[round(k*(len(order)-1)/7)] for k in range(8)]


def build(root,plan,cluster,checkpoint,update):
 root=Path(root);m=read(plan['manifest']);indices=selected(m);base=root/'C0_quality';base.mkdir(exist_ok=True)
 modes=['text','pilot','oracle','pred25','wrong'];jobs=[];records=[];digest=sha(plan['manifest'])
 input_digest=sha(root/'readouts/inputs/complete.json')
 for k,i in enumerate(indices):
  for mode in modes:
   cp=m['config']['initial_connector']['path'] if mode=='pilot' else str(checkpoint)
   job=dict(input_integrity_sha256=input_digest,manifest=plan['manifest'],manifest_sha256=digest,cache=str(root/'readouts'),index=i,wrong_index=indices[(k+1)%8],mode=mode,
    checkpoint=cp,checkpoint_sha256=sha(cp),negative_text=plan['negative_text'],negative_sha256=plan['negative_sha256'],seed=1045+i,
    output=str(base/'generation'/f'{mode}_{i:03d}'))
   jp=base/'jobs'/f'{mode}_{i:03d}.json';write(jp,job);records.append(job)
   jobs.append(dict(name=f'{mode}_{i:03d}',command=[cluster['python'],cluster['source']+'/scripts/pipeline3_staged_generate.py','--job',str(jp),'--gpu','{GPU}']))
 run_pool(root,cluster,jobs,base/'generation_logs',update)
 metric_plan=dict(metrics_config=plan['metrics_config'],distribution_metrics=False);write(base/'metric_plan.json',metric_plan)
 jobs=[]
 for r in records:
  e=m['dev'][r['index']];name=f"{r['mode']}_{r['index']:03d}";job=dict(name=name,kind='speech',sample_id=e['sample_id'],mode=r['mode'],
   video=str(Path(r['output'])/'generated.mp4'),reply=e['example']['text_plan']['reply'],output=str(base/'scores'/name))
  jp=base/'metric_jobs'/f'{name}.json';write(jp,job)
  jobs.append(dict(name=name,command=[cluster['python'],cluster['source']+'/scripts/pipeline21_residual_eval_metrics.py','--plan',str(base/'metric_plan.json'),'--job',str(jp),'--gpu','{GPU}']))
 for mode in modes:
  path=base/'metrics'/f'{mode}.jsonl';path.parent.mkdir(exist_ok=True)
  rows=[dict(id=m['dev'][r['index']]['sample_id'],generated_video=str(Path(r['output'])/'generated.mp4'),
         reference_video=m['dev'][r['index']]['example']['target_video']) for r in records if r['mode']==mode]
  path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
  job=dict(name=mode+'_identity',kind='visual',mode=mode,manifest=str(path),output=str(base/'scores'/f'{mode}_identity'))
  jp=base/'metric_jobs'/f'{mode}_identity.json';write(jp,job)
  jobs.append(dict(name=job['name'],command=[cluster['python'],cluster['source']+'/scripts/pipeline21_residual_eval_metrics.py','--plan',str(base/'metric_plan.json'),'--job',str(jp),'--gpu','{GPU}']))
 run_pool(root,cluster,jobs,base/'metric_logs',update)
 summary={};identities={}
 for mode in modes:
  chosen=[r for r in records if r['mode']==mode]
  scores=[read(base/'scores'/f'{mode}_{r["index"]:03d}'/'completed.json') for r in chosen]
  sync=[v for s in scores for v in s['syncnet'] if v.get('status')=='success']
  words=sum(s['asr']['reference_words'] for s in scores)
  chars=sum(s['asr']['reference_characters'] for s in scores)
  ident=read(base/'scores'/f'{mode}_identity'/'identity.json')
  identities[mode]=ident
  wave=[read(Path(r['output'])/'completed.json')['waveform'] for r in chosen]
  summary[mode]=dict(scored=len(scores),wer=sum(s['asr']['word_errors'] for s in scores)/max(1,words),
   cer=sum(s['asr']['character_errors'] for s in scores)/max(1,chars),
   sync_success=len(sync),sync_c=sum(v['confidence'] for v in sync)/len(sync) if sync else None,
   sync_d=sum(v['min_dist'] for v in sync)/len(sync) if sync else None,
   id_similarity=ident.get('id_similarity',{}).get('mean'),
   audio_over_one_fraction=sum(v['pre_clamp_over_one_fraction'] for v in wave)/len(wave),
   finite=all(v['finite'] for v in wave))
 thresholds={**IDENTITY_DEFAULTS,**plan['quality_thresholds']};baseline=summary['text'];pilot=summary['pilot'];candidate=summary['pred25'];checks={}
 checks['all_outputs_scored']=all(s['scored']==8 and s['finite'] for s in summary.values())
 checks['speech_nonregression']=candidate['wer']<=min(baseline['wer'],pilot['wer'])+thresholds['wer_tolerance']
 checks['sync_coverage']=candidate['sync_success']>=max(6,baseline['sync_success']-1)
 checks['sync_nonregression']=candidate['sync_c'] is not None and baseline['sync_c'] is not None and candidate['sync_c']>=baseline['sync_c']-thresholds['sync_c_tolerance']
 identity=identity_comparison(identities['text'],identities['pred25'],
     [m['dev'][i]['sample_id'] for i in indices],thresholds)
 checks['identity_coverage']=identity['coverage_passed']
 checks['identity_frame_coverage']=identity['frame_coverage_passed']
 checks['identity_nonregression']=identity['nonregression_passed']
 checks['audio_clipping']=candidate['audio_over_one_fraction']<=max(thresholds['clip_fraction_limit'],baseline['audio_over_one_fraction']+.001)
 report=dict(quality_version=QUALITY_VERSION,identity_comparison=identity,summary=summary,checks=checks,thresholds=thresholds,records=records,manifest_sha256=digest,checkpoint_sha256=sha(checkpoint),
  passed=all(checks.values()),note='Automatic non-regression gate, not proof of perceptual quality; wrong-condition group diagnostic only. Fixed GT replies/timelines.')
 write(base/'report.json',report);write(root/'C0_quality_gate.json',report)
 return report
