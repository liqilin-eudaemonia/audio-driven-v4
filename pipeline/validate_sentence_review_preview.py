"""Validate exact selection provenance and all preview media without touching full output."""
import argparse, concurrent.futures, json, subprocess
from collections import Counter
from pathlib import Path
from build_sentence_review_preview import BASE,OUT,REV,save,audio_slice
import numpy as np

def run(cmd):
 p=subprocess.run(cmd,capture_output=True)
 if p.returncode:raise RuntimeError(p.stderr.decode(errors='replace')[-3000:])
 return p.stdout

def selection():
 data=json.loads((OUT/'manifest.json').read_text());rows=data['rows'];assert len(rows)==105
 assert len({r['pair_key'] for r in rows})==105
 assert len({(r['label'],r['split'],r['interaction']) for r in rows})==105
 counts=Counter(r['sample_group'] for r in rows);assert counts.pop('clean')==30;assert set(counts)==set(data['categories']);assert set(counts.values())=={5}
 found=set();lookup={r['pair_key']:r for r in rows}
 for line in open(BASE/'kept.jsonl'):
  source=json.loads(line)
  if source['pair_key'] not in lookup:continue
  r=lookup[source['pair_key']]
  assert all(r[k]==v for k,v in source.items()),r['pair_key'];found.add(r['pair_key'])
 assert len(found)==105
 for r in rows:
  assert r['duration_s']<=9.95+1e-6
  assert r['context_start_s']<=r['start_s']<r['end_s']<=r['context_end_s']
  if r['sample_group']=='clean':assert r['status']=='clean' and not r['review_flags']
  else:assert r['sample_group'] in r['review_flags'] and r['status']=='review'
  assert r['audio_normalized'] is False and r['visual_filtered'] is False
 result=dict(passed=True,source_manifest_exact_match=105,unique_pairs=105,unique_conversations=105,clean=30,review_groups=dict(counts),source_subsets=dict(Counter(r['label']+'/'+r['split'] for r in rows)),with_micro_reactions=sum(bool(r['micro_reactions']) for r in rows),clean_with_micro_reactions=sum(bool(r['micro_reactions']) and r['status']=='clean' for r in rows),clean_strata=dict(Counter(str(int(r['speaker_trimmed']))+str(int(r['response_prefix'])) for r in rows if r['status']=='clean')),contexts_less_than_5s=[dict(id=r['id'],before=r['context_before_s'],after=r['context_after_s']) for r in rows if min(r['context_before_s'],r['context_after_s'])<5-1e-5])
 save(OUT/'selection_validation.json',result);print('PASS selection: 105 exact rows from latest full manifest',flush=True)

def media_one(r):
 folder=OUT/'media'/r['id'];done=json.loads((folder/'done.json').read_text());assert done['revision']==REV and done['pair_key']==r['pair_key']
 assert done['start_s']==r['start_s'] and done['end_s']==r['end_s'];result={'id':r['id']}
 for kind,duration in [('context',r['context_end_s']-r['context_start_s']),('pair',r['duration_s'])]:
  p=folder/(kind+'.mp4');meta=json.loads(run(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(p)]))
  vv=next(s for s in meta['streams'] if s['codec_type']=='video');aa=next(s for s in meta['streams'] if s['codec_type']=='audio')
  assert (vv['width'],vv['height'])==(1024,720)
  assert aa['channels']==2 and int(aa['sample_rate'])==48000
  actual=float(meta['format']['duration']);assert abs(actual-duration)<.085,(r['id'],kind,actual,duration)
  assert abs(float(aa['duration'])-duration)<.025,(r['id'],kind,'audio duration')
  if kind=='pair':assert actual<=10
  run(['ffmpeg','-v','error','-xerror','-threads','1','-i',str(p),'-map','0:v:0','-map','0:a:0','-f','null','-'])
  result[kind+'_duration']=actual
 return result

def audio_check(r):
 a=r['start_s'];b=r['end_s'];raw=np.stack([audio_slice(r['source'][role]['refs']['wav'],a,b) for role in ['speaker','listener']],axis=1)
 decoded=np.frombuffer(run(['ffmpeg','-v','error','-threads','1','-i',str(OUT/'media'/r['id']/'pair.mp4'),'-map','0:a','-f','f32le','-']),dtype='<f4').reshape(-1,2)
 n=min(len(raw),len(decoded));raw=raw[:n];decoded=decoded[:n];channels=[]
 for ch in range(2):
  x=raw[:,ch].astype('float64');y=decoded[:,ch].astype('float64');power=float(np.dot(x,x));corr=float(np.dot(x,y)/np.sqrt(power*np.dot(y,y)));gain=float(np.sqrt(np.dot(y,y)/power))
  assert corr>.95 and .85<gain<1.15,(r['id'],ch,corr,gain)
  channels.append(dict(channel=ch,role=['speaker','listener'][ch],correlation=corr,rms_gain=gain))
 return dict(id=r['id'],channels=channels)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['selection','media']);args=ap.parse_args()
 if args.stage=='selection':selection()
 else:
  rows=json.loads((OUT/'manifest.json').read_text())['rows']
  with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:details=list(ex.map(media_one,rows))
  audio=[audio_check(rows[i]) for i in [0,29,54,79,104]]
  save(OUT/'validation.json',dict(passed=True,clips=210,full_decode=True,video_dimensions=[1024,720],audio_channels=['speaker','listener'],preview_video_tolerance_seconds=.085,manifest_cuts_unchanged=True,audio_source_comparisons=audio,media=details))
  print('PASS media: 210 full decodes; duration/channel checks; 5 raw-source audio comparisons')
