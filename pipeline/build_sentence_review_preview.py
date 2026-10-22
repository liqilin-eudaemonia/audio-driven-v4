"""Reproducible audit sample; never changes the source manifests or cut times."""
import argparse, concurrent.futures, gzip, hashlib, heapq, io, json, os, sqlite3, subprocess
from collections import Counter
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from math import gcd
ROOT=Path(__file__).resolve().parent
BASE=ROOT/'outputs_sentence_windows_v2'
OUT=BASE/'review_preview_v1'
REV='sentence-review-preview-v1'
SEED='20261022-review-1'
CATEGORIES={
 'end_no_quiet_cut': ['句尾没有稳定安静切口','检查末字是否完整、尾音是否突然中断；背景噪声也可能使能量规则报警。'],
 'insufficient_tail_room':['句尾余量不足','检查最后一个词后是否留有自然收尾，是否受下一次发声或 10 秒上限挤压。'],
 'start_no_quiet_cut':['句首没有稳定安静切口','检查首字、呼吸和起音是否被截断；持续背景声也可能触发。'],
 'speaker_turn_end_unconfirmed':['Speaker 末句完整性不确定','转写末尾缺少规则认可的终止标点；结合上下文判断是否完整表达。'],
 'next_voice_guard_too_close':['终点离后续发声过近','检查是否夹进下一句话的起音，或为避开下一句截掉当前尾音。'],
 'speaker_sentence_start_unconfirmed':['Speaker 句首完整性不确定','前一段转写没有终止标点，当前起点可能接在半句话中。'],
 'transition_overlap':['Speaker→Listener 轻微重叠','实质话轮的转写重叠不超过 0.2 秒；听是否自然接话或真的打断。'],
 'following_overlap':['回应与下一实质话轮轻微重叠','检查 Listener 收尾与下一话轮是否互相截断或带入下一句话。'],
 'previous_overlap':['开头与上一实质话轮轻微重叠','检查片段开始是否带入上一话轮尾音，或 Speaker 起句是否受打断。'],
 'end_crosses_annotated_voice':['终点穿过已转写发声','终点落在某个标注词内部；可能是微反应、其他发声或时间戳错误。'],
 'start_crosses_annotated_voice':['起点穿过已转写发声','起点落在某个标注词内部；检查是谁发声，以及是否真的切断。'],
 'word_timestamp_over_1_5s':['单词时间戳超过 1.5 秒','检查长词是否对应拖音、停顿或标注错位，是否影响边界判断。'],
 'speaker_pause_over_3s':['Speaker 内部停顿超过 3 秒','检查停顿是否自然、是否漏转写或错误合并了话轮。'],
 'start_preroll_tightened_without_quiet_cut':['句首余量收紧且没有安静切口','为满足时长上限压缩了开头余量；重点听第一个词有没有丢起音。'],
 'listener_pause_over_3s':['Listener 内部停顿超过 3 秒','检查回应是否仍为同一完整表达，是否有漏转写或错误合并。']}

def save(p,data):
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2));tmp.replace(p)
def run(args):
 p=subprocess.run(args,capture_output=True)
 if p.returncode:raise RuntimeError(p.stderr.decode(errors='replace')[-3500:])
 return p.stdout
def uri(ref):return f"subfile,,start,{ref['offset']},end,{ref['offset']+ref['size']},,:{ref['archive']}"
def read(ref):
 with open(ref['archive'],'rb') as f:f.seek(ref['offset']);return f.read(ref['size'])
def offer(heap,score,row):
 item=(-score,row['pair_key'],row)
 if len(heap)<180:heapq.heappush(heap,item)
 elif item[0]>heap[0][0]:heapq.heapreplace(heap,item)
def select():
 OUT.mkdir(parents=True,exist_ok=True)
 pools={};counts=Counter();n=0
 for line in open(BASE/'kept.jsonl'):
  r=json.loads(line);n+=1
  h=int(hashlib.sha256((SEED+r['pair_key']).encode()).hexdigest()[:16],16)
  if r['status']=='clean':
   group=f"clean_{int(r['speaker_trimmed'])}{int(r['response_prefix'])}";counts[group]+=1
   offer(pools.setdefault(group,[]),h,r)
  else:
   for flag in r['review_flags']:
    if flag not in CATEGORIES:raise ValueError('New category '+flag)
    counts[flag]+=1
    offer(pools.setdefault(flag+'_random',[]),h,r)
    offer(pools.setdefault(flag+'_isolated',[]),len(r['review_flags'])*(1<<64)+h,r)
 rows=[];seen=set();convs=set()
 def take(pool,num,group,method):
  got=0
  for _,_,source in sorted(pools[pool],reverse=True):
   conv=f"{source['label']}/{source['split']}/{source['interaction']}"
   if source['pair_key'] in seen or conv in convs:continue
   r=dict(source);r.update(id=f'{len(rows)+1:03d}',sample_group=group,sampling_method=method)
   rows.append(r);seen.add(r['pair_key']);convs.add(conv);got+=1
   if got==num:break
  assert got==num,(pool,got)
 for combo,number in [('00',8),('01',8),('10',7),('11',7)]:take('clean_'+combo,number,'clean','按裁剪组合分层随机：'+combo)
 for flag in CATEGORIES:
  take(flag+'_isolated',3,flag,'优先伴随标记较少，再随机')
  take(flag+'_random',2,flag,'该类内随机')
 assert len(rows)==105
 save(OUT/'selection.json',dict(revision=REV,seed=SEED,population_rows=n,population_counts=dict(counts),categories=CATEGORIES,rows=rows))
 print('Selected',len(rows),'unique conversations',dict(Counter(r['label']+'/'+r['split'] for r in rows)),flush=True)

def enrich():
 data=json.loads((OUT/'selection.json').read_text());db=sqlite3.connect('file:'+str(ROOT/'outputs_current_full_v1/metadata.sqlite')+'?mode=ro',uri=True)
 from preview_turn_rules_v2 import raw_utterances
 from preview_listen_response_v3 import floors_from_words
 for r in data['rows']:
  people={}
  for role in ['speaker','listener']:
   key=f"{r['label']}/{r['split']}/{r[role+'_id']}"
   people[role]=json.loads(gzip.decompress(db.execute('SELECT payload FROM people WHERE key=?',(key,)).fetchone()[0]))
  limit=min(min(p['video_duration'],p['wav']['duration']) for p in people.values())
  start=max(0,r['start_s']-5);end=min(limit,r['end_s']+5)
  # Extend display context to nearby utterance boundaries only. No change to target.
  utterances=[u for role,p in people.items() for u in raw_utterances(p['words'],role)]
  for u in utterances:
   if u['start']<start<u['end'] and u['start']>=r['start_s']-12:start=u['start']
   if u['start']<end<u['end'] and u['end']<=r['end_s']+12:end=u['end']
  start=max(0,start);end=min(limit,end)
  r['context_start_s']=round(start,6);r['context_end_s']=round(end,6)
  r['context_before_s']=round(r['start_s']-start,6);r['context_after_s']=round(end-r['end_s'],6)
  r['context_words']={role:[w for w in p['words'] if w[1]>=start and w[0]<=end] for role,p in people.items()}
  r['context_preview']=f"media/{r['id']}/context.mp4";r['pair_preview']=f"media/{r['id']}/pair.mp4"
  evidence=[]
  for role,ww in r['context_words'].items():
   for a,b,t in ww:
    for bound in ['start_s','end_s']:
     if a<r[bound]-1e-6 and b>r[bound]+1e-6:evidence.append(dict(kind=bound+'_crossing',role=role,start=a,end=b,text=t))
    if b-a>1.5 and a<r['response_word_end_s'] and b>r['speaker_word_start_s']:evidence.append(dict(kind='long_word',role=role,start=a,end=b,text=t))
   for u,v in zip(r[role+'_words'],r[role+'_words'][1:]):
    if v[0]-u[1]>3:evidence.append(dict(kind='pause',role=role,start=u[1],end=v[0],text=u[2]+' → '+v[2]))
  fs,_=floors_from_words({r[role+'_person']:p['words'] for role,p in people.items()})
  idx=next(i for i,f in enumerate(fs) if f['turn_id']==r['speaker_turn'])
  for j in range(max(0,idx-1),min(len(fs),idx+3)):
   f=fs[j];evidence.append(dict(kind='original_floor',role='speaker' if f['who']==r['speaker_person'] else 'listener',start=f['start'],end=f['end'],text=f.get('text',''),turn=f['turn_id']))
  r['evidence']=evidence
 save(OUT/'manifest.json',data);db.close();print('Context enriched',flush=True)

def audio_slice(ref,start,end,target_sr=48000):
 with sf.SoundFile(io.BytesIO(read(ref))) as f:
  sr=f.samplerate;f.seek(round(start*sr));a=f.read(round((end-start)*sr),dtype='float32',always_2d=True).mean(axis=1)
 if sr!=target_sr:a=resample_poly(a,target_sr//gcd(sr,target_sr),sr//gcd(sr,target_sr))
 length=round((end-start)*target_sr)
 return np.pad(a,(0,max(0,length-len(a))))[:length]

def render(r):
 folder=OUT/'media'/r['id'];folder.mkdir(parents=True,exist_ok=True)
 if (folder/'done.json').exists():return r['id'],'cached'
 a=r['context_start_s'];b=r['context_end_s'];duration=b-a
 arrays=[audio_slice(r['source'][role]['refs']['wav'],a,b) for role in ['speaker','listener']]
 wav=np.stack(arrays,axis=1);sf.write(folder/'context.wav',wav,48000,subtype='FLOAT')
 streams=[]
 for role in ['speaker','listener']:
  probe=json.loads(run(['ffprobe','-v','error','-select_streams','v:0','-show_streams','-of','json',uri(r['source'][role]['refs']['mp4'])]))['streams'][0]
  sar=probe.get('sample_aspect_ratio','1:1');sn,sd=map(float,sar.split(':')) if ':' in sar else (1,1)
  ratio=probe['width']*(sn/sd if sd and sn else 1)/probe['height']
  w=min(512,int(720*ratio)//2*2);h=min(720,int(512/ratio)//2*2)
  streams.append((max(2,w),max(2,h)))
 args=['ffmpeg','-v','error','-y','-filter_complex_threads','1']
 for role in ['speaker','listener']:args+=['-threads','1','-ss',str(a),'-i',uri(r['source'][role]['refs']['mp4'])]
 args+=['-i',str(folder/'context.wav')]
 graph=';'.join(f'[{i}:v]setpts=PTS-STARTPTS,fps=25,scale={w}:{h},setsar=1,pad=512:720:(ow-iw)/2:(oh-ih)/2:black[v{i}]' for i,(w,h) in enumerate(streams))+';[v0][v1]hstack=inputs=2[v]'
 args+=['-filter_complex',graph,'-map','[v]','-map','2:a','-t',str(duration),'-c:v','libx264','-preset','veryfast','-crf','22','-threads','2','-pix_fmt','yuv420p','-c:a','aac','-b:a','192k','-movflags','+faststart',str(folder/'context.mp4')]
 run(args)
 offset=r['start_s']-a;n=round(r['duration_s']*48000);first=round(offset*48000)
 sf.write(folder/'pair.wav',wav[first:first+n],48000,subtype='FLOAT')
 run(['ffmpeg','-v','error','-y','-threads','1','-ss',str(offset),'-i',str(folder/'context.mp4'),'-i',str(folder/'pair.wav'),'-map','0:v:0','-map','1:a:0','-t',str(r['duration_s']),'-c:v','libx264','-preset','veryfast','-crf','22','-threads','2','-pix_fmt','yuv420p','-c:a','aac','-b:a','192k','-movflags','+faststart',str(folder/'pair.mp4')])
 save(folder/'done.json',dict(revision=REV,pair_key=r['pair_key'],start_s=r['start_s'],end_s=r['end_s'],context_start_s=a,context_end_s=b,display_sizes=streams,source_levels_preserved=True,stereo_channels=['speaker','listener'],audio_peak=np.max(np.abs(wav),axis=0).tolist()))
 (folder/'context.wav').unlink();(folder/'pair.wav').unlink()
 return r['id'],'rendered'

def main():
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['select','enrich','render']);ap.add_argument('--workers',type=int,default=6);ap.add_argument('--limit',type=int);args=ap.parse_args()
 if args.stage=='select':select()
 elif args.stage=='enrich':enrich()
 else:
  rows=json.loads((OUT/'manifest.json').read_text())['rows'][:args.limit];errors=[]
  with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
   jobs={ex.submit(render,r):r['id'] for r in rows}
   for i,f in enumerate(concurrent.futures.as_completed(jobs),1):
    try:print(i,len(rows),f.result(),flush=True)
    except Exception as e:errors.append([jobs[f],str(e)]);print('FAILED',jobs[f],str(e),flush=True)
  save(OUT/'render_status.json',dict(total=len(rows),errors=errors,complete=not errors))
  if errors:raise SystemExit(1)
if __name__=='__main__':main()
