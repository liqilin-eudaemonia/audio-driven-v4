"""Owned warm/fresh 12+12 GPU training, async paired scoring, 24-GPU A transition."""
import argparse,copy,fcntl,html,json,math,os,signal,subprocess,sys,time
from pathlib import Path
S=Path(__file__).resolve().parents[1];sys.path[:0]=[str(S/'src'),str(S/'scripts')]
from omni_av_p3_staged.orchestration import read,write,sha,probe,free_slots,transport,finish_owned
from pipeline3_ab_policy import INDICES,comparable,summarize,decide,plateau

class Controller:
 def __init__(self,root):
  self.r=root;self.plan=read(root/'plan.json');self.cluster=read(root/'cluster.json');self.py=self.cluster['python']
  sys.path.insert(0,str(root/'code'));from online_notify import notify
  self.notify=notify;self.procs={};self.evals={};self.lease=None;self.last_probe=0;self.last_gallery=0;self.last_notice={};self.low={}
  self.db=read(root/'controller.json') if (root/'controller.json').exists() else dict(stage='C0',routes={n:dict(step=0,attempt=0,retries=0,status='ready') for n in ['warm','fresh']},tasks={},decision=None)
  self.manifest=read(self.plan['manifest']);self.negative=read('/public/2_CKPT/0_WM/llq/p3_a_step700_compare_4gpu_20261013/jobs/pred25_004.json')
  self.external=copy.deepcopy(self.cluster);self.external['nodes']=[dict(name='eval194',host='172.29.104.194',gpus=list(range(8)),control_path='/tmp/p3_eval_194_20261013/control',local=False)]
  assert comparable(read(root/'warm/config.json'),read(root/'fresh/config.json'))
  for route in self.db['routes'].values():
   if route['status']=='training':route['status']='ready'
  for t in self.db['tasks'].values():
   if t['status']=='running':t['status']='pending'
 def event(self,kind,**kw):
  with (self.r/'events.jsonl').open('a') as f:f.write(json.dumps(dict(time=time.time(),kind=kind,**kw))+'\n')
 def alert(self,kind,**kw):
  if time.time()-self.last_notice.get(kind,0)<1200:return
  self.last_notice[kind]=time.time();self.event(kind,**kw)
  with (self.r/'alerts.jsonl').open('a') as f:f.write(json.dumps(dict(time=time.time(),kind=kind,**kw))+'\n')
  self.notify(self.r,self.plan['thread'],kind,kw)
 def persist(self):
  write(self.r/'controller.json',self.db)
  write(self.r/'state.json',dict(pid=os.getpid(),time=time.time(),status='training' if self.procs else 'evaluating' if self.evals else 'scheduling',stage=self.db['stage'],routes=self.db['routes'],decision=self.db['decision'],evaluation_active=len(self.evals),evaluation_pending=sum(t['status']=='pending' for t in self.db['tasks'].values()),evaluation_lease=self.lease))
 def checkpoints(self,name):
  found={}
  for p in sorted((self.r/name).glob('attempt*/step_*.complete.json')):
   d=read(p);cp=p.with_suffix('').with_suffix('.pt')
   if cp.exists():found[d['step']]=cp
  return found
 def checkpoint(self,cp):
  d=read(cp.with_suffix('.complete.json'))
  if sha(cp)!=d['sha256']:raise ValueError('Checkpoint hash mismatch')
  return d
 def refresh_route(self,name):
  route=self.db['routes'][name];cps=self.checkpoints(name)
  if cps:
   step=max(cps);self.checkpoint(cps[step]);route.update(step=step,resume=str(cps[step]))
 def start_train(self,name):
  route=self.db['routes'][name]
  if route['status']=='quarantined':return
  self.refresh_route(name)
  target=1000 if name=='A' else 2000
  if route['step']>=target:return
  own=read(self.r/name/'cluster.json')
  health=probe(own)
  if len(free_slots(own,health))!=sum(len(n['gpus']) for n in own['nodes']):
   route['status']='waiting_owned_gpus';return
  i=route['attempt']
  while (self.r/name/f'attempt{i:03d}_agents').exists() or (self.r/name/f'attempt{i:03d}').exists():i+=1
  route['attempt']=i+1;folder=self.r/name/f'attempt{i:03d}'
  cfg=read(self.r/name/'config.json');cl=self.r/name/'cluster.json'
  initial=self.plan['selected_p'] if name!='A' else self.db['decision']['selected']
  args=['--manifest',self.plan['manifest'],'--integrity-root',self.plan['integrity_root'],'--config',str(self.r/name/'config.json'),'--stage','A' if name=='A' else 'C0','--output',str(folder),'--control-root',str(self.r),'--initialization-sha256',initial['sha256'],'--resume' if route.get('resume') else '--initialize',route.get('resume',initial['checkpoint']),'--stop-after',str(min(target,(route['step']//100+1)*100))]
  cmd=[self.py,str(S/'scripts/pipeline3_online_cluster.py'),'--cluster',str(cl),'--log-dir',str(folder)+'_agents','--',*args]
  guard=[self.py,str(S/'scripts/pipeline3_staged_worker.py'),'guard','--',*cmd]
  log=(folder.with_name(folder.name+'_launcher.log')).open('a');proc=subprocess.Popen(guard,stdin=subprocess.PIPE,stdout=log,stderr=subprocess.STDOUT)
  self.procs[name]=dict(process=proc,log=log,start=time.time(),folder=folder,last_progress=time.time())
  route.update(status='training',folder=str(folder),started=time.time());write(folder.with_name(folder.name+'_command.json'),dict(command=cmd,pid=proc.pid))
  self.event('train_started',route=name,step=route['step'],folder=str(folder),pid=proc.pid)
 def poll_training(self):
  for name,item in list(self.procs.items()):
   p=item['process'];f=item['folder'];latest=max((x.stat().st_mtime for x in f.glob('progress_rank*.json')),default=item['start']) if f.exists() else item['start']
   if time.time()-latest>1800 and p.poll() is None:
    self.alert('training_stalled',route=name,folder=str(f))
    if not p.stdin.closed:p.stdin.close()
   rc=p.poll()
   if rc is None:continue
   if not p.stdin.closed:p.stdin.close()
   item['log'].close();del self.procs[name];route=self.db['routes'][name];self.refresh_route(name)
   done=read(f/'completed.json') if (f/'completed.json').exists() else None
   if rc==0 and done and not done['engineering']:
    route.update(status='ready',retries=0);self.event('train_block_completed',route=name,step=done['step'])
   else:
    logs='\n'.join(x.read_text(errors='replace')[-6000:] for x in f.with_name(f.name+'_agents').glob('*.log'))
    fatal=any(x in logs for x in ['Nonfinite online','Frozen base changed','identity mismatch','Checkpoint hash mismatch','CUDA out of memory','Ranks disagree'])
    route['retries']+=1;route['status']='quarantined' if fatal or route['retries']>3 else 'ready'
    self.alert('train_retry' if route['status']=='ready' else 'train_quarantined',route=name,rc=rc,resume=route.get('resume'),error=logs[-2500:])
 def enqueue(self):
  names=['A'] if self.db['stage'] in ('A','completed') else ['warm','fresh']
  for name in names:
   for step,cp in self.checkpoints(name).items():
    if step<=0 or step%100:continue
    label=f'{name}_{step:06d}';folder=self.r/'evaluation'/label
    if (folder/'jobs.json').exists():jobs=read(folder/'jobs.json')
    else:
     mark=self.checkpoint(cp);jobs=[]
     for idx in INDICES:
      out=folder/f'pred25_{idx:03d}';job=dict(version='p3.bernini-beta.v2',manifest=self.plan['manifest'],manifest_sha256=self.plan['manifest_sha256'],integrity_root=self.plan['integrity_root'],input_integrity_sha256=sha(Path(self.plan['integrity_root'])/'complete.json'),checkpoint=str(cp),checkpoint_sha256=mark['sha256'],negative_text=self.negative['negative_text'],negative_sha256=self.negative['negative_sha256'],index=idx,mode='pred25',seed=1045+idx,output=str(out))
      jp=folder/'jobs'/f'pred25_{idx:03d}.json';write(jp,job);jobs.append(str(jp))
     write(folder/'jobs.json',jobs)
    for jp in jobs:
     job=read(jp);idx=job['index'];out=Path(job['output']);key=label+f'/{idx}/generate';metric_key=label+f'/{idx}/speech'
     self.db['tasks'].setdefault(key,dict(kind='generate',job=jp,status='pending',retries=0,label=label,step=step,route=name))
     t=self.db['tasks'][key]
     if (out/'completed.json').exists() and t['status']!='done':
      d=read(out/'completed.json')
      if d['job']!=job or d['video_sha256']!=sha(out/'generated.mp4'):raise ValueError('Generated identity mismatch')
      t['status']='done'
     if t['status']=='done':
      mj=folder/'metric_jobs'/f'{idx:03d}.json'
      if not mj.exists():
       e=self.manifest['dev'][idx];reply=e['example']['text_plan']['reply_text'] if 'reply_text' in e['example']['text_plan'] else e['example']['text_plan']['reply']
       write(mj,dict(name=f'{label}_{idx}',kind='speech',sample_id=e['sample_id'],mode='pred25',video=str(out/'generated.mp4'),reply=reply,output=str(folder/'scores'/f'{idx:03d}')))
      self.db['tasks'].setdefault(metric_key,dict(kind='speech',job=str(mj),status='pending',retries=0,label=label,step=step,route=name))
      mt=self.db['tasks'][metric_key];md=Path(read(mj)['output'])/'completed.json'
      if md.exists():mt['status']='done'
 def poll_eval(self):
  for key,item in list(self.evals.items()):
   p=item['process']
   if time.time()-item['start']>7200 and p.poll() is None and not p.stdin.closed:p.stdin.close()
   rc=p.poll()
   if rc is None:continue
   if not p.stdin.closed:p.stdin.close()
   item['log'].close();del self.evals[key];t=self.db['tasks'][key];out=Path(read(t['job'])['output'])
   if rc==0 and (out/'completed.json').exists():t['status']='done';self.event('eval_completed',task=key)
   else:
    log=Path(item['log_path']).read_text(errors='replace')[-2500:]
    if 'Selected GPU is occupied' in log or 'GPU is not idle' in log:t['status']='pending';continue
    t['retries']+=1;t['status']='pending' if t['retries']<=2 else 'failed'
    if t['status']=='failed' and t['kind']=='speech' and (out/'asr.json').exists() and 'source changed' not in log:
     ident=read(out/'identity_source.json');write(out/'completed.json',dict(ident,asr=read(out/'asr.json'),syncnet=[dict(status='error',reason='SyncNet failed after retries',log=log)],degraded=True));t['status']='done'
    self.alert('evaluation_retry',task=key,retries=t['retries'],status=t['status'],error=log)
 def pending(self):
  pending=[(k,t) for k,t in self.db['tasks'].items() if t['status']=='pending']
  return sorted(pending,key=lambda z:(0 if z[1]['step'] in [300,400,500] else 1,z[1]['step'],0 if z[1]['kind']=='speech' else 1,z[0]))
 def slots(self):
  cl=copy.deepcopy(self.external)
  if self.lease:cl['nodes']+=read(self.r/self.lease/'cluster.json')['nodes']
  elif self.db['stage']=='completed':cl['nodes']+=self.cluster['nodes']
  nodes=probe(cl);write(self.r/'evaluation_gpu_health.json',dict(time=time.time(),nodes=nodes))
  busy={(v['node'],v['gpu']) for v in self.evals.values()}
  return cl,[(n,g) for n,g in free_slots(cl,nodes) if (n['name'],g) not in busy]
 def dispatch_eval(self):
  if time.time()-self.last_probe<30:return
  self.last_probe=time.time();pending=self.pending()
  if not pending:return
  cl,slots=self.slots()
  for (key,t),(node,gpu) in zip(pending,slots):
   jp=Path(t['job']);out=Path(read(jp)['output']);out.mkdir(parents=True,exist_ok=True)
   cmd=[self.py,str(S/('scripts/pipeline3_online_generate.py' if t['kind']=='generate' else 'scripts/pipeline21_residual_eval_metrics.py'))]
   if t['kind']=='speech':cmd+=['--plan',str(self.r/'metric_plan.json')]
   cmd+=['--job',str(jp),'--gpu',str(gpu)]
   cmd=[self.py,str(S/'scripts/pipeline3_staged_worker.py'),'guard','--',*cmd]
   lp=out/(t['kind']+'.log');log=lp.open('a');proc=subprocess.Popen(transport(cl,node,cmd),stdin=subprocess.PIPE,stdout=log,stderr=subprocess.STDOUT)
   self.evals[key]=dict(process=proc,log=log,log_path=str(lp),start=time.time(),node=node['name'],gpu=gpu,lease=self.lease if node['name']!='eval194' else None)
   t.update(status='running');self.event('eval_started',task=key,node=node['name'],gpu=gpu)
 def dev_report(self,name,step):
  files=sorted((self.r/name).glob(f'attempt*/dev_{step:06d}.json'))
  return files[-1] if files else None
 def comparison(self):
  if self.db['decision'] or self.db['stage']!='C0':return
  previous=None
  for step in range(300,2001,100):
   ss={};cps={}
   for name in ['warm','fresh']:
    cp=self.checkpoints(name).get(step)
    dev=self.dev_report(name,step)
    if cp is None or dev is None:return
    folder=self.r/'evaluation'/f'{name}_{step:06d}';samples={}
    for idx in INDICES:
     gp=folder/f'pred25_{idx:03d}'/'completed.json';mp=folder/'scores'/f'{idx:03d}'/'completed.json'
     if not gp.exists() or not mp.exists():return
     samples[idx]=dict(waveform=read(gp)['waveform'],speech=read(mp))
    ss[name]=summarize(read(dev),samples);cps[name]=cp
   result=decide(step,ss['warm'],ss['fresh']);write(self.r/'comparisons'/f'{step:06d}.json',dict(result=result,summaries=ss))
   # Early selection requires a consistent decisive advantage at300 and400.
   if result['action']=='select' and (step>=500 or (previous and previous.get('decisive') and previous.get('winner')==result['winner'])):
    name=result['winner']
    # Keep training while the selected route still improves; C0 is capped at2000.
    history=[]
    for hstep in range(step-200,step+1,100):
     hp=self.dev_report(name,hstep)
     if hp is None:break
     hr=[x['values'] for x in read(hp)['rows'] if x['mode']=='beta']
     history.append(sum(x['video']+x['audio'] for x in hr)/len(hr))
    if step<2000 and not plateau(history):
     result.update(action='continue',reason='Winning route still improving; wait for three-checkpoint AV plateau',candidate_route=name)
     write(self.r/'comparisons'/f'{step:06d}.json',dict(result=result,summaries=ss));previous=result;continue
    # Choose that route's best already-scored checkpoint by fixed-baseline proxy score.
    reference=read(self.r/'comparisons'/'000300.json')['summaries']['warm']['values']
    candidates=[]
    for report in sorted((self.r/'comparisons').glob('*.json')):
     rr=read(report);hs=int(report.stem);sample=rr['summaries'][name]
     if hs>step or sample['bad_audio']:continue
     score=sum(sample['values'][k]/max(reference[k],1e-6) for k in ['video','audio','cer'])/3
     candidates.append((score,hs))
    _,chosen_step=min(candidates);cp=self.checkpoints(name)[chosen_step];marker=self.checkpoint(cp)
    result.update(transition_reason='C0 validation plateau' if step<2000 else 'C0 budget cap2000',checkpoint_candidates=candidates)
    result['selected']=dict(route=name,checkpoint=str(cp),sha256=marker['sha256'],step=chosen_step)
    self.db['decision']=result;write(self.r/'selection.json',result);self.notify(self.r,self.plan['thread'],'route_selected',result)
    # Stop at a saved update boundary; keep all rollback checkpoints.
    for item in self.procs.values():(item['folder']/'STOP').write_text('Autopilot selection complete; save and release for A.\n')
    return
   previous=result
 def transition_a(self):
  if not self.db['decision'] or self.db['stage']!='C0' or self.procs:return
  # Drain only in-flight fallback jobs before reclaiming all24 GPUs.
  if any(x['lease'] for x in self.evals.values()):return
  self.lease=None;name=self.db['decision']['selected']['route'];a=self.r/'A';a.mkdir(exist_ok=True)
  cfg=read(self.r/name/'config.json');cfg.update(world_size=24,accumulation=3)
  write(a/'config.json',cfg);write(a/'cluster.json',self.cluster)
  self.db['stage']='A';self.db['routes']['A']=dict(step=0,attempt=0,retries=0,status='ready')
  self.event('A_scheduled',selected=self.db['decision']['selected']);self.persist()
 def schedule_train(self):
  if self.db['stage']=='completed':return
  if self.db['decision'] and self.db['stage']=='C0':return
  names=['A'] if self.db['stage'] in ('A','completed') else ['warm','fresh']
  if self.lease and not any(x['lease']==self.lease for x in self.evals.values()) and not self.pending():self.lease=None
  for name in names:
   if name in self.procs or name==self.lease:continue
   route=self.db['routes'][name]
   if name=='A' and self.pending() and route['step']>=100 and self.lease is None:
    external=probe(self.external)
    if not free_slots(self.external,external):
     self.lease='A';self.event('A_checkpoint_evaluation_on_owned_gpus');continue
   if name!='A' and self.pending() and self.procs and self.lease is None and route['step']>=100:
    self.lease=name;self.event('training_gpu_evaluation_lease',route=name);continue
   self.start_train(name)
  # Never leave a group parked waiting on external-only tail jobs.
  if self.lease and not any(x['lease']==self.lease for x in self.evals.values()) and not self.pending():
   n=self.lease;self.lease=None;self.start_train(n)
 def gallery(self):
  if time.time()-self.last_gallery<60:return
  self.last_gallery=time.time();cards=[];mapping={}
  for gp in sorted((self.r/'evaluation').glob('*/pred25_*/completed.json')):
   d=read(gp);j=d['job'];label=gp.parents[1].name;idx=j['index'];key=f'p3_ab_{label}/{idx}'
   mapping[key]=str(gp.parent/'generated.mp4');score=gp.parents[1]/'scores'/f'{idx:03d}'/'asr.json';text=read(score)['text'] if score.exists() else 'ASR评测中'
   cards.append(f'<article><h3>{html.escape(label)} · dev{idx}</h3><video controls preload="none" src="/omniresponse-media/{key}"></video><p>{html.escape(text)}</p></article>')
  content='<meta charset="utf-8"><title>P3 warm vs fresh</title><style>body{font:16px sans-serif;margin:24px}main{display:grid;grid-template-columns:repeat(2,1fr);gap:20px}video{width:100%}</style><h1>P3 Beta10：生成Connector温启动／重新初始化</h1><p>同P400、同数据/种子、global72。固定GT回复与时长；Pred25/Euler30/CFG3与7。自动指标仅用于本批代理选优。</p><pre>'+html.escape(json.dumps(self.db['decision'],ensure_ascii=False,indent=2))+'</pre><main>'+''.join(cards)+'</main>'
  (self.r/'index.html.tmp').write_text(content);(self.r/'index.html.tmp').replace(self.r/'index.html')
  mp=Path('/public/2_CKPT/0_WM/llq/p2_p21_gallery_20261009/omniresponse_media.json')
  with mp.with_suffix('.lock').open('a') as lock:
   fcntl.flock(lock,fcntl.LOCK_EX);m=read(mp);m.update(mapping);m['p3_ab/0']=str(self.r/'index.html');write(mp,m)
 def run(self):
  gate=read(self.r/'engineering/completed.json');assert gate['passed']
  assert all(sha(S/name)==digest for name,digest in read(self.r/'source_manifest.json').items())
  self.event('controller_started',pid=os.getpid());self.persist()
  try:
   while not (self.r/'STOP').exists():
    self.poll_training();self.poll_eval();self.enqueue();self.comparison();self.transition_a();self.schedule_train();self.dispatch_eval();self.gallery();self.persist()
    if self.db['stage']=='A' and self.db['routes']['A']['step']>=1000 and not self.procs:
     self.db['stage']='completed';self.event('A_completed');self.notify(self.r,self.plan['thread'],'A_completed',dict(root=str(self.r)));self.persist()
     # Finish all saved-checkpoint inference, never launch B.
    if self.db['stage']=='completed' and not self.pending() and not self.evals:
     write(self.r/'completed.json',dict(time=time.time(),selection=self.db['decision']));return
    time.sleep(5)
  finally:
   # Root STOP is persistent; trainer sees it and saves. Parent death closes owned guards.
   until=time.time()+900 if (self.r/'STOP').exists() else time.time()+30
   for item in list(self.procs.values()):
    p=item['process']
    try:p.wait(timeout=max(1,until-time.time()))
    except subprocess.TimeoutExpired:
     if not p.stdin.closed:p.stdin.close()
   for item in list(self.evals.values()):
    if not item['process'].stdin.closed:item['process'].stdin.close()

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
 lock=(a.root/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 Controller(a.root).run()
