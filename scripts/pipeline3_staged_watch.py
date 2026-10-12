"""Persistent health watch with configurable GPU sampling; respects STOP."""
import argparse,fcntl,os,sys,time,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from omni_av_p3_staged.orchestration import read,write,probe,notify,stopped
from omni_av_p3_staged.monitor import progress_issue


def alive(pid,root):
 try:
  args=Path(f'/proc/{int(pid)}/cmdline').read_bytes().split(b'\0')
  return str(root).encode() in args and any(a.endswith(b'/pipeline3_staged_queue.py') for a in args)
 except (OSError,ValueError,TypeError):return False


def utilization_issues(nodes,state,history,now,threshold=30,timeout=300):
 """Track each training GPU: a busy cluster average can hide an idle card."""
 phase=(state.get('status'),state.get('folder'),state.get('phase_started'))
 if history.get('phase')!=phase:
  history.clear();history.update(phase=phase,since={})
 if state.get('status') not in ('training','testing_24gpu','extracting_full_cache'):
  history['since'].clear();return []
 gpus=[(n['node'],g) for n in nodes if n.get('ok') for g in n.get('gpus',[])]
 low={f'{node}:{g["index"]}':dict(node=node,gpu=g['index'],util=g['util'])
      for node,g in gpus if g['util']<threshold}
 # Cache shards can finish at different times; only flag a low cluster mean.
 if state['status']=='extracting_full_cache':
  low=({'cluster':dict(mean=sum(g['util'] for _,g in gpus)/len(gpus))}
       if gpus and sum(g['util'] for _,g in gpus)/len(gpus)<threshold else {})
 history['since']={k:history['since'].get(k,now) for k in low}
 affected=[dict(**value,seconds_low=now-history['since'][k]) for k,value in low.items()
           if now-history['since'][k]>=timeout]
 return ([dict(kind='sustained_low_gpu_utilization',threshold=threshold,
               duration_seconds=timeout,affected=affected)] if affected else [])


def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--self-test',action='store_true');a=p.parse_args();root=a.root.resolve()
 plan=read(root/'plan.json');cluster=read(root/'cluster.json')
 thread=plan.get('watch_thread',plan['thread'])
 if a.self_test:
  if not notify(root,thread,'notification_self_test',dict(message='通知自检，非异常；继续本任务，不重启其他作业')):raise RuntimeError('Notification delivery failed')
  return
 lock=(root/'watch.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 record=read(root/'watch.json') if (root/'watch.json').exists() else dict(restarts=0)
 interval=max(15,int(plan.get('monitor_seconds',1200)))
 low_threshold=float(plan.get('gpu_low_util_threshold',30))
 low_seconds=max(interval,int(plan.get('gpu_low_util_seconds',300)))
 record.update(pid=os.getpid(),started=time.time(),interval_seconds=interval,
               queue_check_seconds=15,gpu_low_util_threshold=low_threshold,
               gpu_low_util_seconds=low_seconds,thread=thread,script=str(Path(__file__).resolve()))
 write(root/'watch.json',record)
 last_probe=0;alerts={};util_history={};progress_history={}
 while not stopped(root):
  try:
   state=read(root/'state.json')
   if state['status'] in ('completed','stopped'):write(root/'watch_completed.json',dict(time=time.time(),status=state['status']));return
   issues=[]
   if state['status'] in ('failed','needs_attention'):issues.append(dict(kind=state['status'],error=state.get('error',state.get('reason'))))
   elif not alive(state.get('pid'),root):
    issues.append(dict(kind='queue_missing'))
    if record['restarts']<3:
     with (root/'queue_recovery.log').open('a') as log:
      child=subprocess.Popen([cluster['python'],cluster['source']+'/scripts/pipeline3_staged_queue.py','--root',str(root)],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
     record['restarts']+=1;record['last_restart_pid']=child.pid;write(root/'watch.json',record)
   issue=progress_issue(root,state,progress_history,time.time())
   if issue:issues.append(issue)
   if time.time()-last_probe>=interval:
    nodes=probe(cluster);last_probe=time.time()
    if any(not n['ok'] for n in nodes):issues.append(dict(kind='node_unreachable',nodes=[n for n in nodes if not n['ok']]))
    issues.extend(utilization_issues(nodes,state,util_history,last_probe,low_threshold,low_seconds))
    health=dict(time=time.time(),nodes=nodes,status=state['status'],issues=issues);write(root/'health.json',health)
    with (root/'health_history.jsonl').open('a') as f:f.write(__import__('json').dumps(health)+'\n')
   for issue in issues:
    kind=issue['kind']
    if time.time()-alerts.get(kind,0)>=1200:
     with (root/'alerts.jsonl').open('a') as f:f.write(__import__('json').dumps(dict(time=time.time(),**issue))+'\n')
     if notify(root,thread,kind,issue):alerts[kind]=time.time()
  except Exception as e:write(root/'watch_error.json',dict(time=time.time(),error=repr(e)))
  time.sleep(15)
 write(root/'watch_completed.json',dict(time=time.time(),status='user_stop'))
if __name__=='__main__':main()
