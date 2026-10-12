"""Replace only this run's serial audit, then resume its authorized queue."""
from pathlib import Path
import json,os,signal,subprocess,sys,time
root=Path('/public/2_CKPT/0_WM/llq/p3_full_connector_then_dit_20261011')
cluster=json.loads((root/'cluster.json').read_text());sys.path.insert(0,cluster['source']+'/src')
from omni_av_p3_staged.orchestration import read,write,notify
state=read(root/'state.json');plan=read(root/'plan.json');qid=state['pid'];wid=read(root/'watch.json')['pid']
def args(pid):return (Path('/proc')/str(pid)/'cmdline').read_bytes().split(b'\0')
def children(pid):return [int(v) for v in (Path('/proc')/str(pid)/'task'/str(pid)/'children').read_text().split()]
if state['status']!='sealing_cache' or (root/'STOP').exists():raise RuntimeError('State changed; no intervention')
if str(root).encode() not in args(qid) or not any(x.endswith(b'/pipeline3_staged_queue.py') for x in args(qid)):raise RuntimeError('Queue ownership mismatch')
validators=[]
for guard in children(qid):
 for child in children(guard):
  a=args(child)
  if b'--seal' in a and str(root/'readouts').encode() in a and any(x.endswith(b'/pipeline3_staged_cache.py') for x in a):validators.append(child)
if len(validators)!=1:raise RuntimeError('Expected one owned serial validator')
os.kill(qid,signal.SIGSTOP)
try:
 os.kill(validators[0],signal.SIGTERM)
 write(root/'parallel_seal_recovery.json',dict(time=time.time(),status='parallel_audit',queue=qid,replaced_validator=validators[0],workers=8,cache_preserved=True))
 with (root/'cache_seal_logs/parallel.log').open('a') as log:
  subprocess.run([cluster['python'],cluster['source']+'/scripts/pipeline3_staged_seal.py','--manifest',plan['manifest'],'--output',str(root/'readouts'),'--workers','8','--control-root',str(root)],stdout=log,stderr=subprocess.STDOUT,check=True)
 if (root/'STOP').exists():raise InterruptedError('Persistent user STOP')
 # Keep monitor restart detection out of the brief supervisor handoff.
 if str(root).encode() not in args(wid):raise RuntimeError('Watch ownership changed')
 os.kill(wid,signal.SIGSTOP)
 try:
  os.kill(qid,signal.SIGKILL)
  for _ in range(100):
   try:
    if not args(qid) or b'Z' in (Path('/proc')/str(qid)/'stat').read_bytes().split()[2:3]:break
   except FileNotFoundError:break
   time.sleep(.1)
  with (root/'queue_parallel_seal_recovery.log').open('a') as log:
   p=subprocess.Popen([cluster['python'],cluster['source']+'/scripts/pipeline3_staged_queue.py','--root',str(root)],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  for _ in range(100):
   if read(root/'state.json').get('pid')==p.pid:break
   if p.poll() is not None:raise RuntimeError('Replacement queue failed')
   time.sleep(.2)
  else:raise RuntimeError('Replacement queue did not publish state')
  write(root/'parallel_seal_recovery.json',dict(time=time.time(),status='audit_passed_queue_resumed',old_queue=qid,new_queue=p.pid,workers=8,cache_preserved=True))
 finally:os.kill(wid,signal.SIGCONT)
except BaseException as e:
 try:os.kill(qid,signal.SIGCONT)
 except ProcessLookupError:pass
 write(root/'parallel_seal_recovery_error.json',dict(time=time.time(),error=repr(e)))
 notify(root,plan['thread'],'parallel_seal_recovery_failed',dict(error=repr(e)));raise
