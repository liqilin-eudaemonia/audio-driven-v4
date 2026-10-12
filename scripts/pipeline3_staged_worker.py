"""Tiny stdlib process guard and strict idle-GPU probe for the P3 queue."""
import argparse
import json
import os
import select
import signal
import subprocess
import sys


def probe(gpus):
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu,power.draw,temperature.gpu','--format=csv,noheader,nounits'],text=True,timeout=25)
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True,timeout=25)
    processes={}
    for line in apps.splitlines():
        uid,pid=[x.strip() for x in line.split(',')];processes.setdefault(uid,[]).append(int(pid))
    rows=[]
    for line in raw.splitlines():
        i,uid,mem,util,power,temp=[x.strip() for x in line.split(',')]
        if int(i) in gpus:rows.append(dict(index=int(i),memory_mib=float(mem),util=float(util),power=power,temperature=temp,pids=processes.get(uid,[])))
    if {r['index'] for r in rows}!=set(gpus):raise ValueError('Selected GPU absent')
    return dict(gpus=rows)


def guard(command):
    process=subprocess.Popen(command,start_new_session=True)
    def interrupted(*_):raise InterruptedError('Guard interrupted')
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    try:
        while process.poll() is None:
            ready,_,_=select.select([sys.stdin.fileno()],[],[],.5)
            if ready and not os.read(sys.stdin.fileno(),4096):break
        if process.poll() is not None:return process.returncode
    finally:
        if process.poll() is None:
            os.killpg(process.pid,signal.SIGTERM)
            try:process.wait(timeout=20)
            except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
    return process.returncode


def main():
    if sys.argv[1:2]==['guard']:
        args=sys.argv[2:];args=args[1:] if args[:1]==['--'] else args
        raise SystemExit(guard(args))
    p=argparse.ArgumentParser();p.add_argument('action',choices=['probe','idle']);p.add_argument('--gpus',required=True);a=p.parse_args()
    result=probe([int(g) for g in a.gpus.split(',')])
    if a.action=='idle' and any(g['pids'] or g['memory_mib']>=1000 or g['util']>5 for g in result['gpus']):raise RuntimeError('GPU occupied; not touching other jobs')
    print(json.dumps(result))
if __name__=='__main__':main()
