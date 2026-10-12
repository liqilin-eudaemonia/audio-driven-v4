import importlib.util
import io
from pathlib import Path

spec=importlib.util.spec_from_file_location('diagnostic_queue',Path(__file__).resolve().parents[1]/'scripts/pipeline3_diagnostic_queue.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_connection_retry_does_not_abort_other_jobs(tmp_path,monkeypatch):
    import omni_av_p3_staged.orchestration as orchestration
    now=[0];calls=[];finishes=[]
    class Process:
        def __init__(self,failed):self.failed=failed;self.polls=0;self.stdin=io.StringIO()
        def poll(self):
            self.polls+=1
            if self.failed:return 255
            return 0 if self.polls>=5 else None
    def launch(command,stdin,stdout,stderr):
        name=command[-1];calls.append(name);failed=name=='retry' and calls.count(name)==1
        if failed:stdout.write('kex_exchange_identification: Connection closed by remote host\n');stdout.flush()
        return Process(failed)
    monkeypatch.setattr(module.subprocess,'Popen',launch)
    monkeypatch.setattr(module.time,'time',lambda:now[0])
    monkeypatch.setattr(module.time,'sleep',lambda t:now.__setitem__(0,now[0]+16))
    monkeypatch.setattr(orchestration,'probe',lambda c:[])
    monkeypatch.setattr(orchestration,'free_slots',lambda c,s:[(dict(name='node'),0),(dict(name='node'),1)])
    monkeypatch.setattr(orchestration,'transport',lambda c,n,cmd:cmd)
    monkeypatch.setattr(orchestration,'finish_owned',lambda tasks:finishes.append([t['name'] for t in tasks]))
    module.diagnostic_pool(tmp_path,dict(python='python',source='source'),
        [dict(name='retry',command=['retry']),dict(name='healthy',command=['healthy'])],tmp_path/'logs',lambda **kw:None)
    assert calls.count('healthy')==1 and calls.count('retry')==2
    assert finishes==[[]]


def test_only_bounded_transport_errors_are_retryable():
    assert module.retryable_connection_failure(255,'kex_exchange_identification: Connection closed',0)
    assert not module.retryable_connection_failure(255,'kex_exchange_identification: Connection closed',3)
    assert not module.retryable_connection_failure(1,'CUDA out of memory',0)
    assert not module.retryable_connection_failure(255,'Model error',0)
