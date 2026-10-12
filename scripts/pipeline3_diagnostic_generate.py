"""Run the frozen generation worker with an explicitly recorded audio CFG ablation."""
import json
from pathlib import Path
import runpy
import sys


if __name__ == '__main__':
    job = json.loads(Path(sys.argv[sys.argv.index('--job')+1]).read_text())
    source = Path(job['diagnostic_source'])
    sys.path[:0] = [str(source/'src'), str(source/'scripts')]
    import omni_av_p3_staged.sampling as sampling
    original = sampling.guided_sample
    def sample(*args, **kwargs):
        kwargs['audio_cfg'] = float(job.get('audio_cfg',7.))
        return original(*args, **kwargs)
    sampling.guided_sample = sample
    # The worker retains checkpoint/input identity checks, STOP guard and native geometry.
    runpy.run_path(str(source/'scripts/pipeline3_staged_generate.py'),run_name='__main__')
    out=Path(job['output'])/'completed.json'
    done=json.loads(out.read_text());done['sampler']=f"nativeEuler30 CFGv3/a{job.get('audio_cfg',7.)}"
    out.write_text(json.dumps(done,indent=2)+'\n')
