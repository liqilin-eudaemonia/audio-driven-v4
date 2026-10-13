"""Predeclared, conservative dev8 proxy selection; no human-quality claim."""
import math

INDICES = [4, 6, 11, 15, 22, 34, 42, 47]


def comparable(a, b):
    ignored = {'connector_initialization', 'note'}
    return {k:v for k,v in a.items() if k not in ignored} == {k:v for k,v in b.items() if k not in ignored}


def summarize(dev, samples):
    rows = [r for r in dev['rows'] if r['mode']=='beta']
    if len(rows)!=48 or len({r['sample_id'] for r in rows})!=48:
        raise ValueError('Need complete Dev48')
    if set(samples)!=set(INDICES): raise ValueError('Need matched dev8 generation and speech scores')
    values={k:sum(r['values'][k] for r in rows)/len(rows) for k in ['video','audio']}
    words=chars=we=ce=0; sync={}; bad=[]
    for i in INDICES:
        x=samples[i]; w=x['waveform']; score=x['speech']; a=score['asr']
        if not w['finite'] or w['pre_clamp_over_one_fraction']>.001 or a['rms']<1e-5: bad.append(i)
        chars+=a['reference_characters'];ce+=a['character_errors'];words+=a['reference_words'];we+=a['word_errors']
        ok=[s for s in score['syncnet'] if s.get('status')=='success']
        if len(ok)==1: sync[i]=dict(distance=ok[0]['min_dist'],confidence=ok[0]['confidence'])
    if chars<=0: raise ValueError('Missing ASR reference')
    values.update(cer=ce/chars,wer=we/max(words,1))
    if not all(math.isfinite(v) and v>=0 for v in values.values()): raise ValueError('Invalid selection metric')
    return dict(values=values,sync=sync,bad_audio=bad,samples=8)


def decide(step, warm, fresh):
    """Matched resource exposure; >=300. At500+ tie chooses warm, not a superiority claim."""
    if step<300:return dict(action='continue',reason='Minimum300updates per route')
    common=sorted(set(warm['sync']) & set(fresh['sync']))
    keys=['video','audio','cer']; av=dict(warm['values']);bv=dict(fresh['values'])
    if len(common)>=6:
        keys.append('sync_distance')
        av['sync_distance']=sum(warm['sync'][i]['distance'] for i in common)/len(common)
        bv['sync_distance']=sum(fresh['sync'][i]['distance'] for i in common)/len(common)
    # Equal-domain symmetric fractional difference. Positive favors fresh.
    delta={k:2*(av[k]-bv[k])/max(abs(av[k])+abs(bv[k]),1e-8) for k in keys}
    margin=sum(delta.values())/len(delta)
    winner='fresh' if margin>.01 else 'warm'
    healthy=[n for n,x in [('warm',warm),('fresh',fresh)] if not x['bad_audio']]
    if not healthy:return dict(action='continue',reason='Both routes have audio pathology',step=step,delta=delta)
    if len(healthy)==1:winner=healthy[0]
    direction=1 if winner=='fresh' else -1
    decisive=(len(common)>=6 and abs(margin)>=.02 and all(direction*v>=-.05 for v in delta.values()))
    # Early winner must improve both consecutive gates; controller checks prior300/400 decision.
    result=dict(action='select' if (step>=500 or decisive) else 'continue',winner=winner,
                step=step,margin=margin,delta=delta,common_sync_samples=len(common),
                decisive=decisive,reason='dev8 proxy ranking; not perceptual optimum',
                tie_prefer_warm=abs(margin)<=.01,limited_sync=len(common)<6)
    return result


def plateau(history):
    """Three consecutive checked checkpoints, <0.5% best relative objective gain."""
    if len(history)<3:return False
    a,b,c=history[-3:]
    return min(b,c)>=a*.995
